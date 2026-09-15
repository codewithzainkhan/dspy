"""A local web app for live-demoing how DSPy compiles and optimizes a prompt.

Run it with:
    .\\.venv\\Scripts\\python.exe app.py
then open http://127.0.0.1:5000 in a browser.

Everything here is a REAL dspy call against a real LM (your OpenAI key) -
nothing is pre-recorded. See README_WEBAPP.md for the full guide.
"""

import csv
import io
import json
import logging
import os
import re
import threading
import time
from contextlib import redirect_stderr, redirect_stdout
from typing import Literal

import dspy
from flask import Flask, Response, jsonify, render_template, request

import common

app = Flask(__name__)

STATE_LOCK = threading.Lock()


class Trace:
    """Records real wall-clock time between real steps - not fabricated delays."""

    def __init__(self):
        self.events = []
        self._last = time.perf_counter()

    def step(self, msg: str):
        now = time.perf_counter()
        self.events.append({"msg": msg, "ms": round((now - self._last) * 1000, 2)})
        self._last = now
        return self.events[-1]

    def list(self):
        return self.events


def fresh_job():
    return {
        "status": "idle",  # idle | running | done | error
        "log": [],
        "trials": [],
        "before": None,
        "after": None,
        "instructions": None,
        "n_demos": None,
        "error": None,
    }


STATE = {
    "configured": False,
    "auto_configured": False,
    "model": None,
    "train": None,
    "dev": None,
    "signature": None,
    "module": "cot",
    "baseline": None,
    "optimized": None,
    "optimize_job": fresh_job(),
}


def _auto_configure_from_env():
    """If OPENAI_API_KEY is already set in the environment (e.g. a host's secret),
    connect automatically at process startup so visitors never see a key field.

    Deliberately no rate limiting or spend cap here - the deployment owner made
    that call explicitly. If you're reading this before deploying somewhere
    public: every visitor's clicks spend the key below. Rotate it immediately if
    the URL ever leaks past who you intended to share it with.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return
    model = os.environ.get("DSPY_DEMO_MODEL", "openai/gpt-5.4-nano")
    try:
        lm = dspy.LM(model, cache=True)
        dspy.configure(lm=lm)
        STATE["model"] = model
        STATE["configured"] = True
        STATE["auto_configured"] = True
        print(f"[startup] Auto-configured from OPENAI_API_KEY - model={model}")
    except Exception as e:
        print(f"[startup] OPENAI_API_KEY was set but auto-configure failed: {e}")


_auto_configure_from_env()


# ---------------------------------------------------------------- dataset ---

def parse_dataset_text(raw_text: str, fmt: str) -> list[dict]:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        raise ValueError("Paste or upload some data first.")

    rows = []
    if fmt == "json":
        data = json.loads(raw_text)
        if not isinstance(data, list):
            raise ValueError("JSON must be an array of objects, e.g. [{...}, {...}].")
        for i, obj in enumerate(data):
            if not isinstance(obj, dict):
                raise ValueError(f"Row {i + 1} is not an object.")
            for key in ("ticket", "category", "urgency"):
                if key not in obj:
                    raise ValueError(f"Row {i + 1} is missing \"{key}\".")
            rows.append({
                "ticket": str(obj["ticket"]).strip(),
                "category": str(obj["category"]).strip(),
                "urgency": str(obj["urgency"]).strip(),
            })
    else:
        reader = csv.DictReader(io.StringIO(raw_text))
        fieldnames = {f.strip() for f in (reader.fieldnames or [])}
        required = {"ticket", "category", "urgency"}
        if not required.issubset(fieldnames):
            raise ValueError(
                "CSV header must include the columns: ticket, category, urgency "
                f"(found: {', '.join(sorted(fieldnames)) or 'none'})."
            )
        for i, row in enumerate(reader):
            if not (row.get("ticket") or "").strip():
                continue
            rows.append({
                "ticket": row["ticket"].strip(),
                "category": (row.get("category") or "").strip(),
                "urgency": (row.get("urgency") or "").strip(),
            })

    if len(rows) < 4:
        raise ValueError(f"Need at least 4 rows to make a train/dev split - got {len(rows)}.")
    for i, r in enumerate(rows):
        if not r["ticket"] or not r["category"] or not r["urgency"]:
            raise ValueError(f"Row {i + 1} has an empty ticket, category, or urgency.")
    return rows


def split_rows(rows: list[dict], seed: int = 0, dev_frac: float = 0.3):
    import random
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    n_dev = max(1, round(len(shuffled) * dev_frac))
    n_dev = min(n_dev, len(shuffled) - 1)
    return shuffled[n_dev:], shuffled[:n_dev]


def rows_to_examples(rows: list[dict]) -> list["dspy.Example"]:
    return [
        dspy.Example(ticket=r["ticket"], category=r["category"], urgency=r["urgency"]).with_inputs("ticket")
        for r in rows
    ]


def build_signature(categories: list[str], urgencies: list[str], instructions: str):
    Category = Literal[tuple(categories)]
    Urgency = Literal[tuple(urgencies)]
    fields = {
        "ticket": (str, dspy.InputField(desc="The text to classify.")),
        "category": (Category, dspy.OutputField()),
        "urgency": (Urgency, dspy.OutputField()),
    }
    return dspy.Signature(fields, instructions)


MODULE_LABELS = {
    "predict": "dspy.Predict - answers directly, one LM call, no reasoning field",
    "cot": "dspy.ChainOfThought - reasons first, adds a 'reasoning' output field automatically",
}


def make_program(sig, module_name: str):
    """The execution strategy wrapped around a signature - same task, different module."""
    if module_name == "predict":
        return dspy.Predict(sig)
    return dspy.ChainOfThought(sig)


def make_metric(mode: str):
    """The 'rubric' - what counts as the model getting it right. A parameter, not a DSPy built-in."""

    def metric(example, pred, trace=None):
        category_ok = str(getattr(pred, "category", "")).strip().lower() == str(example.category).strip().lower()
        urgency_ok = str(getattr(pred, "urgency", "")).strip().lower() == str(example.urgency).strip().lower()
        if mode == "category":
            ok = category_ok
        elif mode == "urgency":
            ok = urgency_ok
        else:
            if trace is not None:
                return category_ok and urgency_ok
            return (category_ok + urgency_ok) / 2.0
        return ok if trace is not None else float(ok)

    return metric


# --------------------------------------------------------------- logging ---

TRIAL_RE = re.compile(r"^Score: ([\d.]+) with parameters (\[.*\])\.$")
DEFAULT_RE = re.compile(r"^Default program score: ([\d.]+)$")
MAX_LOG_LINES = 400


METRIC_LABELS = {
    "both": "both category and urgency must match",
    "category": "category only",
    "urgency": "urgency only",
}


def append_log(job: dict, msg: str):
    msg = msg.strip()
    if not msg:
        return
    with STATE_LOCK:
        job["log"].append(msg)

        trial_event = None
        m = DEFAULT_RE.match(msg)
        if m:
            trial_event = {"n": len(job["trials"]) + 1, "score": float(m.group(1)), "label": "default program"}
        else:
            m2 = TRIAL_RE.match(msg)
            if m2:
                trial_event = {"n": len(job["trials"]) + 1, "score": float(m2.group(1)), "label": m2.group(2)}

        if trial_event:
            job["trials"].append(trial_event)
            is_new_best = trial_event["score"] > job.get("_best", -1)
            if is_new_best:
                job["_best"] = trial_event["score"]
            friendly = (
                f"  → plain English: trial {trial_event['n']} scored {trial_event['score']:.1f}% "
                f"with {trial_event['label']}" + ("  <- new best so far" if is_new_best else "")
            )
            job["log"].append(friendly)

        if len(job["log"]) > MAX_LOG_LINES:
            del job["log"][: len(job["log"]) - MAX_LOG_LINES]


class JobLogHandler(logging.Handler):
    def __init__(self, job):
        super().__init__()
        self.job = job
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record):
        try:
            append_log(self.job, self.format(record))
        except Exception:
            pass


class JobStdWriter(io.TextIOBase):
    def __init__(self, job):
        self.job = job
        self._buf = ""

    def write(self, s):
        self._buf += s
        while "\n" in self._buf or "\r" in self._buf:
            sep = "\n" if "\n" in self._buf else "\r"
            line, self._buf = self._buf.split(sep, 1)
            if line.strip():
                append_log(self.job, line)
        return len(s)

    def flush(self):
        pass


# --------------------------------------------------------------- routes ---

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/configure", methods=["POST"])
def api_configure():
    body = request.get_json(force=True)
    api_key = (body.get("api_key") or "").strip()
    model = (body.get("model") or "openai/gpt-5.4-nano").strip()
    if not api_key:
        return jsonify({"error": "Paste an OpenAI API key first."}), 400

    os.environ["OPENAI_API_KEY"] = api_key
    try:
        lm = dspy.LM(model, cache=True)
        dspy.configure(lm=lm)
        dspy.Predict("text -> ok: bool")(text="connectivity check")
    except Exception as e:
        return jsonify({"error": f"Couldn't reach the model: {e}"}), 400

    with STATE_LOCK:
        STATE["model"] = model
        STATE["configured"] = True
    return jsonify({"ok": True, "model": model})


@app.route("/api/status")
def api_status():
    with STATE_LOCK:
        return jsonify({
            "configured": STATE["configured"],
            "auto_configured": STATE["auto_configured"],
            "model": STATE["model"],
            "has_dataset": STATE["train"] is not None,
            "n_train": len(STATE["train"]) if STATE["train"] is not None else None,
            "n_dev": len(STATE["dev"]) if STATE["dev"] is not None else None,
            "has_optimized": STATE["optimized"] is not None,
        })


@app.route("/api/dataset", methods=["POST"])
def api_dataset():
    body = request.get_json(force=True)
    source = body.get("source")
    module_name = body.get("module", "cot")
    if module_name not in MODULE_LABELS:
        module_name = "cot"
    tr = Trace()

    try:
        if source == "builtin":
            tr.step("reading the built-in dataset from common.dataset()")
            train_ex, dev_ex = common.dataset()
            sig = common.TriageTicket
            categories = sorted({e.category for e in train_ex + dev_ex})
            urgencies = sorted({e.urgency for e in train_ex + dev_ex})
            tr.step(f"got {len(train_ex)} train + {len(dev_ex)} dev dspy.Example objects, already built")
            tr.step("reusing common.TriageTicket - a signature already defined in the codebase, not built on the fly")
        else:
            fmt = body.get("format", "csv")
            raw_text = body.get("raw_text", "")
            instructions = (body.get("instructions") or "").strip() or "Classify the input text."
            tr.step(f"received {len(raw_text)} character(s) of raw {fmt.upper()} text from the browser")
            rows = parse_dataset_text(raw_text, fmt)
            tr.step(f"parsed {len(rows)} row(s)")
            train_rows, dev_rows = split_rows(rows)
            tr.step(f"shuffled (seed=0) and split into {len(train_rows)} train / {len(dev_rows)} dev row(s)")
            categories = sorted({r["category"] for r in rows})
            urgencies = sorted({r["urgency"] for r in rows})
            tr.step(f"scanned every row for distinct labels - found category values: {', '.join(categories)}")
            tr.step(f"found urgency values: {', '.join(urgencies)}")
            sig = build_signature(categories, urgencies, instructions)
            tr.step(
                "called dspy.Signature({...}) to build a fresh typed contract: "
                f"ticket: str (input) -> category: Literal[{len(categories)} values], "
                f"urgency: Literal[{len(urgencies)} values], instructions=\"{instructions}\""
            )
            train_ex = rows_to_examples(train_rows)
            dev_ex = rows_to_examples(dev_rows)
            tr.step(f"wrapped rows as {len(train_ex) + len(dev_ex)} dspy.Example objects, marked 'ticket' as the input field")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Couldn't parse that dataset: {e}"}), 400

    baseline = make_program(sig, module_name)
    tr.step(f"wrapped the signature in {MODULE_LABELS[module_name]}")

    with STATE_LOCK:
        STATE["train"] = train_ex
        STATE["dev"] = dev_ex
        STATE["signature"] = sig
        STATE["module"] = module_name
        STATE["baseline"] = baseline
        STATE["optimized"] = None
        STATE["optimize_job"] = fresh_job()
    tr.step("ready. No calls to the LM have happened yet - step 3 is where the first one happens.")

    return jsonify({
        "n_train": len(train_ex),
        "n_dev": len(dev_ex),
        "categories": categories,
        "urgencies": urgencies,
        "instructions": sig.instructions,
        "module": module_name,
        "sample": [{"ticket": e.ticket, "category": e.category, "urgency": e.urgency} for e in train_ex[:3]],
        "trace": tr.list(),
    })


@app.route("/api/prompt-preview", methods=["POST"])
def api_prompt_preview():
    body = request.get_json(force=True)
    ticket = body.get("ticket", "")
    which = body.get("which", "baseline")
    tr = Trace()

    with STATE_LOCK:
        program = STATE.get("optimized") if which == "optimized" and STATE.get("optimized") else STATE.get("baseline")

    if program is None:
        return jsonify({"error": "Load a dataset first (step 2)."}), 400

    predictor = program.predictors()[0]
    tr.step(
        f"using the '{which}' predictor - {len(predictor.signature.input_fields)} input field(s), "
        f"{len(predictor.signature.output_fields)} output field(s), {len(predictor.demos)} few-shot demo(s)"
    )
    adapter = dspy.ChatAdapter()
    fd = adapter.format_field_description(predictor.signature)
    tr.step("adapter.format_field_description() - listed every field's name, type, and description")
    fs = adapter.format_field_structure(predictor.signature)
    tr.step("adapter.format_field_structure() - built the [[ ## field ## ]] wire-format contract")
    td = adapter.format_task_description(predictor.signature)
    tr.step("adapter.format_task_description() - pulled signature.instructions verbatim, unchanged")
    messages = adapter.format(signature=predictor.signature, demos=predictor.demos, inputs={"ticket": ticket})
    chars = sum(len(m["content"]) for m in messages)
    tr.step(f"adapter.format() assembled all of it into {len(messages)} message(s), {chars} characters - still zero API calls")

    breakdown = {"field_description": fd, "field_structure": fs, "task_description": td}
    return jsonify({
        "messages": messages, "breakdown": breakdown, "n_demos": len(predictor.demos), "trace": tr.list(),
    })


@app.route("/api/classify/stream", methods=["POST"])
def api_classify_stream():
    body = request.get_json(force=True)
    ticket = (body.get("ticket") or "").strip()
    which = body.get("which", "baseline")

    def sse(obj):
        return "data: " + json.dumps(obj) + "\n\n"

    def gen():
        tr = Trace()

        if not ticket:
            yield sse({"type": "error", "msg": "Type something to classify first."})
            return

        with STATE_LOCK:
            configured = STATE["configured"]
            program = STATE.get("optimized") if which == "optimized" and STATE.get("optimized") else STATE.get("baseline")
            model = STATE.get("model")

        if not configured:
            yield sse({"type": "error", "msg": "Add your API key first (step 1)."})
            return
        if program is None:
            yield sse({"type": "error", "msg": "Load a dataset first (step 2)."})
            return

        predictor = program.predictors()[0]
        yield sse({"type": "step", **tr.step(f"selected the '{which}' program - {len(predictor.demos)} few-shot demo(s)")})

        adapter = dspy.ChatAdapter()
        fd = adapter.format_field_description(predictor.signature)
        yield sse({"type": "step", **tr.step("adapter.format_field_description() - listed every field's name, type, and description")})
        fs = adapter.format_field_structure(predictor.signature)
        yield sse({"type": "step", **tr.step("adapter.format_field_structure() - built the [[ ## field ## ]] wire-format contract")})
        td = adapter.format_task_description(predictor.signature)
        yield sse({"type": "step", **tr.step("adapter.format_task_description() - pulled signature.instructions verbatim")})

        messages = adapter.format(signature=predictor.signature, demos=predictor.demos, inputs={"ticket": ticket})
        chars = sum(len(m["content"]) for m in messages)
        yield sse({"type": "step", **tr.step(f"adapter.format() assembled it all into {len(messages)} message(s), {chars} characters")})

        yield sse({
            "type": "prompt",
            "breakdown": {"field_description": fd, "field_structure": fs, "task_description": td},
            "n_demos": len(predictor.demos),
            "n_messages": len(messages),
        })

        yield sse({"type": "waiting", **tr.step(f"sending the request to {model} - waiting for a real response...")})

        try:
            pred = program(ticket=ticket)
        except Exception as e:
            yield sse({"type": "error", "msg": f"The model call failed: {e}"})
            return

        yield sse({"type": "step", **tr.step("response received - parsing it against the [[ ## field ## ]] wire format")})
        yield sse({"type": "step", **tr.step(f"category -> {pred.category!r}   urgency -> {pred.urgency!r}")})

        out = {"type": "done", "category": pred.category, "urgency": pred.urgency}
        if hasattr(pred, "reasoning") and pred.reasoning:
            out["reasoning"] = pred.reasoning
        yield sse(out)

    return Response(gen(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def run_optimize_job(job, train, dev, sig, module_name, model, metric_mode, optimizer_mode):
    logger_dspy = logging.getLogger("dspy")
    handler = JobLogHandler(job)
    old_level = logger_dspy.level
    logger_dspy.addHandler(handler)
    logger_dspy.setLevel(logging.INFO)
    sink = JobStdWriter(job)

    append_log(job, f"Loaded {len(train)} training example(s) and {len(dev)} dev example(s).")
    append_log(job, f"Rubric: {METRIC_LABELS.get(metric_mode, metric_mode)} counts as correct.")
    append_log(job, f"Search budget: {'MIPROv2 (thorough)' if optimizer_mode == 'thorough' else 'BootstrapFewShot (quick)'}.")

    try:
        metric = make_metric(metric_mode)
        baseline = make_program(sig, module_name)
        evaluate = dspy.Evaluate(devset=dev, metric=metric, num_threads=4, display_progress=False)

        with redirect_stdout(sink), redirect_stderr(sink):
            append_log(job, "Evaluating the untouched baseline program on the dev set...")
            before = evaluate(baseline).score
            append_log(job, f"Baseline score: {before:.1f}% - this is the number to beat.")

            if optimizer_mode == "thorough":
                append_log(job, "Starting MIPROv2: an LM will propose candidate instructions, then a Bayesian search (Optuna) tries combinations of (instruction, few-shot set) against the rubric above.")
                optimizer = dspy.MIPROv2(
                    metric=metric,
                    prompt_model=dspy.LM(model, cache=True),
                    auto="light",
                    num_threads=4,
                )
                optimized = optimizer.compile(
                    baseline, trainset=train, valset=dev, requires_permission_to_run=False,
                )
            else:
                append_log(job, "Starting BootstrapFewShot: running the program on training examples and keeping the ones it got fully right as worked examples in the prompt.")
                optimizer = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=4, max_labeled_demos=8)
                optimized = optimizer.compile(baseline, trainset=train)

            append_log(job, "Evaluating the optimized program on the same dev set, same rubric...")
            after = evaluate(optimized).score
            append_log(job, f"Optimized score: {after:.1f}%  (before was {before:.1f}%)")

        predictor = optimized.predictors()[0]
        with STATE_LOCK:
            STATE["optimized"] = optimized
            job["status"] = "done"
            job["before"] = before
            job["after"] = after
            job["instructions"] = predictor.signature.instructions
            job["n_demos"] = len(predictor.demos)
    except Exception as e:
        with STATE_LOCK:
            job["status"] = "error"
            job["error"] = str(e)
    finally:
        logger_dspy.removeHandler(handler)
        logger_dspy.setLevel(old_level)


@app.route("/api/optimize/start", methods=["POST"])
def api_optimize_start():
    body = request.get_json(force=True)
    metric_mode = body.get("metric", "both")
    optimizer_mode = body.get("optimizer", "quick")

    with STATE_LOCK:
        if not STATE["configured"]:
            return jsonify({"error": "Add your API key first (step 1)."}), 400
        if STATE["train"] is None:
            return jsonify({"error": "Load a dataset first (step 2)."}), 400
        if STATE["optimize_job"]["status"] == "running":
            return jsonify({"error": "An optimization is already running."}), 409
        if optimizer_mode == "thorough" and len(STATE["train"]) < 4:
            return jsonify({"error": "Need at least 4 training rows for the thorough search - try Quick instead."}), 400

        job = fresh_job()
        job["status"] = "running"
        STATE["optimize_job"] = job
        train, dev, sig, module_name, model = (
            STATE["train"], STATE["dev"], STATE["signature"], STATE["module"], STATE["model"],
        )

    thread = threading.Thread(
        target=run_optimize_job,
        args=(job, train, dev, sig, module_name, model, metric_mode, optimizer_mode),
        daemon=True,
    )
    thread.start()
    return jsonify({"ok": True})


@app.route("/api/optimize/status")
def api_optimize_status():
    with STATE_LOCK:
        job = STATE["optimize_job"]
        return jsonify({
            "status": job["status"],
            "log": job["log"][-200:],
            "trials": job["trials"],
            "before": job["before"],
            "after": job["after"],
            "instructions": job["instructions"],
            "n_demos": job["n_demos"],
            "error": job["error"],
        })


if __name__ == "__main__":
    # PORT is injected by most hosting platforms (Render, Railway, Fly...).
    # Locally it's unset, so this still defaults to 127.0.0.1:5000 as before.
    port = int(os.environ.get("PORT", 5000))
    host = "0.0.0.0" if "PORT" in os.environ else "127.0.0.1"
    app.run(host=host, port=port, debug=False, threaded=False)

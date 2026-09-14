# Code walkthrough

This document explains every file in this repo, step by step: what it does,
why it's written that way, and which DSPy mechanism it's demonstrating. It's a
companion to [README.md](README.md), which is written for presenting the demo
live; this one is written for reading and understanding the code itself.

Read it top to bottom, or jump to the file you care about — each section is
self-contained.

There is also a browser-based version of these same two ideas —
[app.py](app.py), a small local Flask app with a beginner-friendly guide built
into the page itself, where you can type your own text or load your own
dataset and see real, live results instead of reading code. See
[README_WEBAPP.md](README_WEBAPP.md) for how to run it.

## Contents

1. [common.py](#commonpy) — the shared foundation every script imports
2. [00_check.py](#00_checkpy) — pre-flight check
3. [01_signatures.py](#01_signaturespy) — declaring a task
4. [02_modules.py](#02_modulespy) — swapping execution strategy
5. [03_react_tools.py](#03_react_toolspy) — tool-using agent
6. [04_optimize.py](#04_optimizepy) — the optimizer, before/after
7. [05_prompt_anatomy.py](#05_prompt_anatomypy) — how a signature becomes an LM prompt
8. [06_instruction_proposer.py](#06_instruction_proposerpy) — an LM writing candidate prompts
9. [07_bayesian_search.py](#07_bayesian_searchpy) — the search that picks the winner
10. [How it fits together](#how-it-fits-together) — the mental model

---

## `common.py`

Everything the other scripts share: LM configuration, the task definition,
the dataset, and the scoring functions. No script talks to the OpenAI API or
defines the task independently — they all import from here, which is why each
demo script can stay short enough to read out loud.

### Model configuration

```python
MODEL = os.environ.get("DSPY_DEMO_MODEL", "openai/gpt-5.4-nano")
PROMPT_MODEL = os.environ.get("DSPY_DEMO_PROMPT_MODEL", MODEL)
```

Two model slots, both overridable by environment variable without touching
code:
- `MODEL` — the "task model" that actually classifies tickets.
- `PROMPT_MODEL` — the model an *optimizer* uses to **write** candidate
  instructions (see `06`/`07`). It defaults to the same model as `MODEL`, but
  since it runs far fewer calls, it's cheap to point at something stronger.

```python
def configure_lm(model: str = MODEL, **kwargs) -> dspy.LM:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(...)
    lm = dspy.LM(model, cache=True, **kwargs)
    dspy.configure(lm=lm)
    return lm
```

`dspy.LM(...)` wraps a model behind DSPy's LM interface (it uses `litellm`
under the hood, which is why `"openai/gpt-5.4-nano"` — a provider-prefixed
string — is enough to route the call correctly). `dspy.configure(lm=lm)` sets
this as the **default LM** for every `dspy.Predict` / `dspy.ChainOfThought` /
`dspy.ReAct` call made afterwards, anywhere in the process, unless a call is
wrapped in `dspy.context(lm=...)` to temporarily override it (used in `04` to
run the task model and prompt model side by side).

`cache=True` is what makes the whole demo safe to run live: identical calls
(same messages, same model, same params) are served from an on-disk cache
instead of hitting the API, so rehearsing a script and then re-running it for
the room costs nothing and returns instantly.

### The signature — the task definition

```python
Category = Literal["billing", "bug", "feature_request", "account_access", "shipping"]
Urgency = Literal["low", "medium", "high"]

class TriageTicket(dspy.Signature):
    """Classify an inbound customer support ticket."""
    ticket: str = dspy.InputField(desc="The raw text the customer sent in.")
    category: Category = dspy.OutputField()
    urgency: Urgency = dspy.OutputField()
```

A `dspy.Signature` is a typed contract, not a prompt. It declares:
- one input field (`ticket`, a string, with a description DSPy will surface
  to the LM),
- two output fields, each constrained to a `Literal` type.

The class docstring (`"""Classify an inbound customer support ticket."""`)
becomes `TriageTicket.instructions` — the *only* free-text instruction in the
whole signature. Everything else here is structural: field names, types,
descriptions. `05_prompt_anatomy.py` shows exactly how this gets compiled into
an actual LM prompt.

The `Literal` types matter beyond documentation: DSPy's parser enforces them
when reading the LM's response back. `pred.urgency` is guaranteed to be
exactly `"low"`, `"medium"`, or `"high"` — never `"HIGH!!"` or a stray
sentence — because a value outside the literal fails to parse and raises,
rather than silently passing through.

### The dataset

```python
RAW = [
    ("I was charged twice for my May subscription ...", "billing", "high"),
    ...
]
```

36 hand-written `(ticket_text, category, urgency)` tuples. The urgency labels
follow a rubric that is written down **only in this file's comments**, never
in the signature or in any prompt:

- `high` — can't use the product, money already left the account wrongly, or
  the account is compromised.
- `medium` — degraded but has a workaround, money at risk but not lost,
  delivery past the promised date.
- `low` — feature requests, cosmetic issues, admin questions.

Several examples are deliberate traps where tone and urgency disagree — e.g.
`"ASAP!!! I need someone to call me about adding one more seat"` is labeled
`low`, because it's a feature/admin request no matter how it's punctuated.
This is the whole point of the demo: a zero-shot model tends to key off tone
("URGENT!!!") instead of the substantive rubric, and the gap between the
zero-shot score and the optimized score in `04`/`06`/`07` is a direct,
measurable readout of whether the program learned the *real* rule or the
surface one.

```python
def dataset() -> tuple[list[dspy.Example], list[dspy.Example]]:
    examples = [
        dspy.Example(ticket=t, category=c, urgency=u).with_inputs("ticket")
        for t, c, u in RAW
    ]
    return examples[:20], examples[20:]
```

Wraps each tuple in a `dspy.Example` — DSPy's basic data record — and marks
`ticket` as the input field via `.with_inputs("ticket")` (everything else on
the `Example` — `category`, `urgency` — is implicitly the label DSPy compares
predictions against). Returns a fixed 20-example train / 16-example dev split.

### The metrics

```python
def triage_metric(example, pred, trace=None) -> float:
    category_ok = str(getattr(pred, "category", "")).strip().lower() == example.category
    urgency_ok = str(getattr(pred, "urgency", "")).strip().lower() == example.urgency
    if trace is not None:
        return category_ok and urgency_ok
    return (category_ok + urgency_ok) / 2.0
```

A metric is just a Python function `(example, prediction) -> score`. This one
averages two field-level accuracies into a 0.0–1.0 score. The `trace`
parameter is a DSPy convention: it's non-`None` specifically when the metric
is being called *during bootstrapping* (deciding whether a generated trace is
good enough to keep as a few-shot demo — see `BootstrapFewShot` in `04`), and
the function demands a **perfect** match in that case — a half-right example
makes a bad teaching demo, even if it's a fine partial score for reporting.

`urgency_only` and `category_only` are the same idea, isolated to one field —
used in `04` to report the breakdown, and it's `urgency_only` where DSPy's
optimizer earns its keep, since that's the field with the hidden rubric.

---

## `00_check.py`

Pre-flight check — nothing here is presented, it's run once beforehand.

```python
configure_lm()
train, dev = dataset()
pred = dspy.Predict(TriageTicket)(ticket=train[0].ticket)
```

Confirms three things in one shot: the API key works, the model name is valid
for your account, and — as a side effect of `cache=True` — the very first
real call gets cached, so it's instant during the actual `01_signatures.py`
run. If the model name is wrong, the `except` block reports it clearly and
tells you how to override it (`$env:DSPY_DEMO_MODEL`), instead of letting a
cryptic litellm error surface later mid-demo.

---

## `01_signatures.py`

Demonstrates: **a signature is declared, not written as a prompt.**

```python
quick = dspy.Predict("ticket -> category, urgency")
```

The shorthand string form of a signature — useful for a one-line sketch, but
untyped (both fields come back as plain strings). Shown first for contrast.

```python
triage = dspy.Predict(TriageTicket)
for ex in train[:4]:
    pred = triage(ticket=ex.ticket)
    ...
```

`dspy.Predict(sig)` is the simplest **module**: it takes a signature and,
when called, runs exactly one LM call per invocation, using the default
adapter (`ChatAdapter`) to turn the signature + inputs into messages and parse
the response back into typed fields. Calling `triage(ticket=...)` returns a
`Prediction` object with `.category` and `.urgency` attributes already
validated against the `Literal` types.

```python
dspy.inspect_history(n=1)
```

Prints the exact messages sent to the LM and the raw completion for the last
`n` calls — DSPy's built-in way to see "what actually happened" without
instrumenting anything yourself. `05_prompt_anatomy.py` builds on this by
constructing that same message list directly, without making a call at all.

---

## `02_modules.py`

Demonstrates: **the signature is fixed; only the execution strategy changes.**

```python
strategies = {
    "Predict        (one shot)": dspy.Predict(TriageTicket),
    "ChainOfThought (reason first)": dspy.ChainOfThought(TriageTicket),
}
evaluate = dspy.Evaluate(devset=dev, metric=triage_metric, num_threads=8, ...)
for name, module in strategies.items():
    scores[name] = evaluate(module).score
```

`dspy.ChainOfThought(sig)` wraps the *same* `TriageTicket` signature but
injects an extra `reasoning` output field and instructs the LM to think before
answering — a one-line change from `Predict`, with no change to the task
definition at all. `dspy.Evaluate` runs a module over every example in
`devset`, applies `metric` to each, and returns an aggregate `.score` (plus,
with threading, does it in parallel — `num_threads=8`).

```python
pred = cot(ticket=dev[5].ticket)
print(pred.reasoning)
```

Because `reasoning` is a normal signature output field (not a side channel),
it's just an attribute on the prediction — inspectable like any other field,
which is what makes ChainOfThought's "thinking" auditable rather than a black
box.

The script's closing comment lists other modules that drop into the same
slot (`ReAct`, `ProgramOfThought`, `BestOfN`, `Refine`, `Parallel`) — the
point being that "should this step reason step by step, retry, or sample N
times" is a module choice, orthogonal to the signature, and `dspy.Evaluate`
gives you a number to decide with instead of guessing.

---

## `03_react_tools.py`

Demonstrates: **plain Python functions become an agent's tools.**

```python
def lookup_order(order_id: str) -> dict:
    """Look up an order by its numeric ID. Returns status, promised delivery date, carrier and total."""
    return ORDERS.get(order_id.strip("#"), {"error": f"no order {order_id}"})
```

Three ordinary functions (`lookup_order`, `lookup_customer`, `refund_policy`)
backed by in-memory fake data (`ORDERS`, `CUSTOMERS`). Each function's type
hints and docstring *are* the tool schema an LM needs to decide when and how
to call it — there's no separate JSON schema to hand-maintain and keep in
sync.

```python
class ResolveTicket(dspy.Signature):
    """Investigate a support ticket using the back-office tools and propose a resolution."""
    ticket: str = dspy.InputField()
    resolution: str = dspy.OutputField(desc="What we should do, in two sentences.")
    refund_usd: float = dspy.OutputField(desc="Dollar amount to refund; 0 if none.")

agent = dspy.ReAct(ResolveTicket, tools=[lookup_order, lookup_customer, refund_policy])
```

`dspy.ReAct` is another module wrapped around a signature, same as `Predict`
or `ChainOfThought` — but this one runs a loop: think, pick a tool, observe
its result, repeat, until it's ready to produce the signature's output
fields. You still just call `agent(ticket=TICKET)`.

```python
trajectory = pred.trajectory
for i in range(...):
    print(trajectory.get(f'thought_{i}', ''))
    print(trajectory.get(f'tool_name_{i}', ''), trajectory.get(f'tool_args_{i}', ''))
    print(trajectory.get(f'observation_{i}', ''))
```

`pred.trajectory` is a flat dict with numbered keys (`thought_0`,
`tool_name_0`, `tool_args_0`, `observation_0`, `thought_1`, ...) recording
every step of the loop — structured data you can print, log, or assert
against, not a log file to scrape.

The closing point: `agent` is still just a `dspy.Module`, so it's a valid
target for the same optimizers used in `04`/`07` — the ReAct loop's internal
instructions can be tuned by a metric exactly like `TriageTicket`'s can.

---

## `04_optimize.py`

Demonstrates: **compiling a program against a metric — the payoff.**

```python
USE_BOOTSTRAP = "bootstrap" in sys.argv
baseline = dspy.ChainOfThought(TriageTicket)
before = report("baseline", baseline)
```

Scores the untouched baseline first — three numbers (`overall`, `category`,
`urgency`), each from a separate `dspy.Evaluate` pass with a different
metric — establishing the "before" everything else is measured against.

```python
if USE_BOOTSTRAP:
    optimizer = dspy.BootstrapFewShot(metric=triage_metric, max_bootstrapped_demos=4, max_labeled_demos=8)
    optimized = optimizer.compile(baseline, trainset=train)
```

**`BootstrapFewShot`** — the cheap path (~30s). It runs `baseline` on the
training set, keeps the traces where `triage_metric(..., trace=...)` demanded
a perfect match, and attaches up to 4 of those as few-shot demos on the
predictor. It never touches the instruction text — the docstring stays
exactly as written. All of the "after" improvement, if any, comes purely from
in-context examples.

```python
else:
    optimizer = dspy.MIPROv2(
        metric=triage_metric,
        prompt_model=dspy.LM(PROMPT_MODEL, cache=True),
        auto="light",
        num_threads=8,
    )
    optimized = optimizer.compile(baseline, trainset=train, valset=dev, requires_permission_to_run=False)
```

**`MIPROv2`** — the full path (minutes). This is the default when no argument
is passed. It runs a three-stage pipeline (bootstrap demo candidates → an LM
proposes instruction candidates → Bayesian search picks the best
instruction/demo combination) — covered move-by-move in `06` and `07`. The
returned `optimized` program has a *new* instruction string baked into its
predictor's signature, not just new demos. `requires_permission_to_run=False`
is set purely so the call doesn't pause for a confirmation prompt mid-demo.

```python
predictor = optimized.predictors()[0]
print(predictor.signature.instructions)
print(len(predictor.demos))
optimized.save("optimized_triage.json")
```

`optimized.predictors()` returns the list of predictor objects inside the
compiled module (here, just one — `TriageTicket`'s). Its `.signature`
carries whatever instruction the optimizer settled on, and `.demos` holds the
selected few-shot examples. `.save(...)` serializes both to JSON — this is
"the compiled prompt" as a reviewable, diffable, versionable artifact, not a
string buried in a Python file. It reloads with
`dspy.ChainOfThought(TriageTicket); m.load("optimized_triage.json")`.

---

## `05_prompt_anatomy.py`

Demonstrates: **the function that turns a signature into an LM prompt,
called directly.**

This script makes no attempt to optimize anything — it exists purely to open
up `01`'s `dspy.inspect_history()` output and show *why* it looks the way it
does.

```python
adapter = dspy.ChatAdapter()
messages = adapter.format(signature=TriageTicket, demos=[], inputs={"ticket": train[0].ticket})
```

`dspy.ChatAdapter` is the object `dspy.Predict` uses internally, every time,
to go from `(signature, demos, inputs)` to the actual list of chat messages.
Calling `.format()` yourself, with no LM call involved, exposes exactly what
gets sent — for free, instantly, with no API cost.

```python
adapter.format_field_description(TriageTicket)   # 1
adapter.format_field_structure(TriageTicket)     # 2
adapter.format_task_description(TriageTicket)    # 3
```

The system message `format()` builds is the concatenation of exactly these
three sub-functions:

1. **`format_field_description`** — lists input/output fields and their
   types (`ticket (str)`, `category (Literal[...])`, ...).
2. **`format_field_structure`** — the wire format contract: a template using
   `[[ ## fieldname ## ]]` markers, so the LM's response can be parsed
   unambiguously by looking for those markers, plus a `[[ ## completed ## ]]`
   sentinel marking the end.
3. **`format_task_description`** — literally `signature.instructions`, i.e.
   the docstring. This is the *only* piece of the whole prompt that an
   optimizer ever rewrites.

```python
demos = [{"ticket": ex.ticket, "category": ex.category, "urgency": ex.urgency} for ex in train[:2]]
messages_with_demos = adapter.format(signature=TriageTicket, demos=demos, inputs={...})
```

Shows that demos are a `format()` argument, not a signature change: passing 2
demos turns a 2-message exchange into 6 (system + 2×(user+assistant demo
pairs) + the real user question), with `TriageTicket` itself untouched.

```python
dspy.Predict(TriageTicket)(ticket=train[0].ticket)
dspy.inspect_history(n=1)
```

One live call at the end, to confirm `inspect_history`'s output matches the
hand-built `adapter.format()` output exactly — proving this isn't a
simplification, it's the real mechanism.

---

## `06_instruction_proposer.py`

Demonstrates: **an LM writing candidate instructions, scored by hand — MIPROv2
step 2, in isolation.**

```python
proposer = GroundedProposer(
    prompt_model=dspy.LM(PROMPT_MODEL, cache=True),
    program=baseline,
    trainset=train,
    program_aware=True,
    use_dataset_summary=True,
    use_task_demos=False,
    use_tip=True,
    set_tip_randomly=True,
)
```

`GroundedProposer` is itself a `dspy.Module` — its job is to write a *new*
candidate instruction for a predictor. "Grounded" means it doesn't invent in a
vacuum; it's given real context:
- `use_dataset_summary=True` → it first runs an LM call to summarize the
  training set into a short description.
- `program_aware=True` → it runs LM calls (`DescribeProgram`,
  `DescribeModule`, both internal signatures) to describe what the program
  as a whole appears to be doing, and this predictor's role in it.
- `set_tip_randomly=True` → it picks a random entry from a fixed dict of
  "tips" (`"Be creative"`, `"Include a persona..."`, `"State a high-stakes
  scenario..."`, etc.) to bias the style of what it writes.

All of that context is fed into one more internal signature,
`GenerateSingleModuleInstruction`, whose output field is the actual proposed
instruction text.

```python
proposed = proposer.propose_instructions_for_program(
    trainset=train, program=baseline, demo_candidates=None, trial_logs={}, N=N,
)[0]
```

Asks for `N=4` candidates for predictor 0 (there's only one predictor in
`baseline`). `demo_candidates=None` deliberately isolates instruction
proposal from few-shot demo selection — otherwise this would need to
bootstrap demo sets first, same as `04`/`07` do internally.

```python
for i, instruction in enumerate(candidates):
    trial = baseline.deepcopy()
    trial.predictors()[0].signature = trial.predictors()[0].signature.with_instructions(instruction)
    score = evaluate(trial).score
```

The actual measurement step: deep-copy the baseline program, swap in one
candidate instruction via `signature.with_instructions(...)` (the same method
`MIPROv2` uses internally — see `07`), and score it on the dev set with the
exact same `triage_metric` used everywhere else. Ranking the results by score
is literally "propose, then measure" — the two halves of what MIPROv2 does
automatically, done by hand here so both halves are visible.

Running this script produced a concrete result: the original one-line
docstring scored **68.8%**; the LM's best proposed replacement scored
**84.4%** — the instruction-writing step alone, with no few-shot demos and no
search algorithm, closing most of the gap `04`'s MIPROv2 run closes.

---

## `07_bayesian_search.py`

Demonstrates: **the search algorithm that decides which candidate wins — made
visible, trial by trial.**

This runs the *exact same* `MIPROv2.compile()` call as `04`'s default path,
with `track_stats=True` (the default) so the optimizer's internal
`trial_logs` survive on the returned program — then reads that log back out
instead of only reporting the final before/after numbers.

```python
for trial_num in sorted(optimized.trial_logs):
    log = optimized.trial_logs[trial_num]
    instr_idx = log.get("0_predictor_instruction")
    demo_idx = log.get("0_predictor_demos")
    ...
```

Internally, MIPROv2 treats "which instruction" and "which demo set" for each
predictor as **categorical hyperparameters**, and hands them to
[Optuna](https://optuna.org/), a general-purpose hyperparameter-search
library, via `optuna.samplers.TPESampler` — a Bayesian method (Tree-structured
Parzen Estimator) that picks the next combination to try based on which
regions of the search space have scored well so far, rather than trying every
combination or picking randomly. Every trial:

1. picks an `(instruction_idx, demo_idx)` pair via `trial.suggest_categorical(...)`,
2. plugs them into a fresh deep copy of the program,
3. scores that copy with `dspy.Evaluate(..., metric=triage_metric)`,
4. reports the score back to Optuna, which updates its model of "what tends
   to score well" before choosing the next trial.

On larger datasets, most trials evaluate on a cheap minibatch and only the
current leader gets promoted to a trustworthy full-dataset evaluation
periodically — the exploration/confirmation split visible in
`_perform_full_evaluation` inside `mipro_optimizer_v2.py`. In this repo's
16-example dev set, that minibatching is disabled entirely (`auto="light"`'s
default minibatch size of 35 exceeds the dev set), so every printed trial in
this script's trace is already a full, trustworthy evaluation — simpler to
read, but not exercising the minibatch/full-eval split MIPROv2 uses at scale.

```python
best = optimized.predictors()[0]
print(optimized.score)
print(best.signature.instructions)
print(len(best.demos))
```

The final program is whichever trial scored highest — literally
`max(triage_metric(...) over every combination tried)`. A real run produced:
default program **68.75%** → best combination (one specific proposed
instruction + one specific bootstrapped demo set) **84.38%**, found on trial
4 out of 10 and re-confirmed by two later trials landing on the same
combination independently — a visible sign the search had actually converged
rather than gotten lucky once.

The closing point of the script: swap `triage_metric` for `urgency_only`, or
for any other function of `(example, prediction) -> score`, and this same
search machinery now optimizes for *that* definition of "best" — the rubric
is a parameter you pass in, not something wired into DSPy.

---

## How it fits together

```
common.py
  ├─ TriageTicket (signature)         ─┐
  ├─ dataset()                         ├─ used by every script below
  └─ triage_metric / urgency_only ...  ┘

00_check.py            sanity check before presenting

01_signatures.py        TriageTicket + dspy.Predict
                         → adapter.format() under the hood (see 05)

02_modules.py            TriageTicket + {Predict, ChainOfThought}
                         → same signature, different module, scored by dspy.Evaluate

03_react_tools.py       ResolveTicket + dspy.ReAct + 3 tool functions
                         → same "signature + module" pattern, now with a tool loop

04_optimize.py           ChainOfThought(TriageTicket)
                         → compiled by BootstrapFewShot (demos only)
                            or MIPROv2 (instructions + demos)          ─┐
                                                                        │
05_prompt_anatomy.py    ChatAdapter.format() called directly           │  the same
                         → what "compiling a prompt" operates on       │  machinery,
                                                                        │  opened up
06_instruction_proposer.py  GroundedProposer called directly           │
                         → how MIPROv2 proposes instruction candidates │
                                                                        │
07_bayesian_search.py    MIPROv2.compile(), trial_logs read back       │
                         → how MIPROv2 picks a winner among candidates ┘
```

Everything downstream of `common.py` is the same task (`TriageTicket`) and
the same metric (`triage_metric`) viewed through a different lens: how it's
executed (`02`, `03`), how it's compiled into a better program (`04`), and —
new in this walkthrough — what "compiled" actually means mechanically (`05`,
`06`, `07`).

# DSPy demo — support-ticket triage

A four-step live demo of [DSPy](https://dspy.ai/) that builds one task all the
way from a declared signature to a compiled, measurably-better program.

Built against **DSPy 3.3.1** (current stable; 3.4.0b1 is the beta line).

This README is written for presenting the demo live from the terminal. For a
step-by-step explanation of what every file's code actually does, see
[CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md). For a **click-through browser
demo** — pick a dataset (built-in or your own), type anything and see it
classified in real time, then watch a real optimization run — see
[README_WEBAPP.md](README_WEBAPP.md) and run `python app.py`.

## The idea that makes the demo land

The task is triaging support tickets into a `category` and an `urgency`.

The urgency labels follow a consistent rubric that is **never written in any
prompt** — it exists only in the training labels:

| urgency | rule |
| --- | --- |
| `high` | can't use the product, money already left the account wrongly, or account compromised |
| `medium` | degraded but has a workaround, money at risk but not lost, delivery past the promised date |
| `low` | feature requests, cosmetic issues, admin questions |

Tone carries **no** signal, and the dataset contains deliberate traps:

- `"URGENT!!! ... adding one more seat to our plan"` → **low**
- `"Please consider adding keyboard shortcuts"` from an enterprise account → **low**
- `"The dropdown text is slightly misaligned in Safari"` → **low**

A zero-shot model over-weights the shouting. The optimizer recovers the real
rule from data. That gap is your before/after number.

## Setup

```powershell
cd C:\Users\hp\dspy-demo
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:OPENAI_API_KEY = "sk-..."
.\.venv\Scripts\python.exe 00_check.py
```

The venv already exists; `requirements.txt` pins `dspy` and `optuna` (MIPROv2's
optimizer backend — without it, `04_optimize.py`'s default mode and
`07_bayesian_search.py` raise `ImportError`). `00_check.py` verifies the key,
validates the model name, and warms the cache — **run it before you present.**

### Model

Defaults to `openai/gpt-5.4-nano`. If that isn't available on your account,
`00_check.py` will say so; override it with:

```powershell
$env:DSPY_DEMO_MODEL = "openai/<model-id>"
```

Optionally point the optimizer's prompt-writing model at something stronger —
it runs far fewer calls than the task model, so it's cheap to upgrade:

```powershell
$env:DSPY_DEMO_PROMPT_MODEL = "openai/<bigger-model-id>"
```

## Running it

| Step | Script | Time | The point |
| --- | --- | --- | --- |
| 1 | `01_signatures.py` | seconds | You declare the task, DSPy writes the prompt |
| 2 | `02_modules.py` | ~30s | Swap the strategy without touching the task |
| 3 | `03_react_tools.py` | ~20s | Plain Python functions become an agent's tools |
| 4 | `04_optimize.py` | minutes | Compile against a metric; show the delta |
| 5 | `05_prompt_anatomy.py` | seconds | The adapter function that turns a signature into messages |
| 6 | `06_instruction_proposer.py` | ~20s | An LM proposes candidate instructions; you score them |
| 7 | `07_bayesian_search.py` | minutes | Watch MIPROv2's search trial-by-trial |

```powershell
.\.venv\Scripts\python.exe 01_signatures.py
.\.venv\Scripts\python.exe 02_modules.py
.\.venv\Scripts\python.exe 03_react_tools.py
.\.venv\Scripts\python.exe 04_optimize.py
.\.venv\Scripts\python.exe 05_prompt_anatomy.py
.\.venv\Scripts\python.exe 06_instruction_proposer.py
.\.venv\Scripts\python.exe 07_bayesian_search.py
```

Steps 5-7 are an optional "how it actually works" deep dive: 5 shows the
function that compiles a signature into an LM prompt (no optimizer involved),
6 pulls MIPROv2's instruction-writing step out in isolation, and 7 re-runs the
step-4 optimization but prints the trial-by-trial search trace instead of just
the before/after number.

Every script prints its own talking points at the end, so you can present
straight from the terminal.

## Live-demo safety

- **Caching is on** (`dspy.LM(..., cache=True)`). A rehearsal run makes the real
  run instant and free. Rehearse each script once beforehand and the demo can't
  stall on a slow API.
- **MIPROv2 is the slow step.** If you're short on time or the network is
  flaky, run the 30-second variant instead:
  ```powershell
  .\.venv\Scripts\python.exe 04_optimize.py bootstrap
  ```
  This uses `BootstrapFewShot`, which only selects few-shot demos — it does
  **not** rewrite the instruction, and the script says so rather than
  overclaiming. The headline number still moves.
- `requires_permission_to_run=False` is set on the MIPROv2 compile so it won't
  stop and wait for a keypress mid-demo.
- Step 4 writes `optimized_triage.json`. Reload it with
  `m = dspy.ChainOfThought(TriageTicket); m.load("optimized_triage.json")` —
  useful for showing that the compiled prompt is a versionable artifact.

## Known rough edges

- **The step-4 delta is not guaranteed.** It depends on the model and on
  optimizer sampling. Rehearse it once; with caching, the rehearsed numbers are
  exactly what the room will see. If the delta comes out flat, raise
  `auto="medium"` or widen the dev set.
- **Step 2 may show ChainOfThought losing to Predict.** That is a fine outcome
  to present honestly — it's the argument for measuring instead of assuming, and
  it sets up step 4.
- The dataset is 36 hand-written examples (20 train / 16 dev). Big enough to
  move a number, small enough to stay cheap; not a benchmark.

## Files

| File | Contents |
| --- | --- |
| `common.py` | LM config, the dataset, the `TriageTicket` signature, the metrics |
| `00_check.py` | Pre-flight: key, model, connectivity, cache warm-up |
| `01_signatures.py` | `dspy.Predict`, typed outputs, `inspect_history()` |
| `02_modules.py` | `Predict` vs `ChainOfThought`, scored with `dspy.Evaluate` |
| `03_react_tools.py` | `dspy.ReAct` over three fake back-office functions |
| `04_optimize.py` | `MIPROv2` / `BootstrapFewShot`, before/after, saved artifact |
| `05_prompt_anatomy.py` | Calls `ChatAdapter.format()` directly to show the messages it builds |
| `06_instruction_proposer.py` | `GroundedProposer` in isolation: propose instructions, then score them |
| `07_bayesian_search.py` | Same MIPROv2 compile as 04, with the Optuna trial trace printed out |

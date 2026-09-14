# DSPy Live Lab — a real, click-through demo

This is a small local website (`app.py`) for showing someone — a professor, a
teammate, yourself in six months — how DSPy actually works, live. Every
button makes a real call into the `dspy` library running on your machine.
Nothing is pre-recorded or faked. If your professor wants to try his own data
or his own wording, he can, right there in the browser, and get a real answer
back.

It's a companion to the numbered scripts (`01_signatures.py` through
`07_bayesian_search.py`) and to [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md),
which explain the same ideas as read-through code. This one is for driving
live in front of someone who doesn't want to read Python.

## Running it

```powershell
cd C:\Users\hp\dspy-demo
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Then open **http://127.0.0.1:5000** in a browser. Stop it with `Ctrl+C` in the
terminal when you're done. Nothing is installed system-wide and nothing is
sent anywhere except OpenAI's API — your key is kept only in the running
Python process's memory, never written to a file.

If `OPENAI_API_KEY` is already set as an environment variable before you start
it, the app connects automatically and step 1 disappears entirely — that's
what [DEPLOY.md](DEPLOY.md) walks through, for putting this on a public URL so
someone else can open it without ever seeing a key field. Read that file's
first section before you do, though — running it that way means your key gets
spent by anyone who has the link.

## What your professor can actually do

- **Use your dataset**, the 36 real support tickets in `common.py`, or
  **paste/upload his own** — a CSV with a `ticket,category,urgency` header, or
  the equivalent JSON. His labels don't have to match yours; DSPy builds a
  fresh typed signature from whatever categories actually appear in his data.
- **Type any sentence** and see, in real time, the exact prompt DSPy builds
  for it, and then the model's real answer.
- **Run a real optimization** against his own data and his own definition of
  "correct," and watch the search happen — a live log of DSPy's real output,
  and a chart of real scores per trial, not a canned animation.

## The two ideas this is built to show

### 1. A signature is not a prompt — a function *compiles* it into one

In DSPy you declare a **signature**: named input fields, named output fields,
each with a type, plus one instruction string. That's it — no "You are a
helpful assistant" text anywhere. Step 3 of the page exposes the function that
turns that declaration into an actual prompt, `ChatAdapter.format()`, and
shows you its literal output, split into the three things it's made of:

1. **Field description** — the plain list of input/output fields and their
   types, generated from the signature.
2. **Field structure** — a strict `[[ ## fieldname ## ]]` wire format, so the
   model's reply can be parsed back into real typed fields instead of scraped
   with regex and hope.
3. **Task description** — literally the one instruction string you wrote (or
   left as the default). This is the *only* free-text part of the whole
   prompt, and the only piece an optimizer ever rewrites.

Click **"Preview prompt"** and this happens with zero API calls — it's pure
string formatting, instant and free. Click **"Classify"** and the *same*
exact prompt gets sent for real, and you see the real answer come back.
That's the whole trick: a prompt in DSPy is the deterministic output of a
function of (signature, few-shot examples, your input) — not a string someone
wrote and is now afraid to touch.

### 2. "Best" means highest score on a rubric you chose — and DSPy searches for it

Step 4 lets you pick what "correct" even means (both fields must match? just
one?) — that dropdown *is* the rubric, expressed as a scoring function. DSPy
then tries different candidate prompts against your dev examples using that
scoring function and keeps whichever one scores highest:

- **Quick (`BootstrapFewShot`)** — runs your program on the training
  examples, keeps the ones it got completely right, and attaches a few of
  those as worked examples in the prompt. Doesn't touch your instruction text.
- **Thorough (`MIPROv2`)** — additionally asks a language model to *write*
  several new candidate instructions (grounded in a summary of your data and
  a description of what the program does), then runs a real search — Optuna's
  Bayesian optimizer — over combinations of (instruction, example set),
  scoring each combination with your rubric, until it converges on the best
  one it found.

Change the dropdown from "both fields" to "urgency only" and re-run: the
exact same search machinery now optimizes for a *different* goal, because the
rubric is a parameter you handed it, not something wired into DSPy. That's
the point worth making to a room: the "intelligence" here isn't the LM
guessing well, it's a search process with a measurable objective, same as any
other optimizer — it just happens to be searching over English text instead
of numbers.

## If you get an error live

- **"Add your API key first"** — go back to step 1.
- **"Couldn't reach the model"** — the key or model name is wrong; check the
  key, or try a different model string (e.g. `openai/gpt-4o-mini`).
- **"Need at least 4 rows..."** — a custom dataset needs enough rows to make a
  train/dev split; add more, or use the built-in dataset to keep going.
- Optimization errors show up in the live log and the banner above it, not as
  a silent hang.

## What's deliberately left out

This is a teaching tool for one signature-shaped task (one text field in, two
label fields out), not a general AutoML product — a real generic "define any
task" builder is a bigger, riskier thing to demo live. If your professor
wants to build "something similar" for a different shape of task (more input
fields, a different number of outputs, non-classification outputs), the honest
answer is: that's a new signature in `common.py`-style code, following the
same three ideas above — it's not something this particular page generalizes
to automatically.

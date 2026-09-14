"""4 - Optimizers: compile the program against a metric.

This is the payoff. Nobody edits a prompt. We hand DSPy a metric and a training
set, and it searches for instructions and few-shot demos that score better.

Usage:
    python 04_optimize.py            # MIPROv2, auto="light"  (a few minutes)
    python 04_optimize.py bootstrap  # BootstrapFewShot       (~30 seconds)
"""

import sys

import dspy

from common import (
    PROMPT_MODEL,
    TriageTicket,
    category_only,
    configure_lm,
    dataset,
    triage_metric,
    urgency_only,
)

configure_lm()
train, dev = dataset()

USE_BOOTSTRAP = "bootstrap" in sys.argv


def report(label: str, program) -> dict:
    """Score a program on the dev set, broken out by field."""
    out = {}
    for name, metric in (("overall", triage_metric),
                         ("category", category_only),
                         ("urgency", urgency_only)):
        out[name] = dspy.Evaluate(
            devset=dev, metric=metric, num_threads=8, display_progress=False
        )(program).score
    print(f"{label:<12} overall {out['overall']:5.1f}%   "
          f"category {out['category']:5.1f}%   urgency {out['urgency']:5.1f}%")
    return out


baseline = dspy.ChainOfThought(TriageTicket)

print("=" * 70)
print("BEFORE")
print("=" * 70)
before = report("baseline", baseline)

print()
print("=" * 70)
print(f"COMPILING with {'BootstrapFewShot' if USE_BOOTSTRAP else 'MIPROv2'}")
print("=" * 70)

if USE_BOOTSTRAP:
    # Cheapest useful optimizer: run the program on the trainset, keep the
    # traces where the metric passed, and use those as few-shot demos.
    optimizer = dspy.BootstrapFewShot(
        metric=triage_metric,
        max_bootstrapped_demos=4,
        max_labeled_demos=8,
    )
    optimized = optimizer.compile(baseline, trainset=train)
else:
    # MIPROv2 also *rewrites the instructions*, proposing candidates with a
    # second LM and searching instruction/demo combinations against the metric.
    optimizer = dspy.MIPROv2(
        metric=triage_metric,
        prompt_model=dspy.LM(PROMPT_MODEL, cache=True),
        auto="light",
        num_threads=8,
    )
    optimized = optimizer.compile(
        baseline,
        trainset=train,
        valset=dev,
        requires_permission_to_run=False,  # keeps the live demo from blocking
    )

print()
print("=" * 70)
print("AFTER")
print("=" * 70)
after = report("optimized", optimized)

print()
print("=" * 70)
print("DELTA")
print("=" * 70)
for field in ("overall", "category", "urgency"):
    d = after[field] - before[field]
    print(f"  {field:<10} {before[field]:5.1f}%  ->  {after[field]:5.1f}%   "
          f"({d:+.1f})")

# --- Show what it actually learned -------------------------------------------
predictor = optimized.predictors()[0]

print()
print("=" * 70)
if USE_BOOTSTRAP:
    # Be honest on stage: BootstrapFewShot only selects demos. The instruction
    # below is the untouched docstring from the signature.
    print("Instruction (UNCHANGED - BootstrapFewShot only picks demos)")
else:
    print("The instruction DSPy wrote (nobody typed this)")
print("=" * 70)
print(predictor.signature.instructions)

print()
print("=" * 70)
print(f"Few-shot demos it selected: {len(predictor.demos)}")
print("=" * 70)
for demo in predictor.demos[:2]:
    ticket = getattr(demo, "ticket", "")
    print(f"\n  ticket:   {ticket[:70]}...")
    print(f"  category: {getattr(demo, 'category', '-')}   "
          f"urgency: {getattr(demo, 'urgency', '-')}")

optimized.save("optimized_triage.json")
print("\nSaved to optimized_triage.json")
print("Reload with:  m = dspy.ChainOfThought(TriageTicket); m.load('optimized_triage.json')")

learned = ("selected demos that carry it"
           if USE_BOOTSTRAP else
           "recovered the rule from data and wrote it into the instruction")

print(f"""
The closing line for the room:
  The urgency rubric was never written in any prompt. It only ever existed in
  the labels - and 'URGENT!!!' tickets that are actually low priority were in
  there as traps. The optimizer {learned}.
  That is the pitch: prompts become a compiled artifact you can regenerate when
  the model, the data, or the spec changes.
""")

"""2 - Modules: change the execution strategy without touching the task.

Talking point: the signature from step 1 is unchanged. We only swap the module
wrapped around it, and measure what that buys us.
"""

import dspy

from common import TriageTicket, configure_lm, dataset, triage_metric

configure_lm()
_, dev = dataset()

evaluate = dspy.Evaluate(
    devset=dev,
    metric=triage_metric,
    num_threads=8,
    display_progress=True,
)

strategies = {
    "Predict        (one shot)": dspy.Predict(TriageTicket),
    "ChainOfThought (reason first)": dspy.ChainOfThought(TriageTicket),
}

scores = {}
for name, module in strategies.items():
    print(f"\n--- {name} " + "-" * (50 - len(name)))
    scores[name] = evaluate(module).score

print()
print("=" * 70)
print("Same signature, different module")
print("=" * 70)
for name, score in scores.items():
    print(f"  {name:<32} {score:.1f}%")

# --- ChainOfThought adds a field you can read --------------------------------
cot = dspy.ChainOfThought(TriageTicket)
pred = cot(ticket=dev[5].ticket)

print()
print("=" * 70)
print("ChainOfThought exposes its reasoning as a normal output field")
print("=" * 70)
print(f"ticket:    {dev[5].ticket}")
print(f"reasoning: {pred.reasoning}")
print(f"-> {pred.category} / {pred.urgency}   (gold: {dev[5].category} / {dev[5].urgency})")

print("""
Other modules that drop into the exact same slot:
  dspy.ReAct(sig, tools=[...])    tool-using agent loop   (see 03)
  dspy.ProgramOfThought(sig)      writes and runs Python to get the answer
  dspy.BestOfN(module, N, ...)    sample N, keep the best by a reward fn
  dspy.Refine(module, N, ...)     retry with feedback until a threshold is met
  dspy.Parallel(...)              fan out across examples

The point: 'should this step reason step-by-step?' is a one-line change, and
step 4 shows you can measure whether it was worth it.
""")

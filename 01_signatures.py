"""1 - Signatures: declare the task, not the prompt.

Talking point: nobody in this file writes a prompt. We declare the inputs, the
outputs, and their *types*. DSPy builds the prompt, and the type annotations
become real parsing constraints rather than a polite request.
"""

import dspy

from common import TriageTicket, configure_lm, dataset

configure_lm()
train, _ = dataset()

# --- A signature can also be a one-line string -------------------------------
# Useful for a quick sketch. The class form in common.py is what you ship.
quick = dspy.Predict("ticket -> category, urgency")
print("=" * 70)
print("Inline string signature")
print("=" * 70)
print(quick(ticket=train[0].ticket))

# --- The typed class form ----------------------------------------------------
triage = dspy.Predict(TriageTicket)

print()
print("=" * 70)
print("Typed class signature (Literal-constrained outputs)")
print("=" * 70)
for ex in train[:4]:
    pred = triage(ticket=ex.ticket)
    mark = "OK " if (pred.category, pred.urgency) == (ex.category, ex.urgency) else "MISS"
    print(f"\n[{mark}] {ex.ticket[:64]}...")
    print(f"       predicted: {pred.category:<16} {pred.urgency}")
    print(f"       gold:      {ex.category:<16} {ex.urgency}")

# --- Show the prompt DSPy actually built -------------------------------------
# This is the moment that lands with engineers: the prompt is a build artifact.
print()
print("=" * 70)
print("The prompt DSPy generated for that last call")
print("=" * 70)
dspy.inspect_history(n=1)

print("""
Point to make here:
  - We never wrote 'You are a helpful support agent...'.
  - The Literal types are enforced on the way out, so `pred.urgency` is always
    one of low/medium/high. No defensive string parsing downstream.
  - Swap the model and this all still works; the task definition is portable.
""")

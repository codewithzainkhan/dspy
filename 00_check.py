"""Pre-flight. Run this BEFORE you stand up in front of anyone.

Confirms the key works, the model name is valid, and warms the cache so the
first live call is instant.
"""

import dspy

from common import MODEL, TriageTicket, configure_lm, dataset

print(f"model: {MODEL}")
configure_lm()

train, dev = dataset()
print(f"dataset: {len(train)} train / {len(dev)} dev")

try:
    pred = dspy.Predict(TriageTicket)(ticket=train[0].ticket)
except Exception as e:
    raise SystemExit(
        f"\nThe call failed: {type(e).__name__}: {e}\n\n"
        f"If this is a model-not-found error, '{MODEL}' isn't available on your "
        f"account. Pick one you have and re-run:\n"
        f"  $env:DSPY_DEMO_MODEL = 'openai/<model-id>'\n"
    )

print(f"live call OK -> category={pred.category} urgency={pred.urgency}")
print("\nReady. Run 01_signatures.py next.")

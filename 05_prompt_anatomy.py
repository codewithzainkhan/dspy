"""5 - Prompt anatomy: how a Signature becomes an actual LM prompt.

Talking point: a "prompt" in DSPy is not a string anyone writes. It is the
return value of a function: Adapter.format(signature, demos, inputs) ->
list[messages]. Swap the signature, the demos, or the adapter, and the exact
same function produces a different prompt. This script calls that function
directly and shows its three moving parts.
"""

import dspy

from common import TriageTicket, configure_lm, dataset

configure_lm()
train, _ = dataset()

# dspy.Predict(sig) uses exactly this adapter internally to talk to the LM.
adapter = dspy.ChatAdapter()

print("=" * 70)
print("STEP A - zero demos: the messages Adapter.format() builds for one ticket")
print("=" * 70)
messages = adapter.format(signature=TriageTicket, demos=[], inputs={"ticket": train[0].ticket})
for m in messages:
    print(f"--- {m['role']} " + "-" * 60)
    print(m["content"])
    print()

print("=" * 70)
print("Anatomy of that system message: three functions, concatenated")
print("=" * 70)

print("1) format_field_description() -> the typed I/O contract")
print("-" * 70)
print(adapter.format_field_description(TriageTicket))

print()
print("2) format_field_structure() -> the [[ ## field ## ]] wire format")
print("-" * 70)
print(adapter.format_field_structure(TriageTicket))

print()
print("3) format_task_description() -> literally signature.instructions")
print("-" * 70)
print(adapter.format_task_description(TriageTicket))

print()
print("=" * 70)
print("STEP B - same signature, now with 2 few-shot demos")
print("=" * 70)
demos = [{"ticket": ex.ticket, "category": ex.category, "urgency": ex.urgency} for ex in train[:2]]
messages_with_demos = adapter.format(signature=TriageTicket, demos=demos, inputs={"ticket": train[5].ticket})
print(f"{len(messages)} messages with 0 demos  ->  {len(messages_with_demos)} messages with 2 demos")
print("(each demo becomes one user turn + one assistant turn, spliced in between")
print(" the system message and the real question - the signature itself never changes)")

print()
print("=" * 70)
print("Confirming this really is what the LM sees: one live call + inspect_history")
print("=" * 70)
dspy.Predict(TriageTicket)(ticket=train[0].ticket)
dspy.inspect_history(n=1)

print("""
Point to make here:
  - "Prompt engineering" here is: pick a signature (types + docstring), pick an
    adapter, pick demos. format() is deterministic given those three - there is
    no hidden template string to go find and hand-edit.
  - The optimizer's only lever on the text you'd normally hand-tune is
    signature.with_instructions(...), which only ever changes piece (3) above.
    It never touches the field spec or the wire format. See 06 and 07.
""")

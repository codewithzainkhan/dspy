"""6 - The instruction proposer: DSPy writing its own candidate prompts.

Talking point: this is MIPROv2's step 2, pulled out in isolation. Underneath,
GroundedProposer is itself a dspy.Module - it summarizes the *dataset*,
describes the *program*, picks a random "tip", and asks a prompt model to
write a new instruction for TriageTicket. We generate a few candidates here
and just measure them, so you see the raw proposer output before any search
algorithm (07) gets involved in picking a winner.
"""

import dspy
from dspy.propose import GroundedProposer

from common import PROMPT_MODEL, TriageTicket, configure_lm, dataset, triage_metric

configure_lm()
train, dev = dataset()

baseline = dspy.ChainOfThought(TriageTicket)

proposer = GroundedProposer(
    prompt_model=dspy.LM(PROMPT_MODEL, cache=True),
    program=baseline,
    trainset=train,
    view_data_batch_size=10,
    program_aware=True,     # ask an LM to describe what this program does
    use_dataset_summary=True,  # ...grounded in a summary of the training data
    use_task_demos=False,      # isolate instruction proposal from demo selection
    use_tip=True,
    set_tip_randomly=True,
)

N = 4
print("=" * 70)
print(f"Asking the prompt model ({PROMPT_MODEL}) to propose {N} candidate instructions")
print("=" * 70)
proposed = proposer.propose_instructions_for_program(
    trainset=train,
    program=baseline,
    demo_candidates=None,
    trial_logs={},
    N=N,
)[0]

original = baseline.predictors()[0].signature.instructions
candidates = [original, *proposed[1:]]  # slot 0: the untouched docstring, for comparison

evaluate = dspy.Evaluate(devset=dev, metric=triage_metric, num_threads=8, display_progress=False)

print()
print("=" * 70)
print("Scoring each candidate instruction on the dev set (same metric as 04)")
print("=" * 70)
results = []
for i, instruction in enumerate(candidates):
    trial = baseline.deepcopy()
    predictor = trial.predictors()[0]
    predictor.signature = predictor.signature.with_instructions(instruction)
    score = evaluate(trial).score
    tag = "ORIGINAL docstring" if i == 0 else f"proposed candidate {i}"
    results.append((score, tag, instruction))
    print(f"\n[{tag}] score={score:.1f}%")
    print(f"  {instruction}")

print()
print("=" * 70)
print("Ranked by score")
print("=" * 70)
for score, tag, _ in sorted(results, reverse=True):
    print(f"  {score:5.1f}%  {tag}")

print("""
Point to make here:
  - Nobody wrote these instructions by hand. An LM wrote them, grounded in a
    generated summary of the *data* and a generated description of the
    *program*, plus a randomly chosen "tip" (be creative / add a persona /
    state a high-stakes scenario / ...).
  - Proposing candidates (this script) and searching for the best one under a
    metric (07) are two separate, composable jobs - MIPROv2 just chains them.
""")

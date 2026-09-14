"""7 - The Bayesian search, made visible.

Talking point: 04_optimize.py shows the headline before/after number. This
script runs the *same* MIPROv2 compile but prints the search trace: which
(instruction, demo-set) combination each trial tried, what score it got
against triage_metric, and how the winner was confirmed. "Best" here has one
precise meaning - the combination that scored highest on triage_metric - and
you can watch Optuna's TPE sampler hunt for it trial by trial.
"""

import dspy

from common import PROMPT_MODEL, TriageTicket, configure_lm, dataset, triage_metric

configure_lm()
train, dev = dataset()

baseline = dspy.ChainOfThought(TriageTicket)

optimizer = dspy.MIPROv2(
    metric=triage_metric,
    prompt_model=dspy.LM(PROMPT_MODEL, cache=True),
    auto="light",
    num_threads=8,
    track_stats=True,  # <- this is what lets us read back the search trace
)

print("=" * 70)
print("Compiling (watch the trial-by-trial log above for the raw Optuna output)")
print("=" * 70)
optimized = optimizer.compile(
    baseline,
    trainset=train,
    valset=dev,
    requires_permission_to_run=False,
)

print()
print("=" * 70)
print("The search trace, reconstructed from optimized.trial_logs")
print("=" * 70)
print("Each row is one point Optuna's TPE sampler chose to evaluate in the")
print("space {candidate instruction} x {candidate demo set}:\n")

for trial_num in sorted(optimized.trial_logs):
    log = optimized.trial_logs[trial_num]
    instr_idx = log.get("0_predictor_instruction")
    demo_idx = log.get("0_predictor_demos")
    if "mb_score" in log:
        kind, score = "minibatch", log["mb_score"]
    elif "full_eval_score" in log:
        kind, score = "FULL EVAL", log["full_eval_score"]
    else:
        continue
    params = f"instruction={instr_idx}, demos={demo_idx}" if instr_idx is not None else "(default program)"
    print(f"  trial {trial_num:<3} [{kind:<9}] score={score:5.1f}%   {params}")

print()
print("=" * 70)
print("What actually won")
print("=" * 70)
best = optimized.predictors()[0]
print(f"score: {optimized.score:.1f}%")
print(f"instruction: {best.signature.instructions}")
print(f"few-shot demos: {len(best.demos)}")

print("""
Point to make here:
  - "Best" is not a vibe - it is the literal argmax of triage_metric over every
    (instruction, demo-set) combination Optuna tried.
  - Minibatch scores are noisy and cheap; full evals are expensive and trusted.
    The optimizer spends most of its budget on cheap minibatch probes and only
    promotes a candidate to a full eval when it looks like the current best -
    that is the exploration/confirmation split you can see in the trace above.
  - Swap triage_metric for urgency_only, or for a completely different rubric,
    and this exact same search machinery now optimizes for THAT definition of
    "best" instead. The rubric is a parameter, not something baked into DSPy.
""")

"""Shared pieces for the DSPy demo: LM config, the dataset, the signature, the metric.

Everything the four demo scripts have in common lives here so each script stays
short enough to read out loud.
"""

import os
from typing import Literal

import dspy

# --- Model configuration -----------------------------------------------------
# Override without editing code:  set DSPY_DEMO_MODEL=openai/<your-model>
MODEL = os.environ.get("DSPY_DEMO_MODEL", "openai/gpt-5.4-nano")

# Optimizers use a second LM to *write* prompts. A stronger model here pays off,
# since it runs far fewer calls than the task model.
PROMPT_MODEL = os.environ.get("DSPY_DEMO_PROMPT_MODEL", MODEL)


def configure_lm(model: str = MODEL, **kwargs) -> dspy.LM:
    """Point DSPy at an LM and return it.

    cache=True is what makes this demo safe to run live: a repeated call is
    served from disk, so re-running a script costs nothing and returns instantly.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set.\n"
            "  PowerShell:  $env:OPENAI_API_KEY = 'sk-...'\n"
            "  bash:        export OPENAI_API_KEY=sk-..."
        )
    lm = dspy.LM(model, cache=True, **kwargs)
    dspy.configure(lm=lm)
    return lm


# --- The signature -----------------------------------------------------------
Category = Literal["billing", "bug", "feature_request", "account_access", "shipping"]
Urgency = Literal["low", "medium", "high"]


class TriageTicket(dspy.Signature):
    """Classify an inbound customer support ticket."""

    ticket: str = dspy.InputField(desc="The raw text the customer sent in.")
    category: Category = dspy.OutputField()
    urgency: Urgency = dspy.OutputField()


# --- The data ----------------------------------------------------------------
# The urgency labels follow a consistent internal rubric that is deliberately
# NOT written down in the signature:
#
#   high   - customer cannot use the product for its purpose, money has already
#            left their account incorrectly, or the account is compromised
#   medium - degraded with a workaround, money at risk but not yet lost, or
#            delivery past the promised date
#   low    - feature requests, cosmetic issues, admin questions, no impact yet
#
# Tone carries no signal. "URGENT!!!" on a seat-count question is still low.
# That is the whole point of the demo: the rubric lives in the data, and the
# optimizer's job is to recover it.

RAW = [
    # (ticket text, category, urgency)
    ("I was charged twice for my May subscription - $49 went out on the 3rd and "
     "again on the 4th. Please refund one of them.", "billing", "high"),
    ("The export button does nothing on the reports page. I can hit the API "
     "instead but it's a lot slower.", "bug", "medium"),
    ("Would love a dark mode for the dashboard. Not urgent, just nice to have.",
     "feature_request", "low"),
    ("I can't log in at all. Password reset emails never arrive. I've been "
     "locked out for two days now.", "account_access", "high"),
    ("URGENT!!! My order #4471 was supposed to arrive Tuesday and it is now "
     "Friday. This is completely unacceptable.", "shipping", "medium"),
    ("Someone else logged into my account from Brazil. I don't know who. I've "
     "already changed my password.", "account_access", "high"),
    ("The invoice PDF still shows my old company name. The billing amount is "
     "correct.", "billing", "low"),
    ("App crashes immediately on launch after the 3.2 update. Completely "
     "unusable on my phone.", "bug", "high"),
    ("Can you add CSV export alongside the existing JSON? We're on the "
     "enterprise plan and this would help a lot.", "feature_request", "low"),
    ("My card was declined and the plan shows past due, but my bank says the "
     "payment cleared. Nothing has actually been deducted.", "billing", "medium"),
    ("Tracking says delivered but nothing arrived. Order #8823.",
     "shipping", "medium"),
    ("The dropdown menu text is slightly misaligned in Safari.", "bug", "low"),
    ("We need SSO via Okta before we can roll this out to the whole team.",
     "feature_request", "low"),
    ("Two-factor codes are being rejected even though my clock is synced. I "
     "cannot get in.", "account_access", "high"),
    ("You charged me $490 instead of $49. The money is gone from my account.",
     "billing", "high"),
    ("Reports are loading slowly, around 20 seconds each. They do still load.",
     "bug", "medium"),
    ("Shipment hasn't moved from 'label created' in 9 days. Order #9001.",
     "shipping", "medium"),
    ("Please consider adding keyboard shortcuts for the common actions.",
     "feature_request", "low"),
    ("I'd like to cancel my subscription and get a prorated refund. No rush.",
     "billing", "low"),
    ("Our entire team of 40 cannot access the workspace - it says "
     "'organization suspended'.", "account_access", "high"),
    ("The webhook fires twice for every event. We dedupe on our side but it's "
     "noisy.", "bug", "medium"),
    ("Charged for a plan I cancelled back in March. $29 came out yesterday.",
     "billing", "high"),
    ("Can we get a Slack integration at some point?", "feature_request", "low"),
    ("Package arrived crushed and the item inside is broken. Order #5512.",
     "shipping", "medium"),
    ("I forgot which email address I signed up with and now I can't get in.",
     "account_access", "high"),
    ("The dashboard is showing last week's numbers instead of today's. We make "
     "decisions off this.", "bug", "high"),
    ("Would be great if the mobile app had an offline mode.",
     "feature_request", "low"),
    ("My annual invoice lists the wrong VAT number. The amount is right, I just "
     "need a corrected PDF for accounting.", "billing", "low"),
    ("The login page returns a 500 for everyone in our org. Nobody can work.",
     "account_access", "high"),
    ("Order #3310 shipped to my old address even though I updated it in "
     "settings before ordering.", "shipping", "medium"),
    ("Search returns no results for terms that definitely exist. Filtering by "
     "tag still works fine.", "bug", "medium"),
    ("ASAP!!! I need someone to call me about adding one more seat to our plan.",
     "billing", "low"),
    ("Pretty disappointed that the color picker only has 8 presets.",
     "feature_request", "low"),
    ("The refund I was promised three weeks ago still hasn't appeared. $220.",
     "billing", "medium"),
    ("I cannot add a payment method - the form rejects every valid card, and my "
     "account lapses tomorrow.", "billing", "high"),
    ("Where's my order? #7742, placed 3 days ago, still no tracking number.",
     "shipping", "low"),
]


def dataset() -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Return (trainset, devset) as DSPy Examples.

    .with_inputs("ticket") is the one piece of bookkeeping DSPy needs: it marks
    which fields are inputs, so everything else is treated as the label.
    """
    examples = [
        dspy.Example(ticket=t, category=c, urgency=u).with_inputs("ticket")
        for t, c, u in RAW
    ]
    return examples[:20], examples[20:]


# --- The metric --------------------------------------------------------------
def triage_metric(example, pred, trace=None) -> float:
    """Average of the two field accuracies.

    The `trace` argument is DSPy's convention for 'am I being used to pick
    few-shot demos right now?'. When it is not None we demand a perfect example,
    because a half-right demo is a bad teaching example.
    """
    category_ok = str(getattr(pred, "category", "")).strip().lower() == example.category
    urgency_ok = str(getattr(pred, "urgency", "")).strip().lower() == example.urgency

    if trace is not None:
        return category_ok and urgency_ok

    return (category_ok + urgency_ok) / 2.0


def urgency_only(example, pred, trace=None) -> float:
    """Urgency accuracy alone - the number that actually moves during the demo."""
    return float(str(getattr(pred, "urgency", "")).strip().lower() == example.urgency)


def category_only(example, pred, trace=None) -> float:
    return float(str(getattr(pred, "category", "")).strip().lower() == example.category)

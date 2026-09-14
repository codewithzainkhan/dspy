"""3 - ReAct: the same declarative style, now with tools.

Talking point: the tools are plain Python functions. Their signatures and
docstrings become the tool schema - there is no separate JSON spec to maintain
and no hand-written agent loop.
"""

import dspy

from common import configure_lm

configure_lm()

# --- A fake back office ------------------------------------------------------
ORDERS = {
    "4471": {"status": "in_transit", "promised": "2026-09-08", "carrier": "DHL",
             "customer": "rita@example.com", "total_usd": 129.00},
    "8823": {"status": "delivered", "promised": "2026-09-01", "carrier": "UPS",
             "customer": "sam@example.com", "total_usd": 64.50},
    "9001": {"status": "label_created", "promised": "2026-09-05", "carrier": "USPS",
             "customer": "lee@example.com", "total_usd": 212.00},
}

CUSTOMERS = {
    "rita@example.com": {"plan": "pro", "since": "2024-02-11", "lifetime_usd": 1840},
    "sam@example.com": {"plan": "free", "since": "2026-07-30", "lifetime_usd": 64},
    "lee@example.com": {"plan": "enterprise", "since": "2023-05-02", "lifetime_usd": 21400},
}


def lookup_order(order_id: str) -> dict:
    """Look up an order by its numeric ID. Returns status, promised delivery date, carrier and total."""
    return ORDERS.get(order_id.strip("#"), {"error": f"no order {order_id}"})


def lookup_customer(email: str) -> dict:
    """Look up a customer by email address. Returns their plan tier, signup date and lifetime spend."""
    return CUSTOMERS.get(email.lower(), {"error": f"no customer {email}"})


def refund_policy(plan: str) -> str:
    """Return the refund policy text that applies to a given plan tier."""
    if plan == "enterprise":
        return "Enterprise: full refund at any time, no approval needed."
    if plan == "pro":
        return "Pro: full refund within 30 days; 50% credit after that."
    return "Free: store credit only, capped at $50."


# --- The agent ---------------------------------------------------------------
class ResolveTicket(dspy.Signature):
    """Investigate a support ticket using the back-office tools and propose a resolution."""

    ticket: str = dspy.InputField()
    resolution: str = dspy.OutputField(desc="What we should do, in two sentences.")
    refund_usd: float = dspy.OutputField(desc="Dollar amount to refund; 0 if none.")


agent = dspy.ReAct(ResolveTicket, tools=[lookup_order, lookup_customer, refund_policy])

TICKET = (
    "URGENT!!! My order #4471 was supposed to arrive Tuesday and it is now "
    "Friday. This is completely unacceptable. I want my money back."
)

print("ticket:", TICKET, "\n")
pred = agent(ticket=TICKET)

print("=" * 70)
print("Trajectory")
print("=" * 70)
trajectory = pred.trajectory
for i in range(len({k for k in trajectory if k.startswith("thought_")})):
    print(f"\nstep {i}")
    print(f"  thought:     {trajectory.get(f'thought_{i}', '')}")
    print(f"  tool:        {trajectory.get(f'tool_name_{i}', '')}"
          f"({trajectory.get(f'tool_args_{i}', '')})")
    print(f"  observation: {trajectory.get(f'observation_{i}', '')}")

print()
print("=" * 70)
print("Result")
print("=" * 70)
print(f"resolution: {pred.resolution}")
print(f"refund_usd: {pred.refund_usd}")

print("""
Point to make here:
  - lookup_order / lookup_customer / refund_policy are ordinary functions. The
    type hints and docstrings ARE the tool schema.
  - The trajectory is inspectable data, not log scrape.
  - And because it is still just a dspy.Module, step 4's optimizer can tune this
    agent's instructions too.
""")

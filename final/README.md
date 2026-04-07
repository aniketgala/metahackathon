---
title: Support Ticket Resolution Environment
emoji: 🎫
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Customer Support Ticket Resolution Environment

A real-world simulation of a customer support agent's workflow. The agent must resolve support tickets by searching a knowledge base, retrieving customer details, and communicating with the customer.

## Tasks

The environment provides 3 tasks of increasing difficulty:

1.  **Easy: Password Reset** - A simple request where the instructions are clearly in the KB.
2.  **Medium: Order Status** - Requires retrieving customer details to find the status of a specific order.
3.  **Hard: Refund Eligibility** - Requires checking both the KB for refund policies and customer details for purchase date to determine eligibility.

## Action Space

- `search_kb(query)`: Search internal documentation.
- `get_customer_details(customer_id)`: Retrieve customer profile, orders, and subscriptions.
- `send_message(message)`: Communicate with the customer.
- `resolve_ticket(resolution)`: Final action to close the ticket with a summary.

## Observation Space

- `ticket_description`: The initial customer issue.
- `customer_id`: The ID of the customer who filed the ticket.
- `search_results`: Articles found in the KB.
- `customer_details`: Data retrieved from the customer database.
- `last_response`: Feedback from the last action taken.
- `is_closed`: Status of the ticket.

## Reward Function

The environment provides dense rewards to guide the agent:
- `+0.1`: First time retrieving useful KB search or customer details.
- `+0.02`: Interaction reward for sending messages.
- `+0.5` to `+0.7`: Successfully resolving the ticket (varies by difficulty).
- `-0.01`: Step penalty for inefficiency.
- `-0.05`: Irrelevant tool calls.
- `-0.4`: Closing the ticket with incorrect resolution.

## Quick Start

```python
from final import FinalAction, FinalEnv

# Use .sync() for a synchronous context manager
with FinalEnv(base_url="http://localhost:8000").sync() as env:
    # Start with easy task
    obs = env.reset().observation
    print(f"Ticket: {obs.ticket_description}")

    # Search KB
    res = env.step(FinalAction(action_type="search_kb", query="password reset"))
    print(f"KB Results: {res.observation.search_results}")

    # Resolve
    res = env.step(FinalAction(action_type="resolve_ticket", resolution="Go to settings > security."))
    print(f"Success: {res.reward > 0.5}")
```

## Setup & Baseline

1. Build the image: `docker build -t support-env .`
2. Run the environment: `docker run -p 8000:8000 support-env`
3. Run baseline: `OPENAI_API_KEY=sk-... python inference.py`

## Baseline Scores

- **Easy**: ~0.93
- **Medium**: ~0.69
- **Hard**: ~0.48

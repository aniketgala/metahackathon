# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Customer Support Ticket Resolution Environment.

This environment simulates a support agent handling tickets using internal tools.
"""

import datetime
from typing import Dict, Any, List
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from models import FinalAction, FinalObservation
except (ImportError, ModuleNotFoundError):
    try:
        from ..models import FinalAction, FinalObservation
    except (ImportError, ValueError):
        from final.models import FinalAction, FinalObservation


class FinalEnvironment(Environment):
    """
    Environment for resolving customer support tickets.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    
    # Class-level state to cycle tasks across connections if needed
    _GLOBAL_TASK_INDEX = -1

    # Internal Knowledge Base
    KNOWLEDGE_BASE = {
        "password_reset": "To reset your password, go to settings > security > change password. If you can't log in, use the 'Forgot Password' link on the login page.",
        "order_tracking": "Orders can be tracked in the 'My Orders' section of your account. Standard shipping takes 3-5 business days.",
        "refund_policy": "Refunds are allowed within 30 days of purchase if the item is in original condition or if it's a subscription with no usage.",
        "subscription_cancel": "Subscriptions can be cancelled anytime from the billing dashboard. Refunds for partial months are not provided.",
    }

    # Internal Customer Database
    CUSTOMER_DB = {
        "CUST123": {
            "name": "John Doe",
            "email": "john@example.com",
            "orders": [
                {"id": "ORD987", "status": "Shipped", "date": "2026-04-01"},
                {"id": "ORD654", "status": "Processing", "date": "2026-04-05"},
            ],
            "subscription": {"active": True, "start_date": "2026-03-15", "type": "Premium"},
        },
        "CUST456": {
            "name": "Jane Smith",
            "email": "jane@example.com",
            "orders": [],
            "subscription": {"active": True, "start_date": "2025-12-01", "type": "Basic"},
        },
    }

    # Task definitions
    TASKS = {
        "easy": {
            "id": "task_easy",
            "description": "User wants to reset their password.",
            "customer_id": "CUST123",
            "expected_info": "password_reset",
        },
        "medium": {
            "id": "task_medium",
            "description": "User wants to know the status of their latest order.",
            "customer_id": "CUST123",
            "expected_info": "ORD654",
        },
        "hard": {
            "id": "task_hard",
            "description": "User wants a refund for their Premium subscription. They started it on 2026-03-15. Check if they are eligible (policy: within 30 days).",
            "customer_id": "CUST123",
            "expected_info": "eligible",
        },
    }

    def __init__(self):
        """Initialize the support ticket environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.task_order = ["easy", "medium", "hard"]
        self.current_task_id = "easy"
        # Don't call reset here to avoid double increment on first reset
        self.is_closed = True # Mark as closed initially

    def reset(self, task_id: str = None) -> FinalObservation:
        """
        Reset the environment for a specific task or cycle to the next one.
        """
        if task_id and task_id in self.TASKS:
            self.current_task_id = task_id
        else:
            FinalEnvironment._GLOBAL_TASK_INDEX = (FinalEnvironment._GLOBAL_TASK_INDEX + 1) % len(self.task_order)
            self.current_task_id = self.task_order[FinalEnvironment._GLOBAL_TASK_INDEX]
        
        task = self.TASKS[self.current_task_id]
        
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.is_closed = False
        self.search_results: List[str] = []
        self.customer_details: Dict[str, Any] = {}
        self.last_response = f"New ticket assigned: {task['description']}"
        self.accumulated_reward = 0.0
        self.history: List[Dict[str, Any]] = []
        
        # Track if they've already used the correct tools to avoid reward spamming
        self.found_kb = False
        self.found_customer = False

        return self._get_observation()

    def step(self, action: FinalAction) -> FinalObservation:  # type: ignore[override]
        """
        Execute an action in the support environment.
        """
        if self.is_closed:
            obs = self._get_observation()
            obs.done = True
            return obs

        self._state.step_count += 1
        reward = -0.01  # Small step penalty for efficiency
        response = ""
        
        # Action logging
        action_data = action.model_dump()
        self.history.append({"step": self._state.step_count, "action": action_data})

        if action.action_type == "search_kb":
            query = action.query.lower() if action.query else ""
            found = []
            for k, v in self.KNOWLEDGE_BASE.items():
                if k in query or any(word in query for word in k.split("_")):
                    found.append(f"{k}: {v}")
            
            if found:
                self.search_results = found
                response = f"Found {len(found)} articles in Knowledge Base."
                if not self.found_kb:
                    reward += 0.1  # Reward for first successful search
                    self.found_kb = True
            else:
                response = "No matching articles found in Knowledge Base."
                reward -= 0.05  # Penalty for bad query

        elif action.action_type == "get_customer_details":
            cid = action.customer_id
            if cid in self.CUSTOMER_DB:
                self.customer_details = self.CUSTOMER_DB[cid]
                response = f"Retrieved details for customer {cid}."
                if not self.found_customer:
                    reward += 0.1  # Reward for first successful lookup
                    self.found_customer = True
            else:
                response = f"Customer ID {cid} not found."
                reward -= 0.05

        elif action.action_type == "send_message":
            msg = action.message
            response = f"Message sent to customer: '{msg}'"
            # Interaction reward
            reward += 0.02

        elif action.action_type == "resolve_ticket":
            self.is_closed = True
            res = action.resolution or ""
            
            # Grader logic inside step for reward shaping
            task = self.TASKS[self.current_task_id]
            success = False
            
            if self.current_task_id == "easy":
                if "password" in res.lower() and "settings" in res.lower():
                    success = True
            elif self.current_task_id == "medium":
                if "processing" in res.lower() or "ORD654" in res.upper():
                    success = True
            elif self.current_task_id == "hard":
                # Eligibility: March 15 to April 7 is 23 days < 30 days.
                if ("eligible" in res.lower() or "approve" in res.lower()) and "30" in res:
                    success = True

            if success:
                success_rewards = {"easy": 0.85, "medium": 0.6, "hard": 0.3}
                reward += success_rewards.get(self.current_task_id, 0.3)
                response = "Ticket resolved successfully."
            else:
                reward -= 0.4  # Penalty for incorrect resolution
                response = "Ticket closed but resolution was incorrect or incomplete."

        self.last_response = response
        
        # Accumulate reward and ensure the total is strictly within (0, 1) if episode is ending
        previous_accumulated = self.accumulated_reward
        self.accumulated_reward += reward
        
        done = self.is_closed or self._state.step_count >= 10
        if done:
            # Clamp the final total reward to [0.01, 0.99] to satisfy "strictly between 0 and 1"
            target_total = max(0.01, min(0.99, self.accumulated_reward))
            # Adjust the current step's reward so that (previous_accumulated + adjusted_reward) == target_total
            reward = target_total - previous_accumulated
            self.accumulated_reward = target_total

        obs = self._get_observation()
        obs.reward = reward
        obs.done = done
        return obs

    def _get_observation(self) -> FinalObservation:
        task = self.TASKS[self.current_task_id]
        return FinalObservation(
            ticket_id=task["id"],
            ticket_description=task["description"],
            customer_id=task["customer_id"],
            search_results=self.search_results,
            customer_details=self.customer_details,
            last_response=self.last_response,
            is_closed=self.is_closed,
            done=self.is_closed,
            reward=0.0, # This is the reward for the step, set in step()
        )

    @property
    def state(self) -> State:
        return self._state

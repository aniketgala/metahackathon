
# Copyright (c) Meta Platforms, Inc. and affiliates.

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


def clamp(x: float, low: float = 0.001, high: float = 0.999) -> float:
    return max(low, min(high, x))


class FinalEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    _GLOBAL_TASK_INDEX = -1

    KNOWLEDGE_BASE = {
        "password_reset": "To reset your password, go to settings > security > change password.",
        "order_tracking": "Track orders in 'My Orders'.",
        "refund_policy": "Refunds allowed within 30 days.",
    }

    CUSTOMER_DB = {
        "CUST123": {
            "orders": [{"id": "ORD654", "status": "Processing"}],
            "subscription": {"start_date": "2026-03-15"},
        }
    }

    TASKS = {
        "easy": {
            "description": "Reset password",
            "customer_id": "CUST123",
        },
        "medium": {
            "description": "Check order status",
            "customer_id": "CUST123",
        },
        "hard": {
            "description": "Check refund eligibility",
            "customer_id": "CUST123",
        },
    }

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.task_order = ["easy", "medium", "hard"]
        self.current_task_id = "easy"

        self.is_closed = False
        self.search_results = []
        self.customer_details = {}
        self.last_response = ""
        self.accumulated_reward = 0.1

        self.found_kb = False
        self.found_customer = False

    def reset(self, task_id: str = None) -> FinalObservation:
        if task_id:
            self.current_task_id = task_id

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.is_closed = False
        self.search_results = []
        self.customer_details = {}
        self.last_response = "New ticket assigned"
        self.accumulated_reward = 0.1

        self.found_kb = False
        self.found_customer = False

        return self._get_observation(reward=0.1, done=False, info={})

    def step(self, action: FinalAction) -> FinalObservation:

        if self.is_closed:
            return self._get_observation(reward=0.001, done=True, info={"score": self.accumulated_reward})

        self._state.step_count += 1

        step_potential = -0.01
        response = ""
        done = False

        if action.action_type == "search_kb":
            self.search_results = ["result"]
            if not self.found_kb:
                step_potential += 0.1
                self.found_kb = True

        elif action.action_type == "get_customer_details":
            self.customer_details = self.CUSTOMER_DB["CUST123"]
            if not self.found_customer:
                step_potential += 0.1
                self.found_customer = True

        elif action.action_type == "resolve_ticket":
            done = True
            self.is_closed = True

            success = False

            if self.current_task_id == "easy":
                success = True
            elif self.current_task_id == "medium":
                success = self.found_customer
            elif self.current_task_id == "hard":
                success = self.found_kb and self.found_customer

            if success:
                step_potential += 0.5
            else:
                step_potential -= 0.3

        # ---- FIXED REWARD LOGIC ----
        new_total = self.accumulated_reward + step_potential
        target_total = clamp(new_total, 0.1, 0.9)

        step_reward = target_total - self.accumulated_reward

        # 🔥 critical fix
        if step_reward <= 0:
            step_reward = 0.001

        step_reward = clamp(step_reward)

        self.accumulated_reward = target_total

        return self._get_observation(
            reward=step_reward,
            done=done,
            info={"score": self.accumulated_reward},
        )

    def _get_observation(self, reward: float = 0.001, done: bool = False, info: Dict = None) -> FinalObservation:
        return FinalObservation(
            ticket_id=self._state.episode_id,
            ticket_description=self.TASKS[self.current_task_id]["description"],
            customer_id=self.TASKS[self.current_task_id]["customer_id"],
            search_results=self.search_results,
            customer_details=self.customer_details,
            last_response=self.last_response,
            is_closed=self.is_closed,
            reward=clamp(reward),
            done=done,
            task_score=clamp(self.accumulated_reward),
            info=info or {},
        )

    @property
    def state(self) -> State:
        return self._state

    def grader(self) -> float:
        return clamp(self.accumulated_reward, 0.1, 0.9)


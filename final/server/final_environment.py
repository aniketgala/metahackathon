
import datetime
from typing import Dict
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from models import FinalAction, FinalObservation
except:
    from final.models import FinalAction, FinalObservation


def clamp(x: float, low: float = 0.001, high: float = 0.999) -> float:
    return max(low, min(high, float(x)))


class FinalEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS = True

    KNOWLEDGE_BASE = {
        "password_reset": "Go to settings > security > change password.",
        "refund_policy": "Refund allowed within 30 days.",
    }

    CUSTOMER_DB = {
        "CUST123": {
            "orders": [{"id": "ORD654", "status": "Processing"}],
            "subscription": {"start_date": "2026-03-15"},
        }
    }

    TASKS = {
        "easy": {"description": "Reset password", "customer_id": "CUST123"},
        "medium": {"description": "Check order status", "customer_id": "CUST123"},
        "hard": {"description": "Check refund eligibility", "customer_id": "CUST123"},
    }

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.current_task_id = "easy"

        self.accumulated_reward = 0.1
        self.is_closed = False

        self.found_kb = False
        self.found_customer = False

        self.search_results = []
        self.customer_details = {}
        self.last_response = ""

    def reset(self, task_id: str = None):
        if task_id:
            self.current_task_id = task_id

        self._state = State(episode_id=str(uuid4()), step_count=0)

        self.accumulated_reward = 0.1
        self.is_closed = False

        self.found_kb = False
        self.found_customer = False

        self.search_results = []
        self.customer_details = {}
        self.last_response = "New ticket assigned"

        return self._get_observation(reward=0.1, done=False)

    def step(self, action: FinalAction):
        if self.is_closed:
            return self._get_observation(reward=0.001, done=True)

        self._state.step_count += 1

        step_potential = -0.01
        done = False

        if action.action_type == "search_kb":
            self.search_results = ["kb_result"]
            if not self.found_kb:
                step_potential += 0.1
                self.found_kb = True

        elif action.action_type == "get_customer_details":
            self.customer_details = self.CUSTOMER_DB["CUST123"]
            if not self.found_customer:
                step_potential += 0.1
                self.found_customer = True

        elif action.action_type == "send_message":
            step_potential += 0.02

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

        # Compute new total
        new_total = self.accumulated_reward + step_potential
        target_total = max(0.1, min(0.9, new_total))

        step_reward = target_total - self.accumulated_reward

        # 🔥 CRITICAL FIXES
        step_reward = clamp(step_reward)
        self.accumulated_reward = clamp(target_total)

        return self._get_observation(reward=step_reward, done=done)

    def _get_observation(self, reward=0.001, done=False):
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
            info={"score": clamp(self.accumulated_reward)},
        )

    @property
    def state(self):
        return self._state

    def grader(self) -> float:
        return clamp(self.accumulated_reward)


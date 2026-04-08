from typing import Dict, Any, List
from openenv.core.env_server.interfaces import Environment
from uuid import uuid4
from pydantic import Field
from models import ShoppingCartAction, ShoppingCartObservation
from collections import defaultdict

class State:
    def __init__(self, episode_id: str, step_count: int, cart: Dict[str, Any]):
        self.episode_id = episode_id
        self.step_count = step_count
        self.cart = cart

class ShoppingCartEnvironment(Environment):
    _GLOBAL_TASK_INDEX = 0
    
    PRODUCTS = {
        "apple": {"price": 1.0, "stock": 100},
        "banana": {"price": 0.5, "stock": 100},
        "orange": {"price": 0.75, "stock": 100},
        "milk": {"price": 3.0, "stock": 50},
        "bread": {"price": 2.5, "stock": 50},
    }

    TASKS = {
        "easy": {
            "id": "task_easy",
            "description": "Add 2 apples to the cart and checkout.",
            "expected_cart": {"apple": 2},
            "expected_checkout": True,
        },
        "medium": {
            "id": "task_medium",
            "description": "Add 3 bananas, then remove 1 banana, then add 1 milk, and checkout.",
            "expected_cart": {"banana": 2, "milk": 1},
            "expected_checkout": True,
        },
        "hard": {
            "id": "task_hard",
            "description": "Add 5 oranges, then update orange quantity to 2, add 1 bread, and checkout. Ensure total price is correct.",
            "expected_cart": {"orange": 2, "bread": 1},
            "expected_checkout": True,
        },
    }

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0, cart=defaultdict(int))
        self.task_order = ["easy", "medium", "hard"]
        self.current_task_id = "easy"
        self.is_checked_out = False
        self.last_action_status = ""
        self.accumulated_reward = 0.001 # Start with non-zero
        self.history = []

    def reset(self, task_id: str = None) -> ShoppingCartObservation:
        if task_id and task_id in self.TASKS:
            self.current_task_id = task_id
        else:
            ShoppingCartEnvironment._GLOBAL_TASK_INDEX = (ShoppingCartEnvironment._GLOBAL_TASK_INDEX + 1) % len(self.task_order)
            self.current_task_id = self.task_order[ShoppingCartEnvironment._GLOBAL_TASK_INDEX]
        
        task = self.TASKS[self.current_task_id]
        
        self._state = State(episode_id=str(uuid4()), step_count=0, cart=defaultdict(int))
        self.is_checked_out = False
        self.last_action_status = f"New shopping cart task: {task['description']}"
        self.accumulated_reward = 0.001 # Start at 0.001
        self.history = []
        
        return self._get_observation(reward=0.001, done=False, info={})

    @property
    def state(self) -> State:
        return self._state

    def grader(self) -> float:
        return max(0.001, min(0.999, self.accumulated_reward))

    def _get_observation(self, reward: float = 0.0, done: bool = False, info: Dict = None) -> ShoppingCartObservation:
        current_items = []
        total_price = 0.0
        for item_id, quantity in self._state.cart.items():
            if quantity > 0 and item_id in self.PRODUCTS:
                price = self.PRODUCTS[item_id]["price"]
                current_items.append({"item_id": item_id, "quantity": quantity, "price": price})
                total_price += price * quantity

        return ShoppingCartObservation(
            cart_id=self._state.episode_id,
            items_in_cart=current_items,
            total_price=total_price,
            last_action_status=self.last_action_status,
            is_checked_out=self.is_checked_out,
            reward=reward,
            done=done,
            task_score=float(self.accumulated_reward),
            info=info or {}
        )

    def step(self, action: ShoppingCartAction) -> ShoppingCartObservation:
        if self.is_checked_out:
            return self._get_observation(reward=0.1, done=True, info={"score": self.accumulated_reward})

        self._state.step_count += 1
        self.history.append(action.dict())
        
        step_potential = -0.01 # Base step penalty
        done = False
        
        if action.action_type == "add_item":
            if action.item_id in self.PRODUCTS and action.quantity > 0:
                self._state.cart[action.item_id] += action.quantity
                self.last_action_status = f"Added {action.quantity} {action.item_id}(s)."
                step_potential += 0.05
            else:
                self.last_action_status = "Invalid item or quantity for add."
                step_potential -= 0.02
        
        elif action.action_type == "remove_item":
            if action.item_id in self.PRODUCTS and action.quantity > 0:
                if self._state.cart[action.item_id] >= action.quantity:
                    self._state.cart[action.item_id] -= action.quantity
                    self.last_action_status = f"Removed {action.quantity} {action.item_id}(s)."
                    step_potential += 0.05
                else:
                    self.last_action_status = "Cannot remove more than in cart."
                    step_potential -= 0.02
            else:
                self.last_action_status = "Invalid item or quantity for remove."
                step_potential -= 0.02

        elif action.action_type == "update_quantity":
            if action.item_id in self.PRODUCTS and action.quantity >= 0:
                self._state.cart[action.item_id] = action.quantity
                self.last_action_status = f"Updated {action.item_id} quantity to {action.quantity}."
                step_potential += 0.05
            else:
                self.last_action_status = "Invalid item or quantity for update."
                step_potential -= 0.02

        elif action.action_type == "view_cart":
            self.last_action_status = "Cart viewed."
            step_potential += 0.01

        elif action.action_type == "checkout":
            self.is_checked_out = True
            done = True
            task = self.TASKS[self.current_task_id]
            
            success = True
            # Check expected items in cart
            for item_id, quantity in task["expected_cart"].items():
                if self._state.cart[item_id] != quantity:
                    success = False
                    break
            # Check for unexpected items
            for item_id in self._state.cart:
                if item_id not in task["expected_cart"] and self._state.cart[item_id] > 0:
                    success = False
                    break
            
            if not self.is_checked_out: # Should be checked out to get success reward
                success = False

            if success:
                success_rewards = {"easy": 0.7, "medium": 0.5, "hard": 0.3}
                step_potential += success_rewards.get(self.current_task_id, 0.3)
                self.last_action_status = "Checkout successful and cart matches task!"
            else:
                step_potential -= 0.3
                self.last_action_status = "Checkout failed: Cart does not match task requirements."
        
        new_accumulated = self.accumulated_reward + step_potential
        target_total = max(0.001, min(0.999, new_accumulated))
        actual_step_reward = target_total - self.accumulated_reward
        self.accumulated_reward = target_total
        
        info = {"score": float(self.accumulated_reward)}
        return self._get_observation(reward=float(actual_step_reward), done=bool(done), info=info)

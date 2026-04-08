import os
import json
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

try:
    from client import ShoppingCartEnv, ShoppingCartAction, ShoppingCartObservation
except (ImportError, ModuleNotFoundError):
    try:
        from final.client import ShoppingCartEnv, ShoppingCartAction, ShoppingCartObservation
    except (ImportError, ModuleNotFoundError):
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent))
        from client import ShoppingCartEnv, ShoppingCartAction, ShoppingCartObservation

# Load environment variables from .env file
load_dotenv()

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
if "huggingface.co" in API_BASE_URL and not API_BASE_URL.endswith("/"):
    API_BASE_URL += "/"
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4-turbo-preview")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or HF_TOKEN
MOCK_MODE = os.getenv("MOCK") == "1"

if not OPENAI_API_KEY and not MOCK_MODE:
    print("Error: OPENAI_API_KEY or HF_TOKEN environment variable is required to run the inference script.")
    print("To test without an API key, set MOCK=1: export MOCK=1")
    exit(1)

if MOCK_MODE:
    print("Running in MOCK mode (no LLM calls will be made)")
    client = None
else:
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)

SYSTEM_PROMPT = """You are a shopping cart assistant. Your goal is to manage the shopping cart according to user requests.
Tools available:
- add_item(item_id, quantity): Add a specified quantity of an item to the cart.
- remove_item(item_id, quantity): Remove a specified quantity of an item from the cart.
- update_quantity(item_id, quantity): Set the quantity of an item in the cart to a specific value.
- view_cart(): View the current contents of the shopping cart.
- checkout(): Finalize the purchase and checkout.

Rules:
1. Always confirm the cart contents before checking out if the task involves multiple steps.
2. Your final action MUST be checkout.

Response format:
You must respond with a JSON object containing the action you want to take.
Example: {"action_type": "add_item", "item_id": "apple", "quantity": 2}
Example: {"action_type": "checkout"}
"""

def get_llm_action(messages: List[Dict[str, str]], task_id: str, step_count: int) -> Dict[str, Any]:
    if MOCK_MODE:
        if task_id == "easy":
            if step_count == 1: return {"action_type": "add_item", "item_id": "apple", "quantity": 2}
            if step_count == 2: return {"action_type": "checkout"}
        
        if task_id == "medium":
            if step_count == 1: return {"action_type": "add_item", "item_id": "banana", "quantity": 3}
            if step_count == 2: return {"action_type": "remove_item", "item_id": "banana", "quantity": 1}
            if step_count == 3: return {"action_type": "add_item", "item_id": "milk", "quantity": 1}
            if step_count == 4: return {"action_type": "checkout"}
        
        if task_id == "hard":
            if step_count == 1: return {"action_type": "add_item", "item_id": "orange", "quantity": 5}
            if step_count == 2: return {"action_type": "update_quantity", "item_id": "orange", "quantity": 2}
            if step_count == 3: return {"action_type": "add_item", "item_id": "bread", "quantity": 1}
            if step_count == 4: return {"action_type": "checkout"}
            
        return {"action_type": "checkout"}
    
    params = {"model": MODEL_NAME, "messages": messages}
    if "huggingface" not in API_BASE_URL:
        params["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**params)
    try:
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {"action_type": "checkout"}

def run_task(task_id: str, env_url: str):
    print(f"[START] Task: {task_id}")
    
    with ShoppingCartEnv(base_url=env_url).sync() as env:
        result = env.reset()
        
        obs = result.observation
        done = False
        step_count = 0
        
        total_reward = obs.reward if obs.reward is not None else 0.001
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"New Shopping Cart Task: {obs.last_action_status}"}
        ]

        while not done and step_count < 10:
            step_count += 1
            
            action_json = get_llm_action(messages, task_id, step_count)
            action = ShoppingCartAction(**action_json)
            
            result = env.step(action)
            obs = result.observation
            done = result.done
            
            reward = result.reward if result.reward is not None else obs.reward
            total_reward += reward
            
            print(f"[STEP] {step_count}: Action={action.action_type}, Reward={reward:.2f}, Task Score={obs.task_score:.2f}")
            
            messages.append({"role": "assistant", "content": json.dumps(action_json)})
            messages.append({"role": "user", "content": f"Observation: {obs.last_action_status}\nItems in Cart: {obs.items_in_cart}\nTotal Price: {obs.total_price:.2f}\nChecked Out: {obs.is_checked_out}"})

        final_score = obs.task_score
        if final_score < 0.001:
            final_score = 0.001
        if final_score > 0.999:
            final_score = 0.999
        print(f"[END] Task: {task_id}, Steps: {step_count}, Total Reward: {total_reward:.4f}, Final Score: {final_score:.4f}")
        return final_score

if __name__ == "__main__":
    ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")
    
    tasks = ["easy", "medium", "hard"]
    scores = {}
    
    for t in tasks:
        try:
            score = run_task(t, ENV_URL)
            scores[t] = 0.001 if score < 0.001 else (0.999 if score > 0.999 else score)
        except Exception as e:
            print(f"Error running task {t}: {e}")
            scores[t] = 0.001
            
    print("\nFinal Baseline Scores:")
    for t, s in scores.items():
        print(f"{t}: {s:.4f}")
import os
import json
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

try:
    from client import FinalEnv, FinalAction, FinalObservation
except (ImportError, ModuleNotFoundError):
    try:
        from final.client import FinalEnv, FinalAction, FinalObservation
    except (ImportError, ModuleNotFoundError):
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent))
        from client import FinalEnv, FinalAction, FinalObservation

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

SYSTEM_PROMPT = """You are a customer support agent. Your goal is to resolve the assigned ticket by using the available tools.
Tools available:
- search_kb(query): Search the internal knowledge base for articles.
- get_customer_details(customer_id): Retrieve details for a specific customer.
- send_message(message): Send a clarifying message to the customer.
- resolve_ticket(resolution): Close the ticket with a final resolution summary.

Rules:
1. Always search the KB or get customer details before resolving if you don't have enough info.
2. Be polite and professional.
3. Your final action MUST be resolve_ticket.

Response format:
You must respond with a JSON object containing the action you want to take.
Example: {"action_type": "search_kb", "query": "password reset"}
Example: {"action_type": "resolve_ticket", "resolution": "Instructed user to use the forgot password link."}
"""

def get_llm_action(messages: List[Dict[str, str]], task_id: str, step_count: int) -> Dict[str, Any]:
    if MOCK_MODE:
        # Simple heuristic-based mock actions for testing baseline differentiation
        if task_id == "easy":
            if step_count == 1: return {"action_type": "search_kb", "query": "password reset"}
            if step_count == 2: return {"action_type": "resolve_ticket", "resolution": "Go to settings > security > change password."}
        
        if task_id == "medium":
            if step_count == 1: return {"action_type": "get_customer_details", "customer_id": "CUST123"}
            if step_count == 2: return {"action_type": "send_message", "message": "I am checking your order status."}
            if step_count == 3: return {"action_type": "resolve_ticket", "resolution": "Your order ORD654 is Processing."}
        
        if task_id == "hard":
            if step_count == 1: return {"action_type": "search_kb", "query": "refund policy"}
            if step_count == 2: return {"action_type": "get_customer_details", "customer_id": "CUST123"}
            if step_count == 3: return {"action_type": "send_message", "message": "I see you started your subscription on 2026-03-15."}
            if step_count == 4: return {"action_type": "resolve_ticket", "resolution": "You are eligible for a refund because it is within 30 days."}
            
        return {"action_type": "resolve_ticket", "resolution": "Task completed."}
    
    params = {"model": MODEL_NAME, "messages": messages}
    if "huggingface" not in API_BASE_URL:
        params["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**params)
    try:
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {"action_type": "resolve_ticket", "resolution": "Task completed."}

def run_task(task_id: str, env_url: str):
    print(f"[START] Task: {task_id}")
    
    with FinalEnv(base_url=env_url).sync() as env:
        # Reset with specific task
        result = env.reset()
        
        obs = result.observation
        done = False
        step_count = 0
        
        # Start with the reward from reset (0.01)
        total_reward = obs.reward if obs.reward is not None else 0.01
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"New Ticket: {obs.ticket_description}\nCustomer ID: {obs.customer_id}"}
        ]

        while not done and step_count < 10:
            step_count += 1
            
            # Call LLM or Mock
            action_json = get_llm_action(messages, task_id, step_count)
            action = FinalAction(**action_json)
            
            # Step in env
            result = env.step(action)
            obs = result.observation
            done = result.done
            
            # Use observation's reward if step reward is None
            reward = result.reward if result.reward is not None else obs.reward
            total_reward += reward
            
            # Log step
            print(f"[STEP] {step_count}: Action={action.action_type}, Reward={reward:.2f}, Task Score={obs.task_score:.2f}")
            
            # Update messages for next step
            messages.append({"role": "assistant", "content": json.dumps(action_json)})
            messages.append({"role": "user", "content": f"Observation: {obs.last_response}\nKB Results: {obs.search_results}\nCustomer Details: {obs.customer_details}\nClosed: {obs.is_closed}"})

        # Use the authoritative task_score from the last observation
        final_score = obs.task_score
        print(f"[END] Task: {task_id}, Steps: {step_count}, Total Reward: {total_reward:.2f}, Final Score: {final_score:.2f}")
        return final_score

if __name__ == "__main__":
    # In a real scenario, the server would be running.
    # For baseline, we assume it's at http://localhost:8000
    ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")
    
    tasks = ["easy", "medium", "hard"]
    scores = {}
    
    for t in tasks:
        # Note: This baseline assumes the server can handle reset(task_id)
        # We might need to manually trigger the reset with task_id via a POST request 
        # if the client doesn't support it directly.
        try:
            score = run_task(t, ENV_URL)
            scores[t] = score
        except Exception as e:
            print(f"Error running task {t}: {e}")
            scores[t] = 0.0
            
    print("\nFinal Baseline Scores:")
    for t, s in scores.items():
        print(f"{t}: {s:.2f}")

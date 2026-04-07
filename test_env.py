from final.server.final_environment import FinalEnvironment
from final.models import FinalAction

def test_easy_task():
    env = FinalEnvironment()
    obs = env.reset(task_id="easy")
    print(f"Task: {obs.ticket_description}")
    
    # Action 1: Search KB
    res = env.step(FinalAction(action_type="search_kb", query="password reset"))
    print(f"Search Results: {res.search_results}")
    assert len(res.search_results) > 0
    assert res.reward > 0
    
    # Action 2: Resolve
    res = env.step(FinalAction(action_type="resolve_ticket", resolution="Go to settings > security > change password."))
    print(f"Resolution Reward: {res.reward}")
    assert res.done
    assert res.reward > 0.5
    print("Easy task passed!")

def test_medium_task():
    env = FinalEnvironment()
    obs = env.reset(task_id="medium")
    print(f"Task: {obs.ticket_description}")
    
    # Action 1: Get customer details
    res = env.step(FinalAction(action_type="get_customer_details", customer_id="CUST123"))
    print(f"Customer Details: {res.customer_details}")
    assert "orders" in res.customer_details
    
    # Action 2: Resolve with order status
    res = env.step(FinalAction(action_type="resolve_ticket", resolution="Your latest order ORD654 is Processing."))
    print(f"Resolution Reward: {res.reward}")
    assert res.done
    assert res.reward > 0.5
    print("Medium task passed!")

def test_hard_task():
    env = FinalEnvironment()
    obs = env.reset(task_id="hard")
    print(f"Task: {obs.ticket_description}")
    
    # Action 1: Search KB for refund policy
    res = env.step(FinalAction(action_type="search_kb", query="refund policy"))
    print(f"KB Results: {res.search_results}")
    
    # Action 2: Get customer details
    res = env.step(FinalAction(action_type="get_customer_details", customer_id="CUST123"))
    print(f"Customer Details: {res.customer_details}")
    
    # Action 3: Resolve
    res = env.step(FinalAction(action_type="resolve_ticket", resolution="You are eligible for a refund as your subscription started on 2026-03-15 (within 30 days)."))
    print(f"Resolution Reward: {res.reward}")
    assert res.done
    assert res.reward > 0.5
    print("Hard task passed!")

if __name__ == "__main__":
    try:
        test_easy_task()
        print("-" * 20)
        test_medium_task()
        print("-" * 20)
        test_hard_task()
        print("\nAll internal tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

from final.server.final_environment import FinalEnvironment
from final.models import FinalAction

def test_easy_task():
    env = FinalEnvironment()
    obs = env.reset(task_id="easy")
    print(f"Task: {obs.ticket_description}")
    
    total_reward = obs.reward
    # Action 1: Search KB
    res = env.step(FinalAction(action_type="search_kb", query="password reset"))
    total_reward += res.reward
    print(f"Search Results: {res.search_results}, Step Reward: {res.reward}")
    
    # Action 2: Resolve
    res = env.step(FinalAction(action_type="resolve_ticket", resolution="Go to settings > security > change password."))
    total_reward += res.reward
    print(f"Resolution Reward: {res.reward}, Total Reward: {total_reward}")
    
    assert res.done
    assert 0.0 < total_reward < 1.0
    print("Easy task passed!")

def test_failure_case():
    env = FinalEnvironment()
    obs = env.reset(task_id="easy")
    print(f"Task: {obs.ticket_description} (Failure case)")
    
    total_reward = obs.reward
    # Action 1: Wrong Resolve
    res = env.step(FinalAction(action_type="resolve_ticket", resolution="Wrong resolution"))
    total_reward += res.reward
    print(f"Resolution Reward: {res.reward}, Total Reward: {total_reward}")
    
    assert res.done
    assert 0.0 < total_reward < 1.0
    print("Failure case passed!")

def test_medium_task():
    env = FinalEnvironment()
    obs = env.reset(task_id="medium")
    print(f"Task: {obs.ticket_description}")
    
    total_reward = obs.reward
    # Action 1: Get customer details
    res = env.step(FinalAction(action_type="get_customer_details", customer_id="CUST123"))
    total_reward += res.reward
    print(f"Customer Details: {res.customer_details}, Step Reward: {res.reward}")
    
    # Action 2: Resolve with order status
    res = env.step(FinalAction(action_type="resolve_ticket", resolution="Your latest order ORD654 is Processing."))
    total_reward += res.reward
    print(f"Resolution Reward: {res.reward}, Total Reward: {total_reward}")
    
    assert res.done
    assert 0.0 < total_reward < 1.0
    print("Medium task passed!")

def test_hard_task():
    env = FinalEnvironment()
    obs = env.reset(task_id="hard")
    print(f"Task: {obs.ticket_description}")
    
    total_reward = obs.reward
    # Action 1: Search KB for refund policy
    res = env.step(FinalAction(action_type="search_kb", query="refund policy"))
    total_reward += res.reward
    print(f"KB Results: {res.search_results}, Step Reward: {res.reward}")
    
    # Action 2: Get customer details
    res = env.step(FinalAction(action_type="get_customer_details", customer_id="CUST123"))
    total_reward += res.reward
    print(f"Customer Details: {res.customer_details}, Step Reward: {res.reward}")
    
    # Action 3: Resolve
    res = env.step(FinalAction(action_type="resolve_ticket", resolution="You are eligible for a refund as your subscription started on 2026-03-15 (within 30 days)."))
    total_reward += res.reward
    print(f"Resolution Reward: {res.reward}, Total Reward: {total_reward}")
    
    assert res.done
    assert 0.0 < total_reward < 1.0
    print("Hard task passed!")

if __name__ == "__main__":
    try:
        test_easy_task()
        print("-" * 20)
        test_failure_case()
        print("-" * 20)
        test_medium_task()
        print("-" * 20)
        test_hard_task()
        print("\nAll internal tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

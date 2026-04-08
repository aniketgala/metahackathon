import pytest
from server.shopping_cart_environment import ShoppingCartEnvironment
from models import ShoppingCartAction

def test_easy_task():
    env = ShoppingCartEnvironment()
    obs = env.reset(task_id="easy")
    print(f"Task: {obs.last_action_status}")
    
    total_reward = obs.reward
    
    # Action 1: Add 2 apples
    res = env.step(ShoppingCartAction(action_type="add_item", item_id="apple", quantity=2))
    total_reward += res.reward
    print(f"Add Apple Reward: {res.reward:.2f}, Total Reward: {total_reward:.2f}")
    
    # Action 2: Checkout
    res = env.step(ShoppingCartAction(action_type="checkout"))
    total_reward += res.reward
    print(f"Checkout Reward: {res.reward:.2f}, Total Reward: {total_reward:.2f}")
    
    assert res.done
    assert 0.1 <= total_reward <= 0.9
    print("Easy task passed!")

def test_medium_task():
    env = ShoppingCartEnvironment()
    obs = env.reset(task_id="medium")
    print(f"Task: {obs.last_action_status}")
    
    total_reward = obs.reward
    
    # Action 1: Add 3 bananas
    res = env.step(ShoppingCartAction(action_type="add_item", item_id="banana", quantity=3))
    total_reward += res.reward
    print(f"Add Banana Reward: {res.reward:.2f}, Total Reward: {total_reward:.2f}")
    
    # Action 2: Remove 1 banana
    res = env.step(ShoppingCartAction(action_type="remove_item", item_id="banana", quantity=1))
    total_reward += res.reward
    print(f"Remove Banana Reward: {res.reward:.2f}, Total Reward: {total_reward:.2f}")
    
    # Action 3: Add 1 milk
    res = env.step(ShoppingCartAction(action_type="add_item", item_id="milk", quantity=1))
    total_reward += res.reward
    print(f"Add Milk Reward: {res.reward:.2f}, Total Reward: {total_reward:.2f}")
    
    # Action 4: Checkout
    res = env.step(ShoppingCartAction(action_type="checkout"))
    total_reward += res.reward
    print(f"Checkout Reward: {res.reward:.2f}, Total Reward: {total_reward:.2f}")
    
    assert res.done
    assert 0.1 <= total_reward <= 0.9
    print("Medium task passed!")

def test_hard_task():
    env = ShoppingCartEnvironment()
    obs = env.reset(task_id="hard")
    print(f"Task: {obs.last_action_status}")
    
    total_reward = obs.reward
    
    # Action 1: Add 5 oranges
    res = env.step(ShoppingCartAction(action_type="add_item", item_id="orange", quantity=5))
    total_reward += res.reward
    print(f"Add Orange Reward: {res.reward:.2f}, Total Reward: {total_reward:.2f}")
    
    # Action 2: Update orange quantity to 2
    res = env.step(ShoppingCartAction(action_type="update_quantity", item_id="orange", quantity=2))
    total_reward += res.reward
    print(f"Update Orange Reward: {res.reward:.2f}, Total Reward: {total_reward:.2f}")
    
    # Action 3: Add 1 bread
    res = env.step(ShoppingCartAction(action_type="add_item", item_id="bread", quantity=1))
    total_reward += res.reward
    print(f"Add Bread Reward: {res.reward:.2f}, Total Reward: {total_reward:.2f}")
    
    # Action 4: Checkout
    res = env.step(ShoppingCartAction(action_type="checkout"))
    total_reward += res.reward
    print(f"Checkout Reward: {res.reward:.2f}, Total Reward: {total_reward:.2f}")
    
    assert res.done
    assert 0.1 <= total_reward <= 0.9
    print("Hard task passed!")

if __name__ == "__main__":
    try:
        test_easy_task()
        print("-" * 20)
        test_medium_task()
        print("-" * 20)
        test_hard_task()
        print("-" * 20)
        print("All Shopping Cart tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")

#!/usr/bin/env python3
"""
Test script for PPO reinforcement learning implementation in advanced_risk.py
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from advanced_risk import (
    TradingEnvironment,
    ActorCriticNetwork,
    PPOAgent,
    RLPortfolioManager,
    HedgingStrategy,
    PositionSizer,
    AdvancedRiskManager
)

def test_trading_environment():
    """Test trading environment"""
    print("Testing TradingEnvironment...")
    
    env = TradingEnvironment(asset_count=5, initial_balance=10000)
    state = env.reset()
    
    assert len(state) == 5 + 5 + 5 + 2, "State dimension mismatch"
    assert env.balance == 10000, "Initial balance incorrect"
    
    action = np.ones(5) / 5
    next_state, reward, done, _ = env.step(action)
    
    assert len(next_state) == len(state), "State dimension changed"
    assert isinstance(reward, (int, float)), "Reward should be numeric"
    assert isinstance(done, bool), "Done should be boolean"
    
    print("✓ TradingEnvironment test passed")


def test_actor_critic_network():
    """Test actor-critic network"""
    print("Testing ActorCriticNetwork...")
    
    state_dim = 17
    action_dim = 5
    network = ActorCriticNetwork(state_dim, action_dim)
    
    state = torch.randn(4, state_dim)
    action_mean, value = network(state)
    
    assert action_mean.shape == (4, action_dim), "Action mean shape incorrect"
    assert value.shape == (4, 1), "Value shape incorrect"
    assert torch.allclose(action_mean.sum(dim=1), torch.ones(4), atol=1e-5), "Actions should sum to 1"
    
    action, logprob, entropy, value = network.get_action_and_value(state)
    
    assert action.shape == (4, action_dim), "Action shape incorrect"
    assert logprob.shape == (4,), "Logprob shape incorrect"
    assert entropy.shape == (4,), "Entropy shape incorrect"
    assert value.shape == (4,), "Value shape incorrect"
    
    print("✓ ActorCriticNetwork test passed")


def test_ppo_agent():
    """Test PPO agent"""
    print("Testing PPOAgent...")
    
    state_dim = 17
    action_dim = 5
    agent = PPOAgent(state_dim, action_dim)
    
    state = np.random.randn(state_dim).astype(np.float32)
    action = agent.select_action(state, training=True)
    
    assert action.shape == (action_dim,), "Action shape incorrect"
    assert len(agent.buffer_states) == 1, "State not stored in buffer"
    
    agent.store_transition(0.5, False)
    assert len(agent.buffer_rewards) == 1, "Reward not stored"
    
    for _ in range(19):
        action = agent.select_action(state, training=True)
        agent.store_transition(np.random.randn(), False)
    
    policy_loss, value_loss = agent.update()
    
    assert isinstance(policy_loss, float), "Policy loss should be float"
    assert isinstance(value_loss, float), "Value loss should be float"
    assert len(agent.buffer_states) == 0, "Buffer should be cleared after update"
    
    print("✓ PPOAgent test passed")


def test_ppo_agent_save_load():
    """Test PPO agent save/load"""
    print("Testing PPOAgent save/load...")
    
    state_dim = 17
    action_dim = 5
    agent1 = PPOAgent(state_dim, action_dim)
    
    os.makedirs('models', exist_ok=True)
    test_path = 'models/test_ppo_agent.pt'
    
    agent1.save(test_path)
    assert os.path.exists(test_path), "Model file not saved"
    
    agent2 = PPOAgent(state_dim, action_dim)
    loaded = agent2.load(test_path)
    assert loaded, "Failed to load model"
    
    agent1.policy.eval()
    agent2.policy.eval()
    
    state = torch.randn(state_dim)
    with torch.no_grad():
        action_mean1, value1 = agent1.policy(state)
        action_mean2, value2 = agent2.policy(state)
    
    assert torch.allclose(action_mean1, action_mean2, atol=1e-5), "Loaded model produces different action means"
    assert torch.allclose(value1, value2, atol=1e-5), "Loaded model produces different values"
    
    os.remove(test_path)
    print("✓ PPOAgent save/load test passed")


def test_rl_portfolio_manager():
    """Test RL portfolio manager"""
    print("Testing RLPortfolioManager...")
    
    manager = RLPortfolioManager(asset_count=5, model_path='models/test_model.pt')
    
    portfolio = np.array([100, 200, 150, 100, 50])
    market_conditions = {
        'volatility': 0.3,
        'trend': 0.1,
        'cash_ratio': 0.1,
        'portfolio_value_ratio': 1.05
    }
    
    new_weights = manager.rebalance(portfolio, market_conditions)
    
    assert new_weights.shape == (5,), "Weights shape incorrect"
    assert np.allclose(new_weights.sum(), 1.0, atol=1e-5), "Weights should sum to 1"
    assert np.all(new_weights >= 0), "Weights should be non-negative"
    
    leverage = manager.compute_leverage(0.3)
    assert manager.leverage_range[0] <= leverage <= manager.leverage_range[1], "Leverage out of range"
    
    print("✓ RLPortfolioManager test passed")


def test_kelly_criterion():
    """Test Kelly criterion"""
    print("Testing Kelly Criterion...")
    
    manager = RLPortfolioManager(asset_count=5)
    
    f1 = manager.apply_kelly_criterion(0.6, 2.0, 1.0)
    assert 0.01 <= f1 <= 0.25, "Kelly fraction out of bounds"
    
    f2 = manager.apply_kelly_criterion(0.5, 1.5, 1.0)
    assert 0.01 <= f2 <= 0.25, "Kelly fraction out of bounds"
    
    f3 = manager.apply_kelly_criterion(0.6, 2.0, 0.0)
    assert f3 == 0.01, "Should return minimum when avg_loss is 0"
    
    f4 = manager.apply_kelly_criterion(0.0, 2.0, 1.0)
    assert f4 == 0.01, "Should return minimum when win_rate is 0"
    
    print("✓ Kelly Criterion test passed")


def test_hedging_strategy():
    """Test hedging strategy"""
    print("Testing HedgingStrategy...")
    
    hedger = HedgingStrategy(max_drawdown_threshold=0.05)
    
    ratio1 = hedger.compute_hedge_ratio(0.02)
    assert ratio1 == 0.0, "Should not hedge below threshold"
    
    ratio2 = hedger.compute_hedge_ratio(0.08)
    assert ratio2 == 0.5, "Should hedge above threshold"
    
    result = hedger.execute_hedge(1000, 0.08, None)
    assert result['hedge_executed'] == True, "Hedge should be executed"
    assert result['ratio'] == 0.5, "Hedge ratio incorrect"
    
    print("✓ HedgingStrategy test passed")


def test_position_sizer():
    """Test position sizer"""
    print("Testing PositionSizer...")
    
    sizer = PositionSizer()
    
    size = sizer.calculate_size(
        account_balance=10000,
        atr=2.0,
        entry_price=100,
        stop_loss_points=5
    )
    
    assert size > 0, "Position size should be positive"
    risk = size * 5 * 100
    assert risk <= 10000 * 0.05 * 1.01, "Risk exceeds max drawdown"
    
    size_zero_sl = sizer.calculate_size(10000, 2.0, 100, 0)
    assert size_zero_sl == 100, "Should return 1% of balance when stop loss is 0"
    
    print("✓ PositionSizer test passed")


def test_advanced_risk_manager():
    """Test advanced risk manager integration"""
    print("Testing AdvancedRiskManager...")
    
    import asyncio
    import glob
    
    for model_file in glob.glob('models/*.pt'):
        try:
            os.remove(model_file)
        except:
            pass
    
    async def run_test():
        manager = AdvancedRiskManager()
        
        market_state = {
            'volatility': 0.4,
            'trend': 0.05
        }
        
        portfolio = {
            'btc': 0.5,
            'eth': 0.3,
            'current_drawdown': 0.02
        }
        
        result = await manager.manage_risk(market_state, portfolio)
        
        assert 'rebalance_weights' in result, "Missing rebalance_weights"
        assert 'leverage' in result, "Missing leverage"
        assert 'hedge' in result, "Missing hedge"
        assert 'max_drawdown_remaining' in result, "Missing max_drawdown_remaining"
        
        assert len(result['rebalance_weights']) == 10, "Incorrect weights length"
        assert result['leverage'] >= 1.0, "Leverage too low"
        
        return result
    
    result = asyncio.run(run_test())
    print("✓ AdvancedRiskManager test passed")


def test_mini_training():
    """Test mini training loop"""
    print("Testing mini training loop...")
    
    import glob
    
    for model_file in glob.glob('models/*.pt'):
        try:
            os.remove(model_file)
        except:
            pass
    
    manager = RLPortfolioManager(asset_count=3, model_path='models/test_mini_model.pt')
    
    episode_rewards = manager.train(episodes=10, update_interval=5)
    
    assert len(episode_rewards) == 10, "Wrong number of episodes"
    assert all(isinstance(r, (int, float)) for r in episode_rewards), "Rewards should be numeric"
    
    manager.agent.save('models/test_mini_model.pt')
    assert os.path.exists('models/test_mini_model.pt'), "Model not saved"
    
    os.remove('models/test_mini_model.pt')
    print("✓ Mini training test passed")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*50)
    print("Running PPO Implementation Tests")
    print("="*50 + "\n")
    
    tests = [
        test_trading_environment,
        test_actor_critic_network,
        test_ppo_agent,
        test_ppo_agent_save_load,
        test_rl_portfolio_manager,
        test_kelly_criterion,
        test_hedging_strategy,
        test_position_sizer,
        test_advanced_risk_manager,
        test_mini_training
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
    
    print("\n" + "="*50)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*50 + "\n")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

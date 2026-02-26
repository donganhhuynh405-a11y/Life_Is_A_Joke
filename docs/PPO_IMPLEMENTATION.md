# PPO Reinforcement Learning Implementation for Advanced Risk Management

## Overview

This implementation provides a production-ready **Proximal Policy Optimization (PPO)** reinforcement learning agent for dynamic portfolio risk management. The system uses an actor-critic architecture to learn optimal asset allocation strategies through interaction with a simulated trading environment.

## Architecture

### 1. Trading Environment (`TradingEnvironment`)

A fully-featured trading simulation environment that provides:

- **State Representation**: Portfolio weights, asset returns, volatility measures, cash ratio, and portfolio value
- **Action Space**: Continuous asset allocation weights (0 to 1) for each asset
- **Reward Function**: Combines returns, volatility penalty, drawdown penalty, and transaction costs
- **Price Simulation**: Realistic price movements with configurable volatility

```python
env = TradingEnvironment(
    asset_count=10,
    initial_balance=10000,
    transaction_cost=0.001
)
```

### 2. Actor-Critic Network (`ActorCriticNetwork`)

Neural network architecture with shared layers and separate heads:

- **Shared Layers**: 2-layer MLP (256 hidden units) for feature extraction
- **Actor Head**: Outputs portfolio weights (softmax distribution)
- **Critic Head**: Outputs state value estimate
- **Action Distribution**: Normal distribution with learnable log-std

```python
network = ActorCriticNetwork(
    state_dim=32,      # Depends on number of assets
    action_dim=10,     # Number of assets
    hidden_dim=256
)
```

### 3. PPO Agent (`PPOAgent`)

Core RL agent implementing the PPO algorithm:

**Key Features:**
- Clipped surrogate objective for stable learning
- Generalized Advantage Estimation (GAE)
- Multiple epoch updates with experience replay
- Gradient clipping for stability
- Automatic model checkpointing

**Hyperparameters:**
```python
agent = PPOAgent(
    state_dim=32,
    action_dim=10,
    lr=3e-4,              # Learning rate
    gamma=0.99,           # Discount factor
    epsilon_clip=0.2,     # PPO clipping parameter
    k_epochs=4,           # Update epochs per training step
    device='cuda'         # GPU support
)
```

### 4. Portfolio Manager (`RLPortfolioManager`)

High-level interface for portfolio management:

```python
manager = RLPortfolioManager(
    asset_count=10,
    leverage_range=(1, 10),
    model_path='models/ppo_agent.pt'
)

# Training
episode_rewards = manager.train(episodes=1000, update_interval=20)

# Inference
new_weights = manager.rebalance(
    portfolio=current_positions,
    market_conditions={
        'volatility': 0.3,
        'trend': 0.05,
        'cash_ratio': 0.1,
        'portfolio_value_ratio': 1.05
    }
)
```

## Training

### Quick Start

```bash
# Train a new agent (1000 episodes)
python train_ppo_agent.py --episodes 1000 --asset-count 10

# Train with custom parameters
python train_ppo_agent.py \
    --episodes 2000 \
    --asset-count 5 \
    --update-interval 20 \
    --model-path models/custom_agent.pt

# Evaluate existing model
python train_ppo_agent.py \
    --evaluate \
    --model-path models/ppo_agent.pt
```

### Training Process

1. **Initialization**: Agent starts with random policy
2. **Experience Collection**: Agent interacts with environment for `update_interval` steps
3. **Policy Update**: PPO algorithm updates policy using collected experiences
4. **Checkpointing**: Model saved every 50 episodes
5. **Evaluation**: Final performance metrics computed after training

### Training Metrics

- **Episode Reward**: Cumulative reward per episode
- **Moving Average**: 50-episode rolling average
- **Policy Loss**: Actor loss (clipped objective)
- **Value Loss**: Critic loss (MSE)

## Production Integration

### Basic Usage

```python
from advanced_risk import AdvancedRiskManager

# Initialize risk manager
risk_manager = AdvancedRiskManager()

# In your trading loop
market_state = {
    'volatility': compute_volatility(),
    'trend': compute_trend()
}

portfolio = {
    'btc': current_btc_position,
    'eth': current_eth_position,
    'current_drawdown': compute_drawdown()
}

# Get risk management decisions
decisions = await risk_manager.manage_risk(market_state, portfolio)

# Apply decisions
new_weights = decisions['rebalance_weights']
leverage = decisions['leverage']
hedge_info = decisions['hedge']
```

### Components

#### 1. Dynamic Leverage Calculation

Adjusts leverage based on market volatility:

```python
leverage = manager.compute_leverage(volatility=0.3)
# Returns value between leverage_range (e.g., 1 to 10)
```

#### 2. Kelly Criterion Position Sizing

Optimal position sizing based on win/loss statistics:

```python
kelly_fraction = manager.apply_kelly_criterion(
    win_rate=0.6,
    avg_win=2.0,
    avg_loss=1.0
)
# Returns fraction of capital to risk (1-25%)
```

#### 3. Hedging Strategy

Automatic hedging when drawdown exceeds threshold:

```python
hedger = HedgingStrategy(max_drawdown_threshold=0.05)
hedge_result = hedger.execute_hedge(
    position_size=1000,
    drawdown=0.08,
    exchange_client=exchange
)
```

#### 4. Position Sizing

ATR-based position sizing with risk limits:

```python
sizer = PositionSizer()
position_size = sizer.calculate_size(
    account_balance=10000,
    atr=2.0,
    entry_price=100,
    stop_loss_points=5
)
```

## Reward Function Design

The reward function balances multiple objectives:

```python
reward = returns - volatility_penalty - drawdown_penalty - cost_penalty

where:
- returns: Portfolio value change normalized by initial value
- volatility_penalty: 0.1 × recent portfolio volatility
- drawdown_penalty: 2.0 × current drawdown (if > 5%)
- cost_penalty: 10.0 × transaction costs / portfolio value
```

This encourages:
- ✅ Positive returns
- ✅ Low volatility (stable performance)
- ✅ Limited drawdowns (capital preservation)
- ✅ Minimal trading costs (efficiency)

## Model Persistence

### Saving

```python
agent.save('models/ppo_agent.pt')
```

Saves:
- Policy network weights
- Optimizer state
- All hyperparameters

### Loading

```python
agent.load('models/ppo_agent.pt')
```

Automatically:
- Restores network weights
- Restores optimizer state
- Sets model to evaluation mode

## Performance Monitoring

### Evaluation Metrics

The evaluation script computes:

1. **Average Return**: Mean portfolio return across episodes
2. **Average Max Drawdown**: Mean maximum drawdown observed
3. **Average Sharpe Ratio**: Risk-adjusted return metric
4. **Win Rate**: Percentage of profitable episodes
5. **Profit Factor**: Ratio of total wins to total losses

### Example Output

```
Evaluation Results:
  Average Return: 12.45%
  Average Max Drawdown: 8.32%
  Average Sharpe Ratio: 1.85
  Win Rate: 65%
  Profit Factor: 2.1
```

## Advanced Features

### 1. Generalized Advantage Estimation (GAE)

Reduces variance in policy gradient estimates:

```python
advantages, returns = agent.compute_gae(next_value)
```

### 2. Gradient Clipping

Prevents exploding gradients during training:

```python
torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
```

### 3. PPO Clipping

Ensures policy updates stay within trust region:

```python
ratios = torch.exp(logprobs - old_logprobs)
surr1 = ratios * advantages
surr2 = torch.clamp(ratios, 1 - epsilon_clip, 1 + epsilon_clip) * advantages
policy_loss = -torch.min(surr1, surr2).mean()
```

### 4. Entropy Regularization

Encourages exploration:

```python
loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
```

## Testing

Comprehensive test suite included:

```bash
# Run all tests
python test_ppo_implementation.py

# Expected output:
# ✓ TradingEnvironment test passed
# ✓ ActorCriticNetwork test passed
# ✓ PPOAgent test passed
# ✓ PPOAgent save/load test passed
# ✓ RLPortfolioManager test passed
# ✓ Kelly Criterion test passed
# ✓ HedgingStrategy test passed
# ✓ PositionSizer test passed
# ✓ AdvancedRiskManager test passed
# ✓ Mini training test passed
```

## Requirements

```
torch>=2.6.0
numpy>=1.24.0
pandas>=2.0.0
```

Optional:
```
matplotlib>=3.0.0  # For plotting training results
```

## Best Practices

### 1. Training

- Start with 1000+ episodes for initial training
- Use 2000-5000 episodes for production models
- Monitor moving average reward for convergence
- Save checkpoints frequently
- Test on validation environment before production

### 2. Hyperparameter Tuning

Key parameters to tune:
- `learning_rate`: 1e-4 to 1e-3
- `gamma`: 0.95 to 0.99
- `epsilon_clip`: 0.1 to 0.3
- `update_interval`: 10 to 50 steps

### 3. Production Deployment

- Load pre-trained model
- Use `training=False` during inference
- Monitor performance metrics continuously
- Retrain periodically with new data
- Implement fail-safes for extreme market conditions

### 4. Risk Management

- Always validate rebalancing weights sum to 1.0
- Enforce position limits independent of RL agent
- Implement circuit breakers for rapid drawdowns
- Use Kelly criterion for additional position sizing validation

## Troubleshooting

### Poor Training Performance

1. Reduce learning rate
2. Increase update interval
3. Adjust reward function weights
4. Increase network capacity

### Model Not Converging

1. Check environment reward signals
2. Verify state normalization
3. Increase training episodes
4. Adjust epsilon_clip parameter

### High Transaction Costs

1. Increase transaction cost penalty in reward
2. Reduce update frequency
3. Implement minimum rebalancing threshold

## References

- Schulman et al. (2017): "Proximal Policy Optimization Algorithms"
- Mnih et al. (2016): "Asynchronous Methods for Deep Reinforcement Learning"
- Jiang et al. (2017): "A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem"

## License

This implementation is part of the life_is_a_joke trading bot project.

## Support

For issues or questions:
1. Check the test suite for usage examples
2. Review training logs for debugging
3. Consult the code documentation

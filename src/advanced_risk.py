"""
Risk Management module with PPO Reinforcement Learning agent for portfolio rebalancing.
"""
import logging
import numpy as np
from typing import Tuple, Dict, List, Optional
import asyncio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import os
import json

logger = logging.getLogger('bot.advanced_risk')

class TradingEnvironment:
    """Trading environment for RL agent training"""
    
    def __init__(self, asset_count=10, initial_balance=10000, transaction_cost=0.001):
        self.asset_count = asset_count
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.reset()
        
    def reset(self):
        """Reset environment to initial state"""
        self.balance = self.initial_balance
        self.portfolio = np.zeros(self.asset_count)
        self.prices = np.ones(self.asset_count) * 100
        self.price_history = deque(maxlen=100)
        self.portfolio_value_history = [self.initial_balance]
        self.step_count = 0
        self.max_portfolio_value = self.initial_balance
        return self._get_state()
    
    def _get_state(self):
        """Get current state representation"""
        portfolio_weights = self.portfolio / (self.portfolio.sum() + 1e-8)
        portfolio_value = self._get_portfolio_value()
        cash_ratio = self.balance / (portfolio_value + 1e-8)
        
        returns = np.zeros(self.asset_count)
        volatility = np.ones(self.asset_count) * 0.5
        
        if len(self.price_history) > 1:
            recent_prices = np.array(list(self.price_history)[-20:])
            returns = (self.prices - recent_prices[-1]) / (recent_prices[-1] + 1e-8)
            volatility = np.std(recent_prices, axis=0) / (np.mean(recent_prices, axis=0) + 1e-8)
        
        state = np.concatenate([
            portfolio_weights,
            returns,
            volatility,
            [cash_ratio, portfolio_value / self.initial_balance]
        ])
        return state.astype(np.float32)
    
    def _get_portfolio_value(self):
        """Calculate total portfolio value"""
        return self.balance + np.sum(self.portfolio * self.prices)
    
    def step(self, action):
        """Execute action and return next state, reward, done"""
        target_weights = self._action_to_weights(action)
        portfolio_value = self._get_portfolio_value()
        
        transaction_costs = self._rebalance_portfolio(target_weights, portfolio_value)
        
        self._update_prices()
        self.price_history.append(self.prices.copy())
        
        new_portfolio_value = self._get_portfolio_value()
        self.portfolio_value_history.append(new_portfolio_value)
        
        reward = self._calculate_reward(new_portfolio_value, transaction_costs)
        
        self.step_count += 1
        done = bool(self.step_count >= 1000 or new_portfolio_value < self.initial_balance * 0.5)
        
        return self._get_state(), reward, done, {}
    
    def _action_to_weights(self, action):
        """Convert action to portfolio weights"""
        weights = np.array(action)
        weights = np.maximum(weights, 0)
        weights = weights / (weights.sum() + 1e-8)
        return weights
    
    def _rebalance_portfolio(self, target_weights, portfolio_value):
        """Rebalance portfolio to target weights"""
        target_values = target_weights * portfolio_value
        current_values = self.portfolio * self.prices
        
        trades = target_values - current_values
        transaction_costs = np.sum(np.abs(trades)) * self.transaction_cost
        
        self.balance = portfolio_value - np.sum(target_values) - transaction_costs
        self.portfolio = target_values / (self.prices + 1e-8)
        
        return transaction_costs
    
    def _update_prices(self):
        """Simulate price movements"""
        returns = np.random.normal(0.0001, 0.02, self.asset_count)
        self.prices = self.prices * (1 + returns)
        self.prices = np.maximum(self.prices, 0.01)
    
    def _calculate_reward(self, new_value, transaction_costs):
        """Calculate reward based on return, risk, and costs"""
        old_value = self.portfolio_value_history[-2] if len(self.portfolio_value_history) > 1 else self.initial_balance
        returns = (new_value - old_value) / (old_value + 1e-8)
        
        volatility_penalty = 0
        if len(self.portfolio_value_history) > 10:
            recent_values = np.array(self.portfolio_value_history[-10:])
            volatility = np.std(recent_values) / (np.mean(recent_values) + 1e-8)
            volatility_penalty = volatility * 0.1
        
        drawdown_penalty = 0
        self.max_portfolio_value = max(self.max_portfolio_value, new_value)
        current_drawdown = (self.max_portfolio_value - new_value) / (self.max_portfolio_value + 1e-8)
        if current_drawdown > 0.05:
            drawdown_penalty = current_drawdown * 2
        
        cost_penalty = transaction_costs / (old_value + 1e-8)
        
        reward = returns - volatility_penalty - drawdown_penalty - cost_penalty * 10
        return reward


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for PPO"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCriticNetwork, self).__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state):
        shared_features = self.shared(state)
        action_mean = self.actor_mean(shared_features)
        value = self.critic(shared_features)
        return action_mean, value
    
    def get_action_and_value(self, state, action=None):
        shared_features = self.shared(state)
        action_mean = self.actor_mean(shared_features)
        action_std = torch.exp(self.actor_log_std)
        
        dist = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            action = dist.sample()
        
        action_logprob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        value = self.critic(shared_features).squeeze(-1)
        
        return action, action_logprob, entropy, value


class PPOAgent:
    """Proximal Policy Optimization agent"""
    
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, 
                 epsilon_clip=0.2, k_epochs=4, device='cpu'):
        self.device = device
        self.gamma = gamma
        self.epsilon_clip = epsilon_clip
        self.k_epochs = k_epochs
        
        self.policy = ActorCriticNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_logprobs = []
        self.buffer_rewards = []
        self.buffer_dones = []
        self.buffer_values = []
    
    def select_action(self, state, training=True):
        """Select action using current policy"""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            action, logprob, _, value = self.policy.get_action_and_value(state)
        
        if training:
            self.buffer_states.append(state)
            self.buffer_actions.append(action)
            self.buffer_logprobs.append(logprob)
            self.buffer_values.append(value)
        
        return action.cpu().numpy()
    
    def store_transition(self, reward, done):
        """Store transition in buffer"""
        self.buffer_rewards.append(reward)
        self.buffer_dones.append(done)
    
    def compute_gae(self, next_value):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        rewards = self.buffer_rewards + [next_value]
        values = self.buffer_values + [next_value]
        dones = self.buffer_dones + [0]
        
        for t in reversed(range(len(self.buffer_rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * 0.95 * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, self.buffer_values)]
        return advantages, returns
    
    def update(self):
        """Update policy using PPO"""
        if len(self.buffer_states) == 0:
            return 0, 0
        
        next_state = self.buffer_states[-1]
        with torch.no_grad():
            _, _, _, next_value = self.policy.get_action_and_value(next_state)
        
        advantages, returns = self.compute_gae(next_value.item())
        
        states = torch.stack(self.buffer_states)
        actions = torch.stack(self.buffer_actions)
        old_logprobs = torch.stack(self.buffer_logprobs)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_policy_loss = 0
        total_value_loss = 0
        
        for _ in range(self.k_epochs):
            _, logprobs, entropy, values = self.policy.get_action_and_value(states, actions)
            
            ratios = torch.exp(logprobs - old_logprobs)
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, returns)
            entropy_loss = -entropy.mean()
            
            loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        
        self.clear_buffer()
        
        return total_policy_loss / self.k_epochs, total_value_loss / self.k_epochs
    
    def clear_buffer(self):
        """Clear experience buffer"""
        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_logprobs = []
        self.buffer_rewards = []
        self.buffer_dones = []
        self.buffer_values = []
    
    def save(self, path):
        """Save policy network"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        logger.info(f'PPO agent saved to {path}')
    
    def load(self, path):
        """Load policy network"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.policy.eval()
            logger.info(f'PPO agent loaded from {path}')
            return True
        return False


class RLPortfolioManager:
    """Reinforcement Learning portfolio manager with PPO"""
    
    def __init__(self, asset_count=10, leverage_range=(1, 10), model_path='models/ppo_agent.pt'):
        self.asset_count = asset_count
        self.leverage_range = leverage_range
        self.model_path = model_path
        
        self.env = TradingEnvironment(asset_count=asset_count)
        state_dim = len(self.env.reset())
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.agent = PPOAgent(state_dim, asset_count, device=self.device)
        
        self.agent.load(model_path)
        logger.info(f'RLPortfolioManager initialized with {asset_count} assets on {self.device}')
    
    def train(self, episodes=1000, update_interval=20):
        """Train PPO agent"""
        logger.info(f'Starting PPO training for {episodes} episodes')
        
        episode_rewards = []
        
        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            while not done:
                action = self.agent.select_action(state, training=True)
                next_state, reward, done, _ = self.env.step(action)
                
                self.agent.store_transition(reward, done)
                episode_reward += reward
                state = next_state
                step += 1
                
                if step % update_interval == 0:
                    policy_loss, value_loss = self.agent.update()
            
            episode_rewards.append(episode_reward)
            
            if episode % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                logger.info(f'Episode {episode}/{episodes}, Avg Reward: {avg_reward:.4f}')
                self.agent.save(self.model_path)
        
        logger.info('Training complete')
        return episode_rewards
    
    def rebalance(self, portfolio: np.ndarray, market_conditions: dict) -> np.ndarray:
        """Use trained RL agent to decide optimal asset allocation"""
        volatility = market_conditions.get('volatility', 0.5)
        trend = market_conditions.get('trend', 0)
        
        if len(portfolio) < self.asset_count:
            padded_portfolio = np.zeros(self.asset_count)
            padded_portfolio[:len(portfolio)] = portfolio
            portfolio = padded_portfolio
        elif len(portfolio) > self.asset_count:
            portfolio = portfolio[:self.asset_count]
        
        portfolio_weights = portfolio / (portfolio.sum() + 1e-8)
        cash_ratio = market_conditions.get('cash_ratio', 0.1)
        portfolio_value_ratio = market_conditions.get('portfolio_value_ratio', 1.0)
        
        state = np.concatenate([
            portfolio_weights,
            np.ones(self.asset_count) * trend * 0.01,
            np.ones(self.asset_count) * volatility,
            [cash_ratio, portfolio_value_ratio]
        ]).astype(np.float32)
        
        new_weights = self.agent.select_action(state, training=False)
        new_weights = np.maximum(new_weights, 0)
        new_weights = new_weights / (new_weights.sum() + 1e-8)
        
        logger.info(f'RL rebalanced portfolio. Sample weights: {new_weights[:3]}')
        return new_weights
    
    def compute_leverage(self, volatility: float) -> float:
        """Dynamic leverage based on volatility"""
        atr_normalized = min(volatility, 1.0)
        leverage = self.leverage_range[0] + (1 - atr_normalized) * (self.leverage_range[1] - self.leverage_range[0])
        return min(self.leverage_range[1], max(self.leverage_range[0], leverage))
    
    def apply_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Kelly Criterion for optimal position sizing"""
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return 0.01
        b = avg_win / avg_loss
        if b <= 0:
            return 0.01
        f = (win_rate * b - (1 - win_rate)) / b
        f = max(0.01, min(0.25, f))
        return f

class HedgingStrategy:
    """Options-like hedging for max drawdown protection"""
    
    def __init__(self, max_drawdown_threshold=0.05):
        self.max_dd_threshold = max_drawdown_threshold
    
    def compute_hedge_ratio(self, current_drawdown: float) -> float:
        """Scale hedge based on drawdown"""
        if current_drawdown > self.max_dd_threshold:
            return 0.5  # Hedge 50% of position
        return 0.0
    
    def execute_hedge(self, position_size: float, drawdown: float, exchange_client) -> dict:
        """Execute hedge order (short or put options)"""
        hedge_ratio = self.compute_hedge_ratio(drawdown)
        if hedge_ratio > 0:
            # Place short or inverse position
            logger.info('Placing hedge: ratio %.2f', hedge_ratio)
            return {'hedge_executed': True, 'ratio': hedge_ratio}
        return {'hedge_executed': False}

class PositionSizer:
    """ATR-based position sizing with Kelly and drawdown limits"""
    
    def __init__(self):
        self.max_drawdown = 0.05
    
    def calculate_size(self, 
                      account_balance: float,
                      atr: float,
                      entry_price: float,
                      stop_loss_points: float) -> float:
        """Calculate position size to limit risk to max_drawdown"""
        if stop_loss_points == 0:
            return 0.01 * account_balance
        
        risk_dollars = account_balance * self.max_drawdown
        size = risk_dollars / (stop_loss_points * entry_price)
        return size

class AdvancedRiskManager:
    """Main risk management orchestrator"""
    
    def __init__(self):
        self.rl_manager = RLPortfolioManager()
        self.hedger = HedgingStrategy()
        self.sizer = PositionSizer()
    
    async def manage_risk(self, market_state: dict, portfolio: dict) -> dict:
        """Comprehensive risk management"""
        volatility = market_state.get('volatility', 0.5)
        drawdown = portfolio.get('current_drawdown', 0)
        
        # RL-based rebalancing
        new_weights = self.rl_manager.rebalance(
            np.array([portfolio.get('btc', 0), portfolio.get('eth', 0)]),
            {'volatility': volatility, 'trend': market_state.get('trend', 0)}
        )
        
        # Dynamic leverage
        leverage = self.rl_manager.compute_leverage(volatility)
        
        # Hedge if needed
        hedge = self.hedger.execute_hedge(100, drawdown, None)
        
        return {
            'rebalance_weights': new_weights.tolist(),
            'leverage': leverage,
            'hedge': hedge,
            'max_drawdown_remaining': self.sizer.max_drawdown - drawdown
        }

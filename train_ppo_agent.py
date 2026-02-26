#!/usr/bin/env python3
"""
Training script for PPO reinforcement learning agent for risk management.
"""
import os
import sys
import logging
import argparse
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from advanced_risk import RLPortfolioManager, TradingEnvironment, PPOAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def plot_training_results(episode_rewards, save_path='models/training_results.png'):
    """Plot training progress"""
    if not HAS_MATPLOTLIB:
        logger.warning('matplotlib not available, skipping plot generation')
        return
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    window_size = 50
    if len(episode_rewards) >= window_size:
        moving_avg = np.convolve(episode_rewards, 
                                 np.ones(window_size)/window_size, 
                                 mode='valid')
        plt.plot(moving_avg)
        plt.xlabel('Episode')
        plt.ylabel('Moving Average Reward')
        plt.title(f'{window_size}-Episode Moving Average')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f'Training plot saved to {save_path}')


def evaluate_agent(manager, num_episodes=10):
    """Evaluate trained agent performance"""
    logger.info(f'Evaluating agent over {num_episodes} episodes')
    
    total_returns = []
    max_drawdowns = []
    sharpe_ratios = []
    
    for episode in range(num_episodes):
        state = manager.env.reset()
        done = False
        episode_values = []
        
        while not done:
            action = manager.agent.select_action(state, training=False)
            next_state, reward, done, _ = manager.env.step(action)
            episode_values.append(manager.env._get_portfolio_value())
            state = next_state
        
        episode_values = np.array(episode_values)
        total_return = (episode_values[-1] - episode_values[0]) / episode_values[0]
        
        max_value = np.maximum.accumulate(episode_values)
        drawdowns = (max_value - episode_values) / max_value
        max_drawdown = np.max(drawdowns)
        
        returns = np.diff(episode_values) / episode_values[:-1]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        
        total_returns.append(total_return)
        max_drawdowns.append(max_drawdown)
        sharpe_ratios.append(sharpe)
    
    logger.info('Evaluation Results:')
    logger.info(f'  Average Return: {np.mean(total_returns)*100:.2f}%')
    logger.info(f'  Average Max Drawdown: {np.mean(max_drawdowns)*100:.2f}%')
    logger.info(f'  Average Sharpe Ratio: {np.mean(sharpe_ratios):.2f}')
    
    return {
        'avg_return': np.mean(total_returns),
        'avg_max_drawdown': np.mean(max_drawdowns),
        'avg_sharpe': np.mean(sharpe_ratios)
    }


def main():
    parser = argparse.ArgumentParser(description='Train PPO agent for portfolio management')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--asset-count', type=int, default=10, help='Number of assets')
    parser.add_argument('--update-interval', type=int, default=20, help='Policy update interval')
    parser.add_argument('--model-path', type=str, default='models/ppo_agent.pt', 
                        help='Path to save/load model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate existing model')
    parser.add_argument('--plot', action='store_true', help='Plot training results')
    
    args = parser.parse_args()
    
    os.makedirs('models', exist_ok=True)
    
    logger.info('Initializing PPO Portfolio Manager')
    manager = RLPortfolioManager(
        asset_count=args.asset_count,
        model_path=args.model_path
    )
    
    if args.evaluate:
        if os.path.exists(args.model_path):
            logger.info(f'Loading model from {args.model_path}')
            manager.agent.load(args.model_path)
            evaluate_agent(manager)
        else:
            logger.error(f'Model not found at {args.model_path}')
            return
    else:
        logger.info(f'Starting training for {args.episodes} episodes')
        episode_rewards = manager.train(
            episodes=args.episodes,
            update_interval=args.update_interval
        )
        
        manager.agent.save(args.model_path)
        
        if args.plot:
            plot_training_results(episode_rewards)
        
        logger.info('Training complete. Evaluating final model...')
        evaluate_agent(manager)
    
    logger.info('Done!')


if __name__ == '__main__':
    main()

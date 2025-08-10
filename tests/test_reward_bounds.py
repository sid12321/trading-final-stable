#!/usr/bin/env python3

import sys
import os
import numpy as np
import pandas as pd

# Add the RL-1 path for importing the environment
sys.path.append(os.path.join(os.path.dirname(__file__), 'RL-1'))

from parameters import *
from env.StockTradingEnv2 import StockTradingEnv2

def create_test_data():
    """Create a small dataset for testing"""
    np.random.seed(42)
    n_steps = 50
    
    # Create synthetic price data
    price_base = 100
    returns = np.random.normal(0, 0.02, n_steps)
    prices = [price_base]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    prices = np.array(prices[1:])
    
    # Create basic technical indicators
    sma5 = pd.Series(prices).rolling(5, min_periods=1).mean()
    volatility = pd.Series(prices).pct_change().rolling(10, min_periods=1).std()
    
    # Create the dataframe with required columns
    test_df = pd.DataFrame({
        'currentt': pd.date_range('2023-01-01 09:15:00', periods=n_steps, freq='T'),
        'vwap2': prices,
        'c': prices,
        'o': prices * 0.999,
        'h': prices * 1.001,
        'l': prices * 0.998,
        'v': np.random.randint(1000, 10000, n_steps),
        'sma5': sma5,
        'volatility': volatility.fillna(0.01),
        'bear_signal': np.random.choice([0, 1], n_steps, p=[0.8, 0.2]),
        'momentum': np.random.normal(0, 0.1, n_steps),
        'rsi': np.random.uniform(20, 80, n_steps)
    })
    
    # Add lag columns
    for lag in [1, 2, 3]:
        test_df[f'lret{lag}'] = test_df['vwap2'].pct_change(lag).fillna(0)
    
    test_df['currentdate'] = test_df['currentt'].dt.date
    
    return test_df

def test_reward_bounds():
    """Test that all rewards are strictly in [-0.1, 0.1]"""
    print("Testing reward bounds...")
    
    # Create test data
    test_df = create_test_data()
    test_signals = ['sma5', 'volatility', 'bear_signal', 'momentum', 'rsi', 'lret1', 'lret2', 'lret3']
    
    # Create environment directly (no vectorization)
    env = StockTradingEnv2(
        test_df, 
        NLAGS=5, 
        NUMVARS=len(test_signals), 
        MAXIMUM_SHORT_VALUE=MAXIMUM_SHORT_VALUE,
        INITIAL_ACCOUNT_BALANCE=INITIAL_ACCOUNT_BALANCE, 
        MAX_STEPS=20,  # Short episodes
        finalsignalsp=test_signals, 
        INITIAL_NET_WORTH=INITIAL_ACCOUNT_BALANCE, 
        INITIAL_SHARES_HELD=0
    )
    
    # Test multiple episodes
    all_rewards = []
    
    for episode in range(5):
        print(f"\nTesting episode {episode + 1}")
        obs, _ = env.reset()
        episode_rewards = []
        
        for step in range(20):  # Maximum steps
            # Random action
            action = np.random.uniform(-1, 1, 2)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_rewards.append(reward)
            all_rewards.append(reward)
            
            print(f"  Step {step}: Action={action[0]:.3f}, Reward={reward:.6f}")
            
            # Check bounds
            if reward < -0.1 or reward > 0.1:
                print(f"  ERROR: Reward {reward} is outside [-0.1, 0.1] bounds!")
            
            if terminated or truncated:
                print(f"  Episode ended at step {step}")
                if 'r' in info:
                    episode_reward = info['r']
                    print(f"  Episode reward: {episode_reward:.6f}")
                    if episode_reward < -0.1 or episode_reward > 0.1:
                        print(f"  ERROR: Episode reward {episode_reward} is outside [-0.1, 0.1] bounds!")
                break
    
    # Statistics
    all_rewards = np.array(all_rewards)
    print(f"\nReward statistics:")
    print(f"  Total rewards tested: {len(all_rewards)}")
    print(f"  Min reward: {np.min(all_rewards):.6f}")
    print(f"  Max reward: {np.max(all_rewards):.6f}")
    print(f"  Mean reward: {np.mean(all_rewards):.6f}")
    print(f"  Std reward: {np.std(all_rewards):.6f}")
    
    # Check if all are in bounds
    out_of_bounds = (all_rewards < -0.1) | (all_rewards > 0.1)
    if np.any(out_of_bounds):
        print(f"  ERROR: {np.sum(out_of_bounds)} rewards are outside [-0.1, 0.1] bounds!")
        print(f"  Out of bounds rewards: {all_rewards[out_of_bounds]}")
    else:
        print("  SUCCESS: All rewards are within [-0.1, 0.1] bounds!")

if __name__ == "__main__":
    test_reward_bounds()
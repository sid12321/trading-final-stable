#!/usr/bin/env python3

import sys
import os
import numpy as np
import pandas as pd

# Add the RL-1 path for importing the environment
sys.path.append(os.path.join(os.path.dirname(__file__), 'RL-1'))

from parameters import *
from common import *

def create_test_data():
    """Create a small dataset for testing"""
    np.random.seed(42)
    n_steps = 200  # Small dataset for quick testing
    
    # Create synthetic price data
    price_base = 100
    returns = np.random.normal(0, 0.02, n_steps)  # 2% daily volatility
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
        'o': prices * 0.999,  # Slightly lower open
        'h': prices * 1.001,  # Slightly higher high
        'l': prices * 0.998,  # Slightly lower low
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

def test_reward_logging():
    """Test the reward logging with a quick training run"""
    print("Setting up test environment...")
    
    # Create test data
    test_df = create_test_data()
    print(f"Created test dataset with {len(test_df)} rows")
    
    # Define simple signal list
    test_signals = ['sma5', 'volatility', 'bear_signal', 'momentum', 'rsi', 'lret1', 'lret2', 'lret3']
    
    print("Setting up environment...")
    
    # Import environment locally to avoid import issues
    from env.StockTradingEnv2 import StockTradingEnv2
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    
    # Create environment
    def make_env():
        return StockTradingEnv2(
            test_df, 
            NLAGS=5, 
            NUMVARS=len(test_signals), 
            MAXIMUM_SHORT_VALUE=MAXIMUM_SHORT_VALUE,
            INITIAL_ACCOUNT_BALANCE=INITIAL_ACCOUNT_BALANCE, 
            MAX_STEPS=15,  # Very short episodes to ensure completion within rollout
            finalsignalsp=test_signals, 
            INITIAL_NET_WORTH=INITIAL_ACCOUNT_BALANCE, 
            INITIAL_SHARES_HELD=0
        )
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=False, norm_reward=False, clip_reward=None, gamma=GAMMA)
    
    print("Creating model...")
    
    # Create simple PPO model
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=1e-3,
        n_steps=32,  # Small steps for quick testing, should allow episodes to complete
        batch_size=32,
        n_epochs=3,
        device=DEVICE
    )
    
    print("Creating custom logger callback...")
    
    # Create custom logger
    custom_logger = CustomLoggerCallback(verbose=2, log_freq=32)  # Log every 32 steps
    
    print("Starting training...")
    
    # Train for a very short time
    model.learn(
        total_timesteps=500,  # Very small number for testing
        callback=custom_logger,
        log_interval=1
    )
    
    print("Training completed!")
    
    # Check if log file was created and has data
    log_file = os.path.join(tmp_path, METRICS_FILE)
    if os.path.exists(log_file):
        print(f"\nLog file created: {log_file}")
        with open(log_file, 'r') as f:
            lines = f.readlines()
            print(f"Log file has {len(lines)} lines")
            if len(lines) > 1:
                print("Sample log entries:")
                for i, line in enumerate(lines):
                    print(f"  Line {i}: {line.strip()}")
                    if i >= 5:  # Show first 5 lines
                        break
    else:
        print(f"Log file not found: {log_file}")

if __name__ == "__main__":
    test_reward_logging()
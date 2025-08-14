#\!/usr/bin/env python3
"""
Analyze reward components in StockTradingEnvOptimized
Captures 100 samples of actual and scaled values for each component
"""

import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append('/Users/skumar81/Desktop/Personal/trading-final-stable')

from StockTradingEnvOptimized import StockTradingEnvOptimized
from parameters import *

def collect_reward_samples(n_samples=100):
    """Collect samples of reward components"""
    
    # Load sample data
    df = pd.read_csv(f"{basepath}/traindata/finalmldfBPCL.csv")
    df = df.drop(['t'], axis=1, errors='ignore')
    
    # Get signals from data
    finalsignals = df.columns[~df.columns.isin(['currentt', 'currento', 'currentdate', 'vwap2'])].tolist()
    
    # Create environment
    env = StockTradingEnvOptimized(
        df=df[-5000:],  # Use last 5000 rows for testing
        NLAGS=5,
        NUMVARS=len(finalsignals),
        finalsignalsp=finalsignals
    )
    
    # Storage for samples
    samples = []
    
    # Reset environment
    obs, _ = env.reset()
    
    print(f"Collecting {n_samples} samples of reward components...")
    
    # Track portfolio values for Sharpe calculation
    portfolio_history = []
    
    for i in range(n_samples):
        # Take random action
        action = np.array([
            np.random.uniform(-1, 1),  # Buy/sell/hold
            np.random.uniform(0.1, 1)   # Amount
        ])
        
        # Store current state for reward calculation
        prev_net_worth = env.net_worth
        prev_shares = env.shares_held
        prev_balance = env.balance
        current_price = env._get_current_price()
        
        # Step environment
        obs, reward, done, truncated, info = env.step(action)
        
        # Track portfolio value
        portfolio_history.append(env.net_worth)
        
        # Calculate reward components (mimicking _calculate_reward logic)
        # Profit component
        net_worth_change = env.net_worth - prev_net_worth
        profit_reward = net_worth_change / env.INITIAL_ACCOUNT_BALANCE
        
        # Step progress component  
        step_reward = 0.001 * (env.current_step / env.MAX_STEPS)
        
        # Action quality component
        action_type = action[0]
        if abs(action_type) < 0.3:
            action_reward = 0
        else:
            if action_type > 0:  # Buy
                if env.shares_held > prev_shares:
                    action_reward = 0.01
                else:
                    action_reward = -0.005
            else:  # Sell
                if env.shares_held < prev_shares:
                    action_reward = 0.01
                else:
                    action_reward = -0.005
        
        # Position efficiency
        position_value = abs(env.shares_held * current_price)
        max_position = env.INITIAL_ACCOUNT_BALANCE
        position_ratio = position_value / max_position
        position_reward = 0.005 * position_ratio if position_ratio < 0.8 else -0.01 * (position_ratio - 0.8)
        
        # Risk penalty
        if position_ratio > 0.9:
            risk_penalty = -0.02 * (position_ratio - 0.9)
        else:
            risk_penalty = 0
        
        # Activity penalty
        activity_penalty = -0.001 if abs(action_type) > 0.3 else 0
        
        # Sharpe-like reward (simplified)
        if len(portfolio_history) > 1:
            recent_history = portfolio_history[-10:] if len(portfolio_history) >= 10 else portfolio_history
            returns = np.diff(recent_history) / (np.array(recent_history[:-1]) + 1e-10)
            if len(returns) > 0:
                sharpe_reward = np.mean(returns) / (np.std(returns) + 1e-10)
            else:
                sharpe_reward = 0
        else:
            sharpe_reward = 0
        
        # Scale components (from StockTradingEnvOptimized)
        def scale_reward_component(value, scale_factor=1.0):
            if value <= 0:
                return 0.0
            scaled = np.tanh(value * scale_factor)
            return np.clip(scaled, -1.0, 1.0)
        
        def scale_penalty_component(value, scale_factor=1.0):
            if value >= 0:
                return 0.0
            scaled = np.tanh(value * scale_factor)
            return np.clip(scaled, -1.0, 1.0)
        
        # Apply scaling (using scale factors from StockTradingEnvOptimized lines 512-518)
        scaled_profit = scale_reward_component(profit_reward, scale_factor=0.25)
        scaled_step = scale_reward_component(step_reward, scale_factor=0.15)
        scaled_action = scale_reward_component(action_reward, scale_factor=0.1)
        scaled_position = scale_reward_component(position_reward, scale_factor=0.05)
        scaled_risk = scale_penalty_component(risk_penalty, scale_factor=0.05)
        scaled_activity = scale_penalty_component(activity_penalty, scale_factor=0.1)
        scaled_sharpe = scale_reward_component(sharpe_reward, scale_factor=0.2)
        
        # Store sample
        sample = {
            'sample_id': i + 1,
            # Actual values
            'actual_profit': profit_reward,
            'actual_step': step_reward,
            'actual_action': action_reward,
            'actual_position': position_reward,
            'actual_risk': risk_penalty,
            'actual_activity': activity_penalty,
            'actual_sharpe': sharpe_reward,
            # Scaled values
            'scaled_profit': scaled_profit,
            'scaled_step': scaled_step,
            'scaled_action': scaled_action,
            'scaled_position': scaled_position,
            'scaled_risk': scaled_risk,
            'scaled_activity': scaled_activity,
            'scaled_sharpe': scaled_sharpe,
            # Total reward
            'total_reward': reward,
            # Additional context
            'action_type': action_type,
            'action_amount': action[1],
            'shares_held': env.shares_held,
            'balance': env.balance,
            'net_worth': env.net_worth,
            'current_price': current_price
        }
        samples.append(sample)
        
        if done or truncated:
            obs, _ = env.reset()
            portfolio_history = []  # Reset history on environment reset
            print(f"  Environment reset at sample {i+1}")
        
        if (i + 1) % 20 == 0:
            print(f"  Collected {i+1}/{n_samples} samples")
    
    return pd.DataFrame(samples)

def analyze_variance(df):
    """Analyze variance in the collected samples"""
    print("\n" + "="*60)
    print("VARIANCE ANALYSIS")
    print("="*60)
    
    components = ['profit', 'step', 'action', 'position', 'risk', 'activity', 'sharpe']
    
    for comp in components:
        actual_col = f'actual_{comp}'
        scaled_col = f'scaled_{comp}'
        
        actual_std = df[actual_col].std()
        scaled_std = df[scaled_col].std()
        actual_range = df[actual_col].max() - df[actual_col].min()
        scaled_range = df[scaled_col].max() - df[scaled_col].min()
        
        print(f"\n{comp.upper()}:")
        print(f"  Actual - Mean: {df[actual_col].mean():.6f}, Std: {actual_std:.6f}, Range: {actual_range:.6f}")
        print(f"  Scaled - Mean: {df[scaled_col].mean():.6f}, Std: {scaled_std:.6f}, Range: {scaled_range:.6f}")
        print(f"  Variance ratio (scaled/actual): {scaled_std/actual_std:.2f}" if actual_std > 0 else "  Actual has no variance")

if __name__ == "__main__":
    # Collect samples
    df_samples = collect_reward_samples(100)
    
    # Save to CSV
    output_file = "reward_components_analysis.csv"
    df_samples.to_csv(output_file, index=False)
    print(f"\n✅ Saved {len(df_samples)} samples to {output_file}")
    
    # Analyze variance
    analyze_variance(df_samples)
    
    # Show summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Select only numeric columns for statistics
    numeric_cols = [col for col in df_samples.columns if col not in ['sample_id']]
    print(df_samples[numeric_cols].describe())
    
    print(f"\n📊 Full data saved to: {output_file}")
    print("You can open this CSV file to analyze the variance and distribution of each component.")

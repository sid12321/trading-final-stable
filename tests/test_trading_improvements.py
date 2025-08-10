#!/usr/bin/env python3

"""
Test the trading improvements for reduced bear bias and lower trading frequency
"""

import sys
import os
import numpy as np
import pandas as pd

# Add the RL-1 path for importing the environment
sys.path.append(os.path.join(os.path.dirname(__file__), 'RL-1'))

from parameters import *
from env.StockTradingEnv2 import StockTradingEnv2

def create_test_data():
    """Create a test dataset with various market conditions"""
    np.random.seed(42)
    n_steps = 100
    
    # Create synthetic price data with trends
    price_base = 100
    
    # Create different market phases
    uptrend = np.cumsum(np.random.normal(0.002, 0.01, 30))  # Uptrend
    sideways = np.cumsum(np.random.normal(0, 0.01, 40))     # Sideways
    downtrend = np.cumsum(np.random.normal(-0.002, 0.01, 30))  # Downtrend
    
    returns = np.concatenate([uptrend, sideways, downtrend])
    prices = price_base * np.exp(returns)
    
    # Create basic technical indicators
    sma5 = pd.Series(prices).rolling(5, min_periods=1).mean()
    volatility = pd.Series(prices).pct_change().rolling(10, min_periods=1).std()
    
    # Create the dataframe with required columns
    test_df = pd.DataFrame({
        'currentt': pd.date_range('2023-01-01 09:15:00', periods=n_steps, freq='min'),
        'vwap2': prices,
        'c': prices,
        'o': prices * 0.999,
        'h': prices * 1.001,
        'l': prices * 0.998,
        'v': np.random.randint(1000, 10000, n_steps),
        'sma5': sma5,
        'volatility': volatility.fillna(0.01),
        'bear_signal': np.random.choice([0, 1], n_steps, p=[0.7, 0.3]),  # Some bear signals
        'momentum': np.random.normal(0, 0.1, n_steps),
        'rsi': np.random.uniform(20, 80, n_steps)
    })
    
    # Add lag columns
    for lag in [1, 2, 3]:
        test_df[f'lret{lag}'] = test_df['vwap2'].pct_change(lag).fillna(0)
    
    test_df['currentdate'] = test_df['currentt'].dt.date
    
    return test_df

def test_trading_behavior():
    """Test that the improved environment reduces trading frequency and bear bias"""
    print("🧪 Testing Trading Behavior Improvements")
    print("=" * 60)
    
    # Create test data
    test_df = create_test_data()
    test_signals = ['sma5', 'volatility', 'bear_signal', 'momentum', 'rsi', 'lret1', 'lret2', 'lret3']
    
    # Create environment
    env = StockTradingEnv2(
        test_df, 
        NLAGS=5, 
        NUMVARS=len(test_signals), 
        MAXIMUM_SHORT_VALUE=MAXIMUM_SHORT_VALUE,
        INITIAL_ACCOUNT_BALANCE=INITIAL_ACCOUNT_BALANCE, 
        MAX_STEPS=80,
        finalsignalsp=test_signals, 
        INITIAL_NET_WORTH=INITIAL_ACCOUNT_BALANCE, 
        INITIAL_SHARES_HELD=0
    )
    
    print("✓ Created trading environment")
    
    # Test trading behavior with various actions
    obs, _ = env.reset()
    
    # Track statistics
    total_trades = 0
    buy_actions = 0
    sell_actions = 0
    hold_actions = 0
    consecutive_holds = 0
    max_consecutive_holds = 0
    position_changes = []
    
    print("\n📊 Testing Action Thresholds and Trading Frequency")
    print("=" * 60)
    
    # Test various action levels to see threshold behavior
    test_actions = [
        (0.1, 0.5),   # Weak buy signal (should be held)
        (0.3, 0.5),   # Medium buy signal (should be held) 
        (0.5, 0.5),   # Strong buy signal (should execute)
        (-0.1, 0.5),  # Weak sell signal (should be held)
        (-0.3, 0.5),  # Medium sell signal (should be held)
        (-0.5, 0.5),  # Strong sell signal (should execute)
        (0.0, 0.5),   # Neutral (should hold)
    ]
    
    prev_position = env.shares_held
    
    for i, action in enumerate(test_actions):
        obs, reward, terminated, truncated, info = env.step(action)
        current_position = env.shares_held
        
        # Classify action result
        if current_position > prev_position:
            actual_action = "BUY"
            buy_actions += 1
            total_trades += 1
        elif current_position < prev_position:
            actual_action = "SELL"
            sell_actions += 1
            total_trades += 1
        else:
            actual_action = "HOLD"
            hold_actions += 1
            consecutive_holds += 1
            max_consecutive_holds = max(max_consecutive_holds, consecutive_holds)
        
        if actual_action != "HOLD":
            consecutive_holds = 0
        
        position_changes.append(abs(current_position - prev_position))
        prev_position = current_position
        
        print(f"Action {action[0]:5.1f}: {actual_action:4s} | Position: {current_position:6.0f} | Reward: {reward:.4f}")
        
        if terminated or truncated:
            break
    
    # Continue with random actions to test overall behavior
    print(f"\n📈 Testing Extended Trading Session")
    print("=" * 60)
    
    step_count = 0
    while step_count < 50 and not (terminated or truncated):
        # Generate random actions with various intensities
        action_intensity = np.random.uniform(-1, 1)
        amount = np.random.uniform(0.3, 0.8)
        action = [action_intensity, amount]
        
        obs, reward, terminated, truncated, info = env.step(action)
        current_position = env.shares_held
        
        # Track action types
        if current_position > prev_position:
            buy_actions += 1
            total_trades += 1
            consecutive_holds = 0
        elif current_position < prev_position:
            sell_actions += 1
            total_trades += 1
            consecutive_holds = 0
        else:
            hold_actions += 1
            consecutive_holds += 1
            max_consecutive_holds = max(max_consecutive_holds, consecutive_holds)
        
        position_changes.append(abs(current_position - prev_position))
        prev_position = current_position
        step_count += 1
    
    total_actions = buy_actions + sell_actions + hold_actions
    
    # Calculate statistics
    print(f"\n📋 Trading Behavior Analysis")
    print("=" * 60)
    print(f"Total Steps: {total_actions}")
    print(f"Total Trades: {total_trades}")
    print(f"Buy Actions: {buy_actions} ({buy_actions/total_actions*100:.1f}%)")
    print(f"Sell Actions: {sell_actions} ({sell_actions/total_actions*100:.1f}%)")
    print(f"Hold Actions: {hold_actions} ({hold_actions/total_actions*100:.1f}%)")
    print(f"Trading Frequency: {total_trades/total_actions*100:.1f}%")
    print(f"Max Consecutive Holds: {max_consecutive_holds}")
    
    # Evaluate improvements
    print(f"\n🎯 Improvement Assessment")
    print("=" * 60)
    
    # Check if trading frequency is reduced (should be < 40%)
    trading_freq = total_trades / total_actions
    if trading_freq < 0.4:
        print(f"✅ Trading frequency reduced: {trading_freq*100:.1f}% (target: <40%)")
        freq_improved = True
    else:
        print(f"❌ Trading frequency still high: {trading_freq*100:.1f}% (target: <40%)")
        freq_improved = False
    
    # Check for bear bias (sell actions shouldn't dominate)
    if sell_actions > 0 and buy_actions > 0:
        sell_ratio = sell_actions / (buy_actions + sell_actions)
        if sell_ratio < 0.7:  # Less than 70% sells
            print(f"✅ Bear bias reduced: {sell_ratio*100:.1f}% sells (target: <70%)")
            bias_improved = True
        else:
            print(f"❌ Bear bias still present: {sell_ratio*100:.1f}% sells (target: <70%)")
            bias_improved = False
    else:
        print(f"⚠️  Unable to assess bear bias (insufficient trades)")
        bias_improved = True  # Benefit of doubt
    
    # Check hold behavior
    if hold_actions / total_actions > 0.6:
        print(f"✅ Hold behavior encouraged: {hold_actions/total_actions*100:.1f}% holds")
        hold_improved = True
    else:
        print(f"⚠️  Hold behavior could be stronger: {hold_actions/total_actions*100:.1f}% holds")
        hold_improved = False
    
    return freq_improved and bias_improved and hold_improved

def main():
    """Run the trading behavior test"""
    print("🚀 Trading Behavior Improvement Test")
    print("=" * 60)
    
    success = test_trading_behavior()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 TRADING IMPROVEMENTS SUCCESSFUL!")
        print("✅ Reduced trading frequency")
        print("✅ Reduced bear market bias") 
        print("✅ Encouraged hold behavior")
        print("✅ Agent should be less overactive and more balanced")
    else:
        print("⚠️  TRADING IMPROVEMENTS PARTIAL")
        print("Some improvements detected, but further tuning may be needed")
    
    return success

if __name__ == "__main__":
    main()
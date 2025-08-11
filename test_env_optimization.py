#!/usr/bin/env python3
"""
Test script to compare performance between standard and optimized environments
"""

import time
import numpy as np
import pandas as pd
from parameters import *
from StockTradingEnv2 import StockTradingEnv2
from StockTradingEnvOptimized import StockTradingEnvOptimized

def test_environment_speed():
    """Compare speed of standard vs optimized environment"""
    
    print("🔧 Testing Environment Optimization Performance")
    print("=" * 60)
    
    # Load test data
    test_symbol = 'BPCL'
    try:
        df = pd.read_csv(f'{basepath}/traindata/finalmldf{test_symbol}.csv')
        print(f"✅ Loaded data for {test_symbol}: {len(df)} rows")
    except FileNotFoundError:
        print(f"❌ Data file not found for {test_symbol}")
        print("Creating synthetic test data...")
        # Create synthetic data for testing
        n_rows = 1000
        n_signals = 20
        
        dates = pd.date_range(start='2023-01-01', periods=n_rows, freq='5min')
        df = pd.DataFrame({
            'currentdate': dates.date,
            'vwap2': np.random.randn(n_rows).cumsum() + 100,
            'Open': np.random.randn(n_rows).cumsum() + 100,
            'High': np.random.randn(n_rows).cumsum() + 102,
            'Low': np.random.randn(n_rows).cumsum() + 98,
            'Close': np.random.randn(n_rows).cumsum() + 100,
        })
        
        # Add signal columns
        signal_names = [f'signal_{i}' for i in range(n_signals)]
        for sig in signal_names:
            df[sig] = np.random.randn(n_rows)
        
        print(f"✅ Created synthetic data: {len(df)} rows, {n_signals} signals")
    
    # Get signal columns (all columns except standard OHLC and date/time columns)
    exclude_cols = ['currentdate', 'Open', 'High', 'Low', 'Close', 'vwap2', 'Volume', 
                   'Datetime', 'datetime', 'Date', 'date', 'Time', 'time', 'index']
    
    # Get numeric columns only (exclude any string/datetime columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    finalsignalsp = [col for col in numeric_cols if col not in exclude_cols][:20]  # Limit to 20 signals
    
    print(f"📊 Using {len(finalsignalsp)} signals for testing")
    
    # Test parameters
    n_steps = 100
    n_episodes = 5
    
    # Test standard environment
    print("\n📈 Testing Standard Environment (StockTradingEnv2)")
    print("-" * 40)
    
    env_standard = StockTradingEnv2(
        df, 
        NLAGS=5, 
        NUMVARS=len(finalsignalsp),
        MAXIMUM_SHORT_VALUE=INITIAL_ACCOUNT_BALANCE,
        finalsignalsp=finalsignalsp
    )
    
    # Warm-up run
    obs = env_standard.reset()[0]
    
    # Time standard environment
    start_time = time.time()
    total_steps = 0
    
    for episode in range(n_episodes):
        obs = env_standard.reset()[0]
        for step in range(n_steps):
            action = env_standard.action_space.sample()
            obs, reward, terminated, truncated, info = env_standard.step(action)
            total_steps += 1
            if terminated or truncated:
                break
    
    standard_time = time.time() - start_time
    standard_steps_per_sec = total_steps / standard_time
    
    print(f"⏱️  Total time: {standard_time:.3f} seconds")
    print(f"📊 Total steps: {total_steps}")
    print(f"⚡ Steps/second: {standard_steps_per_sec:.1f}")
    
    # Test optimized environment
    print("\n🚀 Testing Optimized Environment (StockTradingEnvOptimized)")
    print("-" * 40)
    
    env_optimized = StockTradingEnvOptimized(
        df, 
        NLAGS=5, 
        NUMVARS=len(finalsignalsp),
        MAXIMUM_SHORT_VALUE=INITIAL_ACCOUNT_BALANCE,
        finalsignalsp=finalsignalsp
    )
    
    # Warm-up run
    obs = env_optimized.reset()[0]
    
    # Time optimized environment
    start_time = time.time()
    total_steps = 0
    
    for episode in range(n_episodes):
        obs = env_optimized.reset()[0]
        for step in range(n_steps):
            action = env_optimized.action_space.sample()
            obs, reward, terminated, truncated, info = env_optimized.step(action)
            total_steps += 1
            if terminated or truncated:
                break
    
    optimized_time = time.time() - start_time
    optimized_steps_per_sec = total_steps / optimized_time
    
    print(f"⏱️  Total time: {optimized_time:.3f} seconds")
    print(f"📊 Total steps: {total_steps}")
    print(f"⚡ Steps/second: {optimized_steps_per_sec:.1f}")
    
    # Calculate speedup
    print("\n📈 Performance Comparison")
    print("=" * 60)
    speedup = standard_time / optimized_time
    speedup_pct = (speedup - 1) * 100
    
    print(f"🎯 Speedup: {speedup:.2f}x")
    print(f"📊 Performance improvement: {speedup_pct:.1f}%")
    print(f"⚡ Standard env: {standard_steps_per_sec:.1f} steps/sec")
    print(f"🚀 Optimized env: {optimized_steps_per_sec:.1f} steps/sec")
    
    # Test observation consistency
    print("\n🔍 Testing Observation Consistency")
    print("-" * 40)
    
    # Reset both environments
    obs_standard = env_standard.reset()[0]
    obs_optimized = env_optimized.reset()[0]
    
    # Check if observations match
    obs_match = np.allclose(obs_standard, obs_optimized, rtol=1e-5)
    print(f"✅ Initial observations match: {obs_match}")
    
    # Take same actions and compare
    for i in range(5):
        action = env_standard.action_space.sample()
        obs_s, reward_s, _, _, _ = env_standard.step(action)
        obs_o, reward_o, _, _, _ = env_optimized.step(action)
        
        obs_match = np.allclose(obs_s, obs_o, rtol=1e-5)
        reward_match = np.isclose(reward_s, reward_o, rtol=1e-5)
        
        print(f"  Step {i+1}: Obs match={obs_match}, Reward match={reward_match}")
    
    print("\n✅ Environment optimization test complete!")
    
    return {
        'standard_time': standard_time,
        'optimized_time': optimized_time,
        'speedup': speedup,
        'speedup_pct': speedup_pct
    }

if __name__ == "__main__":
    results = test_environment_speed()
    
    print("\n" + "=" * 60)
    print("🎉 OPTIMIZATION SUMMARY")
    print("=" * 60)
    if results['speedup'] > 1.0:
        print(f"✅ Optimized environment is {results['speedup']:.2f}x faster!")
        print(f"   Performance gain: {results['speedup_pct']:.1f}%")
    else:
        print(f"⚠️  No significant speedup detected")
        print(f"   Consider profiling to identify bottlenecks")
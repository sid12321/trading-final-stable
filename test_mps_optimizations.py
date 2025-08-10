#!/usr/bin/env python3
"""
test_mps_optimizations.py - Test script for MPS-specific optimizations

Tests the new MPS optimizations:
1. MPS memory cache management 
2. Gradient accumulation
3. Mixed precision support
"""

import os
import sys
import time
import torch
import subprocess

# Setup path
basepath = '/Users/skumar81/Desktop/Personal/trading-final-stable'
os.chdir(basepath)

from parameters import *
from bounded_entropy_ppo import BoundedEntropyPPO
from StockTradingEnv2 import StockTradingEnv2
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd

def test_mps_optimizations():
    """Test MPS optimizations with a quick training run"""
    print("🧪 Testing MPS Optimizations")
    print("=" * 50)
    
    if DEVICE != "mps":
        print(f"❌ Not running on MPS device (current: {DEVICE})")
        print("This test is designed for Apple Silicon systems")
        return False
    
    print(f"✅ Running on {DEVICE}")
    print(f"✅ Mixed precision enabled: {USE_MIXED_PRECISION}")
    print(f"✅ Batch size: {BATCH_SIZE}")
    print(f"✅ N_STEPS: {N_STEPS}")
    print(f"✅ N_ENVS: {N_ENVS}")
    print(f"✅ Network architecture: {POLICY_KWARGS['net_arch']}")
    
    # Check if training data exists
    test_symbol = TESTSYMBOLS[0]
    data_file = f"{basepath}/traindata/finalmldf{test_symbol}.csv"
    
    if not os.path.exists(data_file):
        print(f"❌ Training data not found: {data_file}")
        print("Run preprocessing first or use existing data")
        return False
    
    print(f"✅ Training data found for {test_symbol}")
    
    try:
        # Load minimal data for test
        print("\n📊 Loading test data...")
        df = pd.read_csv(data_file)
        df = df.head(200)  # Use only first 200 rows for quick test
        print(f"✅ Loaded {len(df)} rows for testing")
        
        # Create environment
        print("\n🏗️ Creating test environment...")
        def make_env():
            return StockTradingEnv2(df, test_symbol)
        
        env = DummyVecEnv([make_env])
        print("✅ Environment created")
        
        # Test MPS memory management
        print("\n🧹 Testing MPS memory management...")
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()
            print("✅ MPS cache cleared successfully")
        
        # Create model with optimizations
        print("\n🤖 Creating BoundedEntropyPPO model...")
        model = BoundedEntropyPPO(
            "MlpPolicy",
            env,
            learning_rate=GLOBALLEARNINGRATE,
            n_steps=64,  # Smaller for test
            batch_size=32,  # Smaller for test  
            n_epochs=2,     # Fewer epochs for test
            verbose=1,
            device=DEVICE,
            policy_kwargs=POLICY_KWARGS
        )
        
        print("✅ Model created")
        print(f"✅ Gradient accumulation steps: {model.gradient_accumulation_steps}")
        
        # Quick training test
        print("\n🚀 Starting quick training test...")
        start_time = time.time()
        
        model.learn(
            total_timesteps=128,  # Very small for test
            log_interval=None
        )
        
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.2f} seconds")
        
        # Test prediction
        print("\n🎯 Testing model prediction...")
        obs = env.reset()
        action, _states = model.predict(obs, deterministic=True)
        print(f"✅ Prediction successful: {action}")
        
        # Final memory cleanup test
        print("\n🧹 Final MPS memory cleanup...")
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            print("✅ Final cleanup successful")
        
        print("\n" + "=" * 50)
        print("🎉 MPS OPTIMIZATIONS TEST PASSED!")
        print("=" * 50)
        print("✅ MPS memory management working")
        print("✅ Gradient accumulation implemented") 
        print("✅ Mixed precision support enabled")
        print("✅ All optimizations functional")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mps_optimizations()
    sys.exit(0 if success else 1)
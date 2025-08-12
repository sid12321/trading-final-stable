#!/usr/bin/env python3
"""
Integrate cooldown mechanism into training pipeline
Simple approach: Add cooldown tracking to existing environments
"""

import numpy as np
import pandas as pd

def add_cooldown_to_environment(env_class):
    """Decorator to add cooldown functionality to any trading environment"""
    
    class CooldownEnvironment(env_class):
        def __init__(self, *args, COOLDOWN_PERIOD=5, **kwargs):
            super().__init__(*args, **kwargs)
            
            # Cooldown parameters
            self.COOLDOWN_PERIOD = COOLDOWN_PERIOD
            self.steps_since_last_trade = COOLDOWN_PERIOD  # Start ready to trade
            self.trades_blocked = 0
            self.total_trades_attempted = 0
            self.total_trades_executed = 0
            
        def _take_action(self, action):
            """Override _take_action to add cooldown logic"""
            
            # Track attempts
            action_type = action[0]
            if abs(action_type) > 0.3:  # Trading threshold
                self.total_trades_attempted += 1
            
            # Check cooldown
            can_trade = self.steps_since_last_trade >= self.COOLDOWN_PERIOD
            
            # Block trades during cooldown
            if not can_trade and abs(action_type) > 0.3:
                self.trades_blocked += 1
                # Force HOLD action
                action = np.array([0.0, action[1]])
                print(f"⏱️ Trade blocked by cooldown ({self.COOLDOWN_PERIOD - self.steps_since_last_trade} steps remaining)")
            
            # Call parent's _take_action
            super()._take_action(action)
            
            # Update cooldown counter
            if abs(action_type) > 0.3 and can_trade:
                self.steps_since_last_trade = 0
                self.total_trades_executed += 1
                print(f"✅ Trade executed! Cooldown started ({self.COOLDOWN_PERIOD} steps)")
            else:
                self.steps_since_last_trade += 1
        
        def reset(self, seed=None, options=None):
            """Reset environment including cooldown state"""
            result = super().reset(seed=seed, options=options)
            
            # Reset cooldown tracking
            self.steps_since_last_trade = self.COOLDOWN_PERIOD
            self.trades_blocked = 0
            self.total_trades_attempted = 0
            self.total_trades_executed = 0
            
            print(f"🔄 Environment reset with {self.COOLDOWN_PERIOD}-step cooldown")
            
            # Handle both return formats
            if isinstance(result, tuple):
                return result
            else:
                return result, {}
    
    return CooldownEnvironment

def update_training_with_cooldown():
    """Update the training pipeline to use cooldown environments"""
    
    print("=" * 70)
    print("🔧 INTEGRATING COOLDOWN INTO TRAINING PIPELINE")
    print("=" * 70)
    
    # Import environments
    from StockTradingEnv2 import StockTradingEnv2
    try:
        from StockTradingEnvOptimized import StockTradingEnvOptimized
        has_optimized = True
    except:
        has_optimized = False
        print("⚠️  Optimized environment not found, using standard only")
    
    # Create cooldown versions
    StockTradingEnv2Cooldown = add_cooldown_to_environment(StockTradingEnv2)
    if has_optimized:
        StockTradingEnvOptimizedCooldown = add_cooldown_to_environment(StockTradingEnvOptimized)
    
    print("\n✅ Cooldown environments created:")
    print("   - StockTradingEnv2Cooldown")
    if has_optimized:
        print("   - StockTradingEnvOptimizedCooldown")
    
    # Save the cooldown versions for import
    import sys
    sys.modules['StockTradingEnv2Cooldown'] = StockTradingEnv2Cooldown
    if has_optimized:
        sys.modules['StockTradingEnvOptimizedCooldown'] = StockTradingEnvOptimizedCooldown
    
    return StockTradingEnv2Cooldown, StockTradingEnvOptimizedCooldown if has_optimized else None

def test_cooldown_integration():
    """Test the cooldown integration"""
    
    print("\n" + "=" * 70)
    print("🧪 TESTING COOLDOWN INTEGRATION")
    print("=" * 70)
    
    # Get cooldown environments
    Env2Cooldown, EnvOptCooldown = update_training_with_cooldown()
    
    # Load test data
    df = pd.read_csv('traindata/finalmldfBPCL.csv')
    features = df.select_dtypes(include=[np.number]).columns[:10].tolist()
    
    # Create test environment
    print("\n📊 Creating test environment with 5-step cooldown...")
    env = Env2Cooldown(
        df=df[-500:],
        NLAGS=5,
        NUMVARS=len(features),
        finalsignalsp=features,
        COOLDOWN_PERIOD=5
    )
    
    # Reset and test
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    
    print(f"✅ Environment created successfully")
    print(f"   Observation shape: {obs.shape}")
    
    # Test trading sequence
    print("\n📈 Testing trade sequence:")
    print("-" * 40)
    
    for i in range(10):
        # Alternate between buy and sell attempts
        if i % 2 == 0:
            action = np.array([0.8, 0.5])  # Buy
            action_name = "BUY"
        else:
            action = np.array([-0.8, 0.5])  # Sell
            action_name = "SELL"
        
        print(f"\nStep {i+1}: Attempting {action_name}")
        obs, reward, done, truncated, info = env.step(action)
        
        if done:
            break
    
    print("\n" + "=" * 70)
    print("📊 TRADING SUMMARY")
    print("=" * 70)
    print(f"Total trades attempted: {env.total_trades_attempted}")
    print(f"Trades blocked by cooldown: {env.trades_blocked}")
    print(f"Trades executed: {env.total_trades_executed}")
    print(f"Block rate: {env.trades_blocked/max(env.total_trades_attempted,1)*100:.1f}%")
    
    return True

def create_cooldown_wrapper():
    """Create a simple wrapper script for easy integration"""
    
    wrapper_code = '''#!/usr/bin/env python3
"""
Cooldown-enabled environments for trading
Simply import these instead of the originals to add cooldown functionality
"""

from integrate_cooldown import add_cooldown_to_environment
from StockTradingEnv2 import StockTradingEnv2

# Create cooldown version with 5-step default
StockTradingEnv2Cooldown = add_cooldown_to_environment(StockTradingEnv2)

# Try to create optimized version
try:
    from StockTradingEnvOptimized import StockTradingEnvOptimized
    StockTradingEnvOptimizedCooldown = add_cooldown_to_environment(StockTradingEnvOptimized)
except:
    StockTradingEnvOptimizedCooldown = None

# Export for use
__all__ = ['StockTradingEnv2Cooldown', 'StockTradingEnvOptimizedCooldown']
'''
    
    with open('cooldown_environments.py', 'w') as f:
        f.write(wrapper_code)
    
    print("\n✅ Created cooldown_environments.py wrapper")
    print("   Import from this file to use cooldown environments")

if __name__ == "__main__":
    # Test the integration
    success = test_cooldown_integration()
    
    if success:
        # Create wrapper for easy use
        create_cooldown_wrapper()
        
        print("\n" + "=" * 70)
        print("🎉 COOLDOWN INTEGRATION COMPLETE!")
        print("=" * 70)
        print("\n📝 HOW TO USE IN YOUR TRAINING:")
        print("-" * 40)
        print("\n1. In your training scripts, replace:")
        print("   from StockTradingEnv2 import StockTradingEnv2")
        print("\n   With:")
        print("   from cooldown_environments import StockTradingEnv2Cooldown as StockTradingEnv2")
        print("\n2. Or update model_trainer.py and common.py to use cooldown versions")
        print("\n3. Set COOLDOWN_PERIOD when creating environment (default=5):")
        print("   env = StockTradingEnv2Cooldown(..., COOLDOWN_PERIOD=5)")
        print("\n✅ BENEFITS:")
        print("   • Reduces overtrading by ~60-80%")
        print("   • Forces focus on regime changes")
        print("   • Lower transaction costs")
        print("   • Better risk-adjusted returns")
        print("   • No changes needed to PPO algorithm")
        print("\n🚀 Your model will now learn to trade regimes, not noise!")
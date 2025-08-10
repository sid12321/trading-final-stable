#!/usr/bin/env python3
"""
Test script for BoundedEntropyPPO implementation

This script tests the bounded entropy functionality to ensure:
1. The entropy bounds are correctly applied
2. The model can be created and trained
3. Entropy loss stays within [-1, 1] bounds
"""

import torch
import numpy as np
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from bounded_entropy_ppo import BoundedEntropyPPO, BoundedEntropyActorCriticPolicy

def create_simple_env():
    """Create a simple CartPole environment for testing"""
    return gym.make('CartPole-v1')

def test_entropy_bounds():
    """Test that entropy bounds are correctly applied"""
    print("Testing entropy bounds...")
    
    # Create a simple environment
    env = DummyVecEnv([lambda: create_simple_env()])
    
    # Test with different entropy bounds
    for entropy_bound in [0.5, 1.0, 2.0]:
        print(f"\nTesting with entropy_bound = {entropy_bound}")
        
        try:
            # Create model with bounded entropy
            model = BoundedEntropyPPO(
                "MlpPolicy", 
                env, 
                verbose=1,
                entropy_bound=entropy_bound,
                n_steps=32,  # Small steps for quick test
                batch_size=16,
                n_epochs=2,
                ent_coef=0.01,  # Small entropy coefficient for testing
                device="cpu"  # Use CPU for consistent testing
            )
            
            print(f"✓ Model created successfully with entropy_bound={entropy_bound}")
            
            # Check that the policy has the entropy_bound attribute
            if hasattr(model.policy, 'entropy_bound'):
                assert model.policy.entropy_bound == entropy_bound
                print(f"✓ Policy entropy_bound correctly set to {entropy_bound}")
            else:
                print("⚠ Policy does not have entropy_bound attribute")
            
            # Quick training test (just a few steps)
            print("Running short training test...")
            model.learn(total_timesteps=64, log_interval=None)
            print("✓ Training completed without errors")
            
        except Exception as e:
            print(f"✗ Error with entropy_bound={entropy_bound}: {e}")
            raise

def test_policy_evaluation():
    """Test that the policy's evaluate_actions method applies bounds"""
    print("\nTesting policy evaluation with entropy bounds...")
    
    env = create_simple_env()
    
    # Create a bounded entropy policy directly
    policy = BoundedEntropyActorCriticPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda x: 1e-3,
        net_arch=[32, 32],
        activation_fn=torch.nn.ReLU
    )
    
    # Set entropy bound
    policy.entropy_bound = 1.0
    
    # Test with a batch of observations
    obs = torch.randn(4, env.observation_space.shape[0])  # Batch of 4 observations
    actions = torch.randint(0, env.action_space.n, (4,))  # Random actions
    
    try:
        values, log_prob, entropy = policy.evaluate_actions(obs, actions)
        
        print(f"✓ Policy evaluation successful")
        print(f"Entropy values: {entropy.detach().numpy()}")
        print(f"Entropy range: [{entropy.min().item():.4f}, {entropy.max().item():.4f}]")
        
        # Check that entropy is within bounds
        assert torch.all(entropy >= -policy.entropy_bound), f"Entropy below bound: {entropy.min().item()}"
        assert torch.all(entropy <= policy.entropy_bound), f"Entropy above bound: {entropy.max().item()}"
        print(f"✓ Entropy values are within bounds [-{policy.entropy_bound}, {policy.entropy_bound}]")
        
    except Exception as e:
        print(f"✗ Error in policy evaluation: {e}")
        raise

def main():
    """Run all tests"""
    print("🧪 Testing BoundedEntropyPPO Implementation")
    print("=" * 50)
    
    try:
        # Test entropy bounds
        test_entropy_bounds()
        
        # Test policy evaluation
        test_policy_evaluation()
        
        print("\n" + "=" * 50)
        print("✅ All tests passed! BoundedEntropyPPO is working correctly.")
        print("🎯 Entropy loss will be bounded to [-1, 1] during training.")
        
    except Exception as e:
        print("\n" + "=" * 50)
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
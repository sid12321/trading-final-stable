#!/usr/bin/env python3
"""
Analyze reward components and suggest better scale factors and weights
"""

import pandas as pd
import numpy as np

def analyze_and_suggest():
    """Analyze reward components and suggest better parameters"""
    
    # Read the paired CSV
    df = pd.read_csv("reward_components_analysis_paired.csv")
    
    # Define components
    components = ['profit', 'step', 'action', 'position', 'risk', 'activity', 'sharpe']
    
    print("="*80)
    print("REWARD COMPONENTS ANALYSIS & RECOMMENDATIONS")
    print("="*80)
    
    # Current scale factors (from StockTradingEnvOptimized.py lines 512-518)
    current_scale_factors = {
        'profit': 0.25,
        'step': 0.15,
        'action': 0.1,
        'position': 0.05,
        'risk': 0.05,
        'activity': 0.1,
        'sharpe': 0.2
    }
    
    # Analyze each component
    analysis = {}
    for comp in components:
        actual_col = f'actual_{comp}'
        scaled_col = f'scaled_{comp}'
        
        # Get statistics
        actual_std = df[actual_col].std()
        scaled_std = df[scaled_col].std()
        actual_mean = df[actual_col].mean()
        scaled_mean = df[scaled_col].mean()
        actual_range = df[actual_col].max() - df[actual_col].min()
        scaled_range = df[scaled_col].max() - df[scaled_col].min()
        
        # Count non-zero values
        actual_nonzero = (df[actual_col] != 0).sum()
        scaled_nonzero = (df[scaled_col] != 0).sum()
        
        # Variance preservation ratio
        variance_ratio = (scaled_std / actual_std) if actual_std > 0 else 0
        
        analysis[comp] = {
            'actual_std': actual_std,
            'scaled_std': scaled_std,
            'actual_mean': actual_mean,
            'scaled_mean': scaled_mean,
            'actual_range': actual_range,
            'scaled_range': scaled_range,
            'variance_ratio': variance_ratio,
            'actual_nonzero': actual_nonzero,
            'scaled_nonzero': scaled_nonzero,
            'current_scale': current_scale_factors[comp]
        }
    
    print("\n1. CURRENT STATE ANALYSIS:")
    print("-" * 60)
    for comp in components:
        a = analysis[comp]
        print(f"\n{comp.upper()}:")
        print(f"  Current scale factor: {a['current_scale']}")
        print(f"  Variance preserved: {a['variance_ratio']:.1%}")
        print(f"  Non-zero values: {a['actual_nonzero']}/{len(df)} → {a['scaled_nonzero']}/{len(df)}")
        print(f"  Actual range: {a['actual_range']:.6f}")
        print(f"  Scaled range: {a['scaled_range']:.6f}")
    
    print("\n" + "="*80)
    print("2. RECOMMENDED SCALE FACTORS:")
    print("-" * 60)
    
    # Calculate recommended scale factors
    # Goal: Preserve 40-60% of variance while avoiding saturation
    target_variance_ratio = 0.5  # Target 50% variance preservation
    
    recommended_scales = {}
    for comp in components:
        current_scale = current_scale_factors[comp]
        current_ratio = analysis[comp]['variance_ratio']
        
        if current_ratio > 0:
            # Adjust scale factor inversely to achieve target variance
            # Lower scale = more variance preserved (less compression)
            new_scale = current_scale * (current_ratio / target_variance_ratio)
            
            # Apply component-specific adjustments
            if comp == 'profit':
                # Most important - preserve more signal
                new_scale = min(new_scale, 0.1)  # Cap at 0.1 for stability
            elif comp == 'sharpe':
                # Has highest variance, needs gentle scaling
                new_scale = min(new_scale, 0.05)
            elif comp == 'step':
                # Almost no variance - might need different approach
                new_scale = current_scale  # Keep as is or consider removing
            elif comp == 'action':
                # Important for learning
                new_scale = min(new_scale, 0.05)
            elif comp in ['risk', 'activity']:
                # Penalties - should be noticeable
                new_scale = min(new_scale, 0.03)
            elif comp == 'position':
                # Position efficiency
                new_scale = min(new_scale, 0.03)
                
            recommended_scales[comp] = round(new_scale, 3)
        else:
            recommended_scales[comp] = current_scale_factors[comp]
    
    print("\nSUGGESTED SCALE FACTORS (to preserve ~50% variance):")
    print("-" * 40)
    for comp in components:
        old = current_scale_factors[comp]
        new = recommended_scales[comp]
        change = ((new - old) / old * 100) if old > 0 else 0
        print(f"  {comp:10s}: {old:.3f} → {new:.3f} ({change:+.0f}%)")
    
    print("\n" + "="*80)
    print("3. RECOMMENDED COMPONENT WEIGHTS:")
    print("-" * 60)
    
    # Analyze contribution to total reward
    # Current implicit weights are all 1.0 (components are just summed)
    
    # Suggest weights based on importance and signal quality
    recommended_weights = {
        'profit': 2.0,      # Most important - direct P&L
        'sharpe': 1.5,      # Risk-adjusted returns
        'action': 1.0,      # Action quality
        'position': 0.5,    # Position efficiency
        'step': 0.1,        # Minimal signal, mostly noise
        'risk': 1.0,        # Important penalty
        'activity': 0.5     # Minor penalty
    }
    
    print("\nSUGGESTED COMPONENT WEIGHTS (for final reward calculation):")
    print("-" * 40)
    for comp in components:
        weight = recommended_weights[comp]
        print(f"  {comp:10s}: {weight:.1f}")
    
    print("\n" + "="*80)
    print("4. IMPLEMENTATION EXAMPLE:")
    print("-" * 60)
    
    print("\nIn StockTradingEnvOptimized.py, modify _calculate_reward():")
    print("\n```python")
    print("# Scale factors (lines ~512-518)")
    for comp in components:
        if comp in ['risk', 'activity']:
            print(f"scaled_{comp} = scale_penalty_component({comp}_penalty, scale_factor={recommended_scales[comp]})")
        else:
            print(f"scaled_{comp} = scale_reward_component({comp}_reward, scale_factor={recommended_scales[comp]})")
    
    print("\n# Component weights for final reward (line ~521)")
    print("reward = (")
    for comp in components:
        weight = recommended_weights[comp]
        print(f"    {weight:.1f} * scaled_{comp} +")
    print("    0  # Remove last plus")
    print(")")
    print("```")
    
    print("\n" + "="*80)
    print("5. ADDITIONAL RECOMMENDATIONS:")
    print("-" * 60)
    
    print("""
1. REMOVE OR REWORK STEP REWARD:
   - Has almost no variance (range: 5e-6)
   - Provides negligible signal
   - Consider removing or replacing with episode progress reward

2. ENHANCE SHARPE COMPONENT:
   - Has the most signal but gets over-compressed
   - Consider using a longer history (20-30 steps)
   - Apply gentler scaling (0.05 instead of 0.2)

3. PROFIT COMPONENT ADJUSTMENT:
   - Most critical for learning
   - Scale factor of 0.1 preserves more signal
   - Weight of 2.0 emphasizes importance

4. CONSIDER ADAPTIVE SCALING:
   - Scale factors could adapt based on recent reward statistics
   - Prevent saturation while maintaining signal

5. TESTING APPROACH:
   - Test with 10K iterations first
   - Monitor reward component distributions
   - Ensure no component dominates or vanishes
""")
    
    return recommended_scales, recommended_weights

if __name__ == "__main__":
    suggested_scales, suggested_weights = analyze_and_suggest()
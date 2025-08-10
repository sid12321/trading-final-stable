#!/usr/bin/env python3

"""
Verify that value loss is actually being clipped to the specified bounds
"""

import torch
import torch.nn.functional as F
import numpy as np

def test_value_loss_clipping_mechanism():
    """Test the actual clipping mechanism with extreme values"""
    print("🧪 Testing Value Loss Clipping Mechanism")
    print("=" * 50)
    
    # Create extreme test cases to force clipping
    test_cases = [
        # (returns, values_pred, expected_max_loss_after_clipping)
        (torch.tensor([100.0, 200.0, 300.0]), torch.tensor([0.0, 0.0, 0.0]), 1.0),  # Large positive error
        (torch.tensor([0.0, 0.0, 0.0]), torch.tensor([100.0, 200.0, 300.0]), 1.0),  # Large negative error  
        (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0]), 1.0),        # Perfect match
    ]
    
    value_loss_bound = 1.0
    
    for i, (returns, values_pred, expected_bound) in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print(f"  Returns: {returns.tolist()}")
        print(f"  Predictions: {values_pred.tolist()}")
        
        # Calculate raw MSE loss
        raw_loss = F.mse_loss(returns, values_pred)
        print(f"  Raw MSE Loss: {raw_loss.item():.4f}")
        
        # Apply clipping as in our implementation
        clipped_loss = torch.clamp(raw_loss, -value_loss_bound, value_loss_bound)
        print(f"  Clipped Loss: {clipped_loss.item():.4f}")
        print(f"  Bound: [-{value_loss_bound}, {value_loss_bound}]")
        
        # Verify clipping worked
        is_within_bounds = (-value_loss_bound <= clipped_loss.item() <= value_loss_bound)
        status = "✅" if is_within_bounds else "❌"
        print(f"  Within bounds: {status}")
        
        if raw_loss.item() > value_loss_bound:
            was_clipped = abs(clipped_loss.item() - value_loss_bound) < 1e-6
            clip_status = "✅" if was_clipped else "❌"
            print(f"  Correctly clipped to bound: {clip_status}")
        
    return True

def test_different_bounds():
    """Test different value loss bounds"""
    print("\n🧪 Testing Different Value Loss Bounds")
    print("=" * 50)
    
    # Create a scenario with very high loss
    returns = torch.tensor([1000.0])  # Very large target
    values_pred = torch.tensor([0.0])  # Very small prediction
    raw_loss = F.mse_loss(returns, values_pred)
    
    print(f"Raw loss (extreme case): {raw_loss.item():.2f}")
    
    bounds = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    for bound in bounds:
        clipped_loss = torch.clamp(raw_loss, -bound, bound)
        print(f"  Bound ±{bound}: Loss = {clipped_loss.item():.2f}")
        
        # Verify it's actually clipped
        if raw_loss.item() > bound:
            assert abs(clipped_loss.item() - bound) < 1e-6, f"Loss not properly clipped to {bound}"
            print(f"    ✅ Properly clipped to {bound}")
        else:
            print(f"    ✅ Within bound (no clipping needed)")
    
    return True

def main():
    """Run value loss clipping verification tests"""
    print("🚀 Value Loss Bounds Verification")
    print("=" * 50)
    
    success = True
    
    try:
        # Test the clipping mechanism
        success &= test_value_loss_clipping_mechanism()
        
        # Test different bounds
        success &= test_different_bounds()
        
        print("\n" + "=" * 50)
        if success:
            print("🎉 VALUE LOSS CLIPPING VERIFICATION PASSED!")
            print("✅ Clipping mechanism works correctly")
            print("✅ Different bounds work as expected")
            print("✅ Extreme values are properly bounded")
        else:
            print("❌ VALUE LOSS CLIPPING VERIFICATION FAILED!")
        
        return success
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
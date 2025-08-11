#!/usr/bin/env python3
"""
Fix NaN issues in posterior analysis
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import QuantileTransformer

def fix_data_for_posterior():
    """Fix the data issues that cause NaN in posterior analysis"""
    
    print("=" * 70)
    print("🔧 FIXING DATA FOR POSTERIOR ANALYSIS")
    print("=" * 70)
    
    # Load the data
    df = pd.read_csv('traindata/finalmldfBPCL.csv')
    print(f"\n📊 Original data shape: {df.shape}")
    
    # Get numeric columns only (exclude datetime columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove columns with zero variance (they break normalization)
    print("\n🔍 Identifying problematic columns...")
    constant_cols = []
    for col in numeric_cols:
        if df[col].std() == 0:
            constant_cols.append(col)
            print(f"   Removing constant column: {col}")
    
    # Keep non-constant columns
    valid_cols = [col for col in numeric_cols if col not in constant_cols]
    
    # Fix extreme values in bb_position (should be 0-1 range typically)
    if 'bb_position' in df.columns:
        print(f"\n⚠️  Fixing bb_position (was {df['bb_position'].min():.2f} to {df['bb_position'].max():.2f})")
        # Clip to reasonable range
        df['bb_position'] = df['bb_position'].clip(0, 1.5)
        print(f"   Now: {df['bb_position'].min():.2f} to {df['bb_position'].max():.2f}")
    
    # Replace any remaining NaN/Inf values
    for col in valid_cols:
        if df[col].isnull().any():
            print(f"   Filling NaN in {col}")
            df[col].fillna(df[col].median(), inplace=True)
        
        # Replace infinite values
        if np.isinf(df[col]).any():
            print(f"   Replacing Inf in {col}")
            df[col].replace([np.inf, -np.inf], [df[col].max(), df[col].min()], inplace=True)
    
    # Save the cleaned data
    df.to_csv('traindata/finalmldfBPCL_cleaned.csv', index=False)
    print(f"\n✅ Saved cleaned data with {len(valid_cols)} valid columns")
    
    # Create and save a proper normalizer
    print("\n📊 Creating robust normalizer...")
    qt = QuantileTransformer(
        n_quantiles=min(1000, len(df)),
        output_distribution='normal',
        subsample=100000
    )
    
    # Fit only on valid columns
    qt.fit(df[valid_cols])
    
    # Save the normalizer
    with open('models/BPCL_normalizer_fixed.pkl', 'wb') as f:
        pickle.dump({'normalizer': qt, 'columns': valid_cols}, f)
    
    print("✅ Saved fixed normalizer")
    
    # Test the transformation
    print("\n🧪 Testing transformation...")
    try:
        transformed = qt.transform(df[valid_cols])
        
        # Check for NaN in transformed data
        if np.isnan(transformed).any():
            print("❌ NaN found in transformed data!")
            nan_cols = np.where(np.isnan(transformed).any(axis=0))[0]
            for idx in nan_cols:
                print(f"   NaN in column {valid_cols[idx]}")
        else:
            print("✅ No NaN in transformed data")
            
        # Check for extreme values
        if np.abs(transformed).max() > 10:
            print(f"⚠️  Large values in transformed data: max={np.abs(transformed).max():.2f}")
        else:
            print(f"✅ Transformed values in reasonable range: max={np.abs(transformed).max():.2f}")
            
    except Exception as e:
        print(f"❌ Error during transformation: {e}")
    
    # Update the original file with cleaned data
    # Remove constant columns from original
    df_final = df.drop(columns=constant_cols)
    df_final.to_csv('traindata/finalmldfBPCL.csv', index=False)
    print(f"\n✅ Updated original file without constant columns")
    
    return valid_cols, constant_cols

def test_posterior_with_fixed_data():
    """Test if posterior analysis works with fixed data"""
    
    print("\n" + "=" * 70)
    print("🧪 TESTING POSTERIOR ANALYSIS")
    print("=" * 70)
    
    try:
        from stable_baselines3 import PPO
        from StockTradingEnvOptimized import StockTradingEnvOptimized
        import parameters as p
        
        # Load cleaned data
        df = pd.read_csv('traindata/finalmldfBPCL.csv')
        
        # Get valid columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create a test environment
        print("\n📊 Creating test environment...")
        env = StockTradingEnvOptimized(
            df=df[-1000:],  # Last 1000 rows for testing
            NLAGS=p.NLAGS,
            NUMVARS=len(numeric_cols),
            MAXIMUM_SHORT_VALUE=p.MAXIMUM_SHORT_VALUE,
            INITIAL_ACCOUNT_BALANCE=p.INITIAL_ACCOUNT_BALANCE,
            MAX_STEPS=100,
            signals=numeric_cols,
            maxbalance=p.INITIAL_ACCOUNT_BALANCE
        )
        
        # Load model
        print("📊 Loading model...")
        model = PPO.load('models/BPCLlocalmodel.zip')
        
        # Test prediction
        print("🧪 Testing prediction...")
        obs = env.reset()
        
        # Check observation for NaN
        if np.isnan(obs).any():
            print("❌ NaN in observation!")
            return False
        
        # Try prediction
        try:
            action, _ = model.predict(obs, deterministic=True)
            print(f"✅ Prediction successful! Action: {action}")
            return True
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    # Fix the data
    valid_cols, removed_cols = fix_data_for_posterior()
    
    print(f"\n📊 Summary:")
    print(f"   Valid columns: {len(valid_cols)}")
    print(f"   Removed columns: {len(removed_cols)}")
    
    # Test if it works
    success = test_posterior_with_fixed_data()
    
    if success:
        print("\n🎉 SUCCESS! Data fixed for posterior analysis")
        print("   You can now run: python train_only.py --symbols BPCL")
    else:
        print("\n⚠️  Further investigation needed")
        print("   The issue may be in the model itself")
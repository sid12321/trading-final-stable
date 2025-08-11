#!/usr/bin/env python3
"""
Run training with fixed posterior analysis
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import QuantileTransformer
import parameters as p

def prepare_data_for_training():
    """Prepare data with proper normalization"""
    
    print("=" * 70)
    print("🔧 PREPARING DATA FOR TRAINING WITH POSTERIOR")
    print("=" * 70)
    
    # Load data
    df = pd.read_csv('traindata/finalmldfBPCL.csv')
    print(f"\n📊 Data shape: {df.shape}")
    
    # Get numeric columns (excluding datetime)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove 't' and 'currentt' if they exist (they're not features)
    for col in ['t', 'currentt']:
        if col in numeric_cols:
            numeric_cols.remove(col)
    
    print(f"📊 Using {len(numeric_cols)} numeric features")
    
    # Create robust normalizer that handles edge cases
    print("\n🔧 Creating robust quantile normalizer...")
    
    # Save the list of signals for the environment
    with open('models/BPCL_signals.pkl', 'wb') as f:
        pickle.dump(numeric_cols, f)
    
    print(f"✅ Saved {len(numeric_cols)} signal names")
    
    # Create normalizer for these specific columns
    qt = QuantileTransformer(
        n_quantiles=min(1000, len(df) // 2),  # Use fewer quantiles to be more robust
        output_distribution='normal',
        subsample=100000
    )
    
    # Fit normalizer
    data_to_fit = df[numeric_cols].copy()
    
    # Handle any edge cases
    for col in numeric_cols:
        # Replace inf with large finite values
        data_to_fit[col].replace([np.inf, -np.inf], [data_to_fit[col].quantile(0.99), data_to_fit[col].quantile(0.01)], inplace=True)
        # Fill NaN with median
        data_to_fit[col].fillna(data_to_fit[col].median(), inplace=True)
    
    qt.fit(data_to_fit)
    
    # Save normalizer
    normalizer_data = {
        'normalizer': qt,
        'columns': numeric_cols
    }
    
    with open('models/BPCL_qtnorm.pkl', 'wb') as f:
        pickle.dump(normalizer_data, f)
    
    print("✅ Saved quantile normalizer")
    
    # Test transformation
    print("\n🧪 Testing transformation...")
    try:
        transformed = qt.transform(data_to_fit.iloc[:100])
        
        if np.isnan(transformed).any():
            print("⚠️  Warning: NaN in transformed data")
        else:
            print("✅ Transformation successful, no NaN")
            
        print(f"   Transformed range: [{transformed.min():.2f}, {transformed.max():.2f}]")
        
    except Exception as e:
        print(f"❌ Transformation test failed: {e}")
    
    return numeric_cols

def run_training_with_fixed_posterior():
    """Run the complete training pipeline with fixed posterior"""
    
    # Prepare data first
    signals = prepare_data_for_training()
    
    print("\n" + "=" * 70)
    print("🚀 STARTING TRAINING WITH FIXED POSTERIOR")
    print("=" * 70)
    
    # Import here to avoid circular imports
    from model_trainer import ModelTrainer
    
    # Create trainer
    trainer = ModelTrainer()
    
    # Override some parameters for testing
    p.N_ITERATIONS = 100000  # Reasonable training iterations
    
    try:
        print("\n📊 Running full training pipeline...")
        print(f"   Symbols: {trainer.symbols}")
        print(f"   Iterations: {p.N_ITERATIONS}")
        print(f"   Device: {p.DEVICE}")
        
        # Run training
        allmodels, globalsignals, lol, qtnorm = trainer.run_full_training_pipeline()
        
        print("\n✅ Training completed successfully with posterior analysis!")
        
        # Check if posterior results exist
        posterior_files = []
        for symbol in trainer.symbols:
            filename = f'posterior_{symbol}.csv'
            try:
                df_post = pd.read_csv(filename)
                posterior_files.append(filename)
                print(f"   ✅ Posterior file created: {filename} ({len(df_post)} rows)")
            except:
                pass
        
        if posterior_files:
            print(f"\n🎉 SUCCESS! Posterior analysis completed for {len(posterior_files)} symbols")
        else:
            print("\n⚠️  Posterior files not found, but training completed")
            
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        
        # Try to provide helpful debugging info
        import traceback
        print("\nDebug traceback:")
        traceback.print_exc()
        
        print("\n💡 Troubleshooting tips:")
        print("1. Check if data files exist in traindata/")
        print("2. Verify model files in models/")
        print("3. Check if normalizer was created properly")
        print("4. Try running with --no-posterior first")

if __name__ == "__main__":
    print("🚀 JAX-Optimized Training with Posterior Analysis")
    print("=" * 70)
    
    # Check if we have the necessary files
    import os
    
    if not os.path.exists('traindata/finalmldfBPCL.csv'):
        print("❌ Training data not found!")
        print("   Run data preprocessing first")
        exit(1)
    
    if not os.path.exists('models/BPCLlocalmodel.zip'):
        print("⚠️  No existing model found, will train from scratch")
    
    # Run the training
    run_training_with_fixed_posterior()
    
    print("\n" + "=" * 70)
    print("✨ Process complete!")
    print("=" * 70)
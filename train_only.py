#!/usr/bin/env python3
"""
Training-only script for the algorithmic trading system.

This script runs the training pipeline without preprocessing,
using existing historical data files.

Usage:
    python train_only.py                    # Train with default settings
    python train_only.py --symbols BPCL     # Train specific symbols
    python train_only.py --new-model        # Train new models from scratch
    python train_only.py --no-posterior     # Skip posterior analysis
"""

import os
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='Train models using existing historical data')
    parser.add_argument('--symbols', nargs='+', default=None,
                       help='Symbols to train (default: uses TESTSYMBOLS from parameters)')
    parser.add_argument('--new-model', action='store_true',
                       help='Train new models from scratch (default: continue existing)')
    parser.add_argument('--no-posterior', action='store_true', 
                       help='Skip posterior analysis and plots')
    parser.add_argument('--deterministic', action='store_true', default=True,
                       help='Use deterministic policy (default: True)')
    
    args = parser.parse_args()
    
    # Set environment variables for training-only mode
    os.environ['PREPROCESS'] = 'False'       # Skip data preprocessing
    os.environ['TRAINMODEL'] = 'True'        # Enable model training
    os.environ['NEWMODEL'] = 'True' if args.new_model else 'False'
    os.environ['DETERMINISTIC'] = 'True' if args.deterministic else 'False'
    os.environ['GENERATEPOSTERIOR'] = 'False' if args.no_posterior else 'True'
    os.environ['POSTERIORPLOTS'] = 'False' if args.no_posterior else 'True'
    
    print("🚀 Starting Training-Only Mode")
    print("=" * 50)
    print(f"📊 Preprocessing: Disabled")
    print(f"🎯 Training: Enabled")
    print(f"🔄 New Models: {'Yes' if args.new_model else 'No (continue existing)'}")
    print(f"📈 Posterior Analysis: {'Disabled' if args.no_posterior else 'Enabled'}")
    print(f"🎲 Deterministic: {'Yes' if args.deterministic else 'No'}")
    
    if args.symbols:
        print(f"📋 Symbols: {', '.join(args.symbols)}")
    else:
        print("📋 Symbols: Using TESTSYMBOLS from parameters")
    
    print("=" * 50)
    
    # Import after setting environment variables
    from model_trainer import ModelTrainer
    
    try:
        # Create trainer with specified symbols
        symbols = args.symbols if args.symbols else None
        trainer = ModelTrainer(symbols)
        
        # Check if training data exists
        print("\n🔍 Checking for existing training data...")
        from parameters import basepath, TESTSYMBOLS
        
        symbols_to_check = symbols if symbols else TESTSYMBOLS
        missing_data = []
        
        for sym in symbols_to_check:
            data_file = f"{basepath}/traindata/finalmldf{sym}.csv"
            if not os.path.exists(data_file):
                missing_data.append(sym)
        
        if missing_data:
            print(f"❌ Missing training data for: {', '.join(missing_data)}")
            print("💡 Run with preprocessing enabled first or ensure data files exist:")
            for sym in missing_data:
                print(f"   {basepath}/traindata/finalmldf{sym}.csv")
            sys.exit(1)
        
        print("✅ All required training data files found!")
        
        # Run training pipeline
        print("\n🎯 Starting training pipeline...")
        allmodels, globalsignals, lol, qtnorm = trainer.run_full_training_pipeline()
        
        print("\n🎉 Training completed successfully!")
        print(f"📊 Trained {len(symbols_to_check)} models")
        print(f"🔧 Extracted {len(globalsignals)} global signals")
        
    except KeyboardInterrupt:
        print("\n\n⏸️  Training interrupted by user")
        if hasattr(trainer, '_print_timing_summary'):
            trainer._print_timing_summary()
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
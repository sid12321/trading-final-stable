#!/usr/bin/env python3
"""
run_optimized_model_training.py - Complete optimized model training pipeline

This script runs the complete hyperparameter optimization and model training process:
1. MCMC hyperparameter optimization (15 iterations, 5 burn-in)
2. Model training with optimized hyperparameters
3. Final model training with full BASEMODELITERATIONS=100000
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

# Setup path
basepath = '/Users/skumar81/Desktop/Personal/trading-final-stable'
os.chdir(basepath)

from parameters import *
from hyperparameter_optimizer import run_hyperparameter_optimization
from model_trainer import ModelTrainer

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def run_complete_optimization_pipeline():
    """Run the complete optimization and training pipeline"""
    
    print("="*80)
    print("COMPLETE OPTIMIZED MODEL TRAINING PIPELINE")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target symbols: {TESTSYMBOLS}")
    print(f"Hyperparameters to optimize: GLOBALLEARNINGRATE, N_EPOCHS, ENT_COEF, N_STEPS, TARGET_KL, GAE_LAMBDA, BATCH_SIZE, SDE_SAMPLE_FREQ")
    print()
    
    # Step 1: Hyperparameter optimization with MCMC
    print("STEP 1: HYPERPARAMETER OPTIMIZATION")
    print("-" * 40)
    
    # Temporarily reduce BASEMODELITERATIONS for MCMC to save time
    original_iterations = BASEMODELITERATIONS
    globals()['BASEMODELITERATIONS'] = REDUCEDMCMCITERATIONS  # Reduced for MCMC iterations
    
    print(f"Using reduced BASEMODELITERATIONS={BASEMODELITERATIONS} for MCMC optimization")
    
    try:
        optimization_results = run_hyperparameter_optimization(
            max_iterations=15,
            burn_in=5,
            resume=False  # Start fresh optimization
        )
        
        best_hyperparams = optimization_results['best_params']
        best_reward = optimization_results['best_reward']
        
        print(f"Hyperparameter optimization completed!")
        print(f"Best reward achieved: {best_reward:.6f}")
        print("Optimized hyperparameters:")
        for param, value in best_hyperparams.items():
            print(f"  {param}: {value}")
        
    except Exception as e:
        print(f"Error in hyperparameter optimization: {e}")
        print("Using default hyperparameters for final training...")
        best_hyperparams = {
            'GLOBALLEARNINGRATE': GLOBALLEARNINGRATE,
            'N_EPOCHS': N_EPOCHS,
            'ENT_COEF': ENT_COEF,
            'N_STEPS': N_STEPS,
            'TARGET_KL': TARGET_KL,
            'GAE_LAMBDA': GAE_LAMBDA,
            'BATCH_SIZE': BATCH_SIZE,
            'SDE_SAMPLE_FREQ': SDE_SAMPLE_FREQ
        }
    
    # Step 2: Final model training with optimized hyperparameters
    print("\n" + "="*80)
    print("STEP 2: FINAL MODEL TRAINING WITH OPTIMIZED HYPERPARAMETERS")
    print("-" * 60)
    
    # Restore full BASEMODELITERATIONS for final training
    globals()['BASEMODELITERATIONS'] = original_iterations
    print(f"Using full BASEMODELITERATIONS={BASEMODELITERATIONS} for final model training")
    
    # Update global parameters with optimized values
    print("Updating global parameters with optimized hyperparameters...")
    for param, value in best_hyperparams.items():
        if param in globals():
            old_value = globals()[param]
            globals()[param] = value
            print(f"  {param}: {old_value} -> {value}")
    
    # Initialize trainer and run final training
    print("\nInitializing model trainer for final training...")
    trainer = ModelTrainer()
    
    try:
        # Run complete training pipeline with optimized hyperparameters
        allmodels, globalsignals, lol, qtnorm = trainer.run_full_training_pipeline()
        
        print("Final model training completed successfully!")
        print(f"Models trained for symbols: {list(allmodels.keys())}")
        print(f"Global signals identified: {len(globalsignals)}")
        
        # Save final configuration
        final_config = {
            'optimization_completed': True,
            'optimization_results': {
                'best_reward': float(best_reward) if 'best_reward' in locals() else None,
                'total_iterations': optimization_results.get('total_iterations', 0) if 'optimization_results' in locals() else 0
            },
            'final_hyperparameters': best_hyperparams,
            'training_completed': True,
            'models_trained': list(allmodels.keys()),
            'global_signals_count': len(globalsignals),
            'final_basemodeliterations': BASEMODELITERATIONS,
            'completion_timestamp': datetime.now().isoformat()
        }
        
        # Convert numpy types to JSON-serializable types
        final_config = convert_numpy_types(final_config)
        
        with open(f"{basepath}/final_optimized_model_config.json", 'w') as f:
            json.dump(final_config, f, indent=2)
        
        print(f"Final configuration saved to: {basepath}/final_optimized_model_config.json")
        
    except Exception as e:
        print(f"Error in final model training: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Validation and summary
    print("\n" + "="*80)
    print("STEP 3: VALIDATION AND SUMMARY")
    print("-" * 30)
    
    # Verify model files exist
    models_created = []
    for symbol in TESTSYMBOLS:
        model_path = f"{basepath}/models/{symbol}localmodel.zip"
        if os.path.exists(model_path):
            models_created.append(symbol)
            file_size = os.path.getsize(model_path) / (1024*1024)  # MB
            print(f"✓ Model created for {symbol}: {file_size:.1f} MB")
        else:
            print(f"✗ Model missing for {symbol}")
    
    print(f"\nSUMMARY:")
    print(f"- Hyperparameter optimization: {'✓ Completed' if 'optimization_results' in locals() else '✗ Failed'}")
    print(f"- MCMC iterations: {optimization_results.get('total_iterations', 0) if 'optimization_results' in locals() else 0}")
    print(f"- Best reward: {best_reward:.6f}" if 'best_reward' in locals() else "- Best reward: N/A")
    print(f"- Final training: {'✓ Completed' if len(models_created) > 0 else '✗ Failed'}")
    print(f"- Models created: {len(models_created)}/{len(TESTSYMBOLS)} symbols")
    print(f"- Final BASEMODELITERATIONS: {BASEMODELITERATIONS}")
    print(f"- Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if len(models_created) == len(TESTSYMBOLS):
        print("\n🎉 COMPLETE SUCCESS: All models trained with optimized hyperparameters!")
        return True
    else:
        print(f"\n⚠️  PARTIAL SUCCESS: {len(models_created)}/{len(TESTSYMBOLS)} models created")
        return False

if __name__ == "__main__":
    success = run_complete_optimization_pipeline()
    sys.exit(0 if success else 1)
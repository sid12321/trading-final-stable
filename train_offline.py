#!/usr/bin/env python3
"""
train_offline.py - Train models using existing preprocessed data without Kite login
"""

import os
import sys
import warnings

# Suppress CUDA warnings
warnings.filterwarnings('ignore', message='CUDA initialization')
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Import parameters and utilities
basepath = '/Users/skumar81/Desktop/Personal/trading-final-stable'
os.chdir(basepath)

# Temporarily disable preprocessing to avoid Kite login
import parameters
original_preprocess = parameters.PREPROCESS
parameters.PREPROCESS = False

from model_trainer import ModelTrainer

if __name__ == "__main__":
    print("Training models offline using existing preprocessed data...")
    
    # Create trainer instance
    trainer = ModelTrainer()
    
    # Skip preprocessing and go directly to training
    print("Skipping data preprocessing (using existing data)")
    
    # Load historical data instead of preprocessing
    trainer.load_historical_data()
    
    # Extract signals from loaded data
    globalsignals = trainer.extract_signals()
    
    # Train models with existing data
    print("Training models with current parameters...")
    reward = trainer.train_models_with_params()
    
    if reward is not None:
        print(f"Training completed! Final reward: {reward}")
    else:
        print("Training failed or was interrupted")
    
    # Restore original preprocessing flag
    parameters.PREPROCESS = original_preprocess
    
    print("Offline training completed!")
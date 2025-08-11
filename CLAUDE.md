# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git Repository Details

- **Repository URL**: https://github.com/sid12321/trading-final-stable
- **Remote**: origin -> https://github.com/sid12321/trading-final-stable.git
- **Default Branch**: master
- **Username**: sid12321

## Project Overview

Algorithmic trading system using PPO reinforcement learning with MCMC hyperparameter optimization, real-time tick processing, and JAX acceleration support.

## Quick Start Commands

### Environment Setup
```bash
# Activate environment
source activate_env.sh  # or: source venv/bin/activate

# Verify dependencies
pip list | grep -E "stable-baselines3|torch|jax"
```

### Training Pipeline
```bash
# Quick test training (uses existing data)
python train_only.py --symbols BPCL --no-posterior

# Full training with specific symbols
python train_only.py --symbols BPCL HDFCLIFE

# Training with new models from scratch
python train_only.py --new-model --symbols BPCL

# Complete pipeline with preprocessing (requires Kite login)
python train_offline.py
```

### Hyperparameter Optimization
```bash
# Quick test (5 iterations, 2 burn-in)
python hyperparameter_optimizer.py  # Runs with default settings

# Full optimization (30 iterations)
python run_optimized_model_training.py

# Resume from previous chain
# Note: Automatically resumes from mcmc_chain.pkl if present
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_bounded_entropy.py -v  # Entropy bounds
pytest tests/test_trading_improvements.py -v  # Trading logic
pytest tests/test_value_loss_clipping.py -v  # Value loss bounds

# Performance benchmarks
python benchmark_training.py  # Training speed test
python test_gpu_utilization.py  # GPU usage verification
```

### Monitoring
```bash
# Start TensorBoard (auto-cleans old logs)
./start_tensorboard.sh
# Access at http://localhost:6006

# Stop all TensorBoard instances
./stop_tensorboard.sh

# Monitor training in real-time
python monitor_training.py
```

### Live Trading
```bash
# Live trading with Kite Connect
python trader.py  # Primary implementation
```

## Architecture

### Data Flow
```
1. Data Preprocessing (if enabled)
   └─> kitelogin.py → Fetch historical data
   └─> lib.py → Generate 110+ technical indicators
   └─> common.py → Quantile normalization
   └─> Save to traindata/finalmldf{SYMBOL}.csv

2. Model Training
   └─> model_trainer.py → Orchestrates training
   └─> StockTradingEnv2.py → Custom Gym environment
   └─> bounded_entropy_ppo.py → Enhanced PPO with bounds
   └─> jax_accelerated_ppo_fixed.py → JAX acceleration (optional)
   └─> Save to models/{SYMBOL}_model.zip

3. Hyperparameter Optimization
   └─> hyperparameter_optimizer.py → MCMC optimization
   └─> Evaluates on validation data
   └─> Saves best params to hyperparameter_results.json

4. Live Trading
   └─> trader.py → Real-time tick processing
   └─> Converts ticks to OHLCV candles
   └─> Applies trained models for decisions
```

### Key Components

**Core Training (`bounded_entropy_ppo.py`)**
- Entropy loss bounded to [-1, 1] via torch.clamp
- Value loss clipping with configurable bounds
- Mixed precision support for GPU optimization
- Dynamic scheduling for learning rate, entropy coefficient

**Environment (`StockTradingEnv2.py`)**
- Daily liquidation at 15:15 IST
- Position limits and risk management
- Reward shaping with P&L tracking
- Action space: Buy/Sell/Hold with continuous amounts

**Signal Generation (`lib.py`, `common.py`)**
- Technical indicators: RSI, MACD, Bollinger Bands, Stochastic
- Market regime detection (bear/bull signals)
- Quantile normalization for all features
- Multi-period lag features (1-23 days)

**Optimization (`hyperparameter_optimizer.py`)**
- MCMC with Metropolis-Hastings algorithm
- 12 parameters: learning rate, epochs, entropy coef, etc.
- Adaptive proposal distributions
- Chain persistence in mcmc_chain.pkl

## Key Parameters (`parameters.py`)

### Critical Settings
```python
TESTSYMBOLS = ['BPCL']  # Symbols for testing
N_ITERATIONS = 100000   # Training iterations
GLOBALLEARNINGRATE = 3e-4  # Base learning rate
ENT_COEF = 0.01  # Entropy coefficient
ENTROPY_BOUND = 1.0  # Entropy loss bound
VALUE_LOSS_BOUND = 10.0  # Value loss bound
```

### Hardware Detection
- Auto-detects CUDA/MPS/CPU
- Platform-specific optimizations
- Memory management per device type

## File Structure

```
├── Core Components
│   ├── bounded_entropy_ppo.py      # Enhanced PPO algorithm
│   ├── StockTradingEnv2.py        # Trading environment
│   ├── model_trainer.py           # Training orchestration
│   └── parameters.py               # Central configuration
│
├── Training Scripts
│   ├── train_only.py               # Training without preprocessing
│   ├── train_offline.py           # Full pipeline
│   └── run_optimized_model_training.py  # Complete optimization
│
├── Trading
│   ├── trader.py                   # Live trading implementation
│   └── kitelogin.py               # Kite Connect authentication
│
├── Optimization
│   ├── hyperparameter_optimizer.py # MCMC optimization
│   └── schedule_utils.py          # Dynamic scheduling
│
├── Data & Models
│   ├── traindata/                  # Historical data CSVs
│   ├── models/                     # Trained models & normalizers
│   └── tmp/                        # Checkpoints & logs
│
└── Testing
    └── tests/                      # Comprehensive test suite
```

## Common Development Tasks

### Debug Training Issues
```bash
# Check for NaN/Inf in rewards
python tests/test_reward_bounds.py

# Verify entropy bounds
python tests/test_bounded_entropy_simple.py

# Monitor GPU memory
watch -n 1 'nvidia-smi' # NVIDIA
# or for Apple Silicon:
python monitor_performance.py
```

### Modify Hyperparameters
1. Edit `parameters.py` for manual changes
2. Or use MCMC optimization to find optimal values
3. Results saved to `hyperparameter_results.json`

### Add New Technical Indicators
1. Add calculation in `lib.py` generate_signals()
2. Update feature list in `common.py`
3. Retrain models with new features

### Resume Training
```bash
# Models automatically checkpoint to tmp/checkpoints/
# Resume by running same command - detects existing checkpoints
python train_only.py --symbols BPCL
```

## Performance Optimization

### GPU Utilization
- JAX acceleration: Auto-enabled if available
- Mixed precision: Set USE_MIXED_PRECISION=True in parameters.py
- Batch size tuning: Adjust BATCH_SIZE for GPU memory

### Training Speed
- Reduce N_ITERATIONS for testing (default: 100000)
- Use --no-posterior flag to skip analysis
- Enable parallel environments (n_envs in parameters.py)

### Memory Management
- Automatic garbage collection between epochs
- GPU memory cleanup in lib.py
- Checkpoint pruning in tmp/checkpoints/

## Troubleshooting

### Import Errors
```bash
# Verify TA-Lib installation
python -c "import talib; print(talib.__version__)"
# If missing: brew install ta-lib && pip install TA-Lib
```

### GPU Not Detected
```bash
# Check device detection
python -c "from parameters import DEVICE; print(f'Using: {DEVICE}')"
```

### Training Instability
- Check entropy bounds are applied (test_bounded_entropy.py)
- Verify reward scaling (test_reward_bounds.py)
- Review learning rate schedule

### Kite Login Issues
```bash
# Test authentication
python test_kite_login.py
# Update credentials in kitelogin.py if needed
```
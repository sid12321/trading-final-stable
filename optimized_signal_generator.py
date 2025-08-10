#!/usr/bin/env python3
"""
Optimized Signal Generator using MCMC/Simulated Annealing
for maximizing individual signal performance in simulate_trades_on_day
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# GPU parallelism support
try:
    import torch
    import torch.multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Detect optimal device
def get_optimal_device():
    """Detect and return the optimal compute device"""
    if TORCH_AVAILABLE:
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            if os.environ.get('SIGNAL_GPU_INIT_PRINTED') != '1':
                print(f"CUDA GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
                os.environ['SIGNAL_GPU_INIT_PRINTED'] = '1'
            return device, torch.cuda.device_count()
    
    device = "cpu"
    import multiprocessing as mp_cpu
    cpu_cores = mp_cpu.cpu_count()
    # Remove duplicate CPU core message - already printed in parameters.py
    return device, cpu_cores

DEVICE, NUM_CORES = get_optimal_device()

def create_optimized_signal(values, lower_quantile=0.3, upper_quantile=0.7, 
                          signal_strength=1.0, smoothing_window=1):
    """
    Enhanced signal creation with optimizable hyperparameters
    
    Parameters:
    - lower_quantile: Threshold for buy signals (default 0.3)
    - upper_quantile: Threshold for sell signals (default 0.7) 
    - signal_strength: Multiplier for signal strength (default 1.0)
    - smoothing_window: Window for signal smoothing (default 1, no smoothing)
    """
    values = np.array(values)
    
    # Apply smoothing if window > 1
    if smoothing_window > 1:
        smoothed_values = pd.Series(values).rolling(
            window=min(smoothing_window, len(values)), 
            min_periods=1
        ).mean().values
    else:
        smoothed_values = values
    
    # Calculate dynamic quantiles
    lq = np.nanquantile(smoothed_values, lower_quantile)
    uq = np.nanquantile(smoothed_values, upper_quantile)
    
    signal = np.zeros_like(smoothed_values)
    signal[smoothed_values <= lq] = signal_strength
    signal[smoothed_values >= uq] = -signal_strength
    signal[np.isnan(smoothed_values)] = 0
    
    return signal

def simulate_trades_on_day_gpu(vwap, action, transaction_cost=0.001, use_gpu=True):
    """
    GPU-accelerated trading simulation using CuPy or PyTorch
    """
    if use_gpu and DEVICE == "cuda":
        if CUPY_AVAILABLE:
            return _simulate_trades_cupy(vwap, action, transaction_cost)
        elif TORCH_AVAILABLE:
            return _simulate_trades_torch(vwap, action, transaction_cost)
    
    # Fallback to CPU implementation
    return simulate_trades_on_day_enhanced(vwap, action, transaction_cost)

def _simulate_trades_cupy(vwap, action, transaction_cost):
    """CuPy implementation for CUDA GPUs"""
    vwap_gpu = cp.array(vwap)
    action_gpu = cp.array(action)
    
    if len(vwap_gpu) != len(action_gpu):
        return [0, 0]
    
    position = 0
    pnl = 0
    total_trades = 0
    
    # Vectorized operations on GPU
    buy_mask = action_gpu > 0
    sell_mask = action_gpu < 0
    
    # Calculate trades and PnL using GPU vectorization
    buy_trades = cp.sum(buy_mask[:-1])
    sell_trades = cp.sum(sell_mask[:-1])
    total_trades = int(buy_trades + sell_trades)
    
    # Calculate position changes
    position_changes = action_gpu[:-1]
    price_impacts = vwap_gpu[1:]
    
    # Calculate PnL
    pnl = float(-cp.sum(position_changes * price_impacts))
    final_position = float(cp.sum(position_changes))
    
    # Close position at day end
    pnl += final_position * float(vwap_gpu[-1])
    
    # Apply transaction costs
    transaction_cost_total = total_trades * transaction_cost * float(cp.mean(vwap_gpu))
    pnl -= transaction_cost_total
    
    return [float(pnl), 0]

def _simulate_trades_torch(vwap, action, transaction_cost):
    """PyTorch implementation for CUDA GPUs"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vwap_tensor = torch.tensor(vwap, dtype=torch.float32, device=device)
    action_tensor = torch.tensor(action, dtype=torch.float32, device=device)
    
    if len(vwap_tensor) != len(action_tensor):
        return [0, 0]
    
    # Vectorized operations on GPU
    buy_mask = action_tensor > 0
    sell_mask = action_tensor < 0
    
    buy_trades = torch.sum(buy_mask[:-1])
    sell_trades = torch.sum(sell_mask[:-1])
    total_trades = int(buy_trades + sell_trades)
    
    # Calculate PnL using tensor operations
    position_changes = action_tensor[:-1]
    price_impacts = vwap_tensor[1:]
    
    pnl = -torch.sum(position_changes * price_impacts)
    final_position = torch.sum(position_changes)
    
    # Close position at day end
    pnl += final_position * vwap_tensor[-1]
    
    # Apply transaction costs
    transaction_cost_total = total_trades * transaction_cost * torch.mean(vwap_tensor)
    pnl -= transaction_cost_total
    
    return [float(pnl.cpu()), 0]

def simulate_trades_on_day_enhanced(vwap, action, transaction_cost=0.001):
    """
    Enhanced trading simulation with transaction costs
    """
    if len(vwap) != len(action):
        return [0, 0]
    
    position = 0
    pnl = 0
    total_trades = 0
    
    for i in range(len(action) - 1):
        if action[i] > 0:  # Buy signal
            position += action[i]
            pnl -= vwap[i + 1] * action[i]
            total_trades += 1
        elif action[i] < 0:  # Sell signal
            position += action[i]  # action[i] is negative
            pnl -= vwap[i + 1] * action[i]  # This adds to pnl since action[i] < 0
            total_trades += 1
    
    # Close all positions at day end
    pnl += position * vwap[-1]
    
    # Apply transaction costs
    transaction_cost_total = total_trades * transaction_cost * np.mean(vwap)
    pnl -= transaction_cost_total
    
    return [pnl, 0]  # Always close positions at day end

def objective_function(params, values, vwap_data, dates, use_gpu=True):
    """
    Objective function to maximize: average PnL across all trading days
    
    Parameters:
    - params: [lower_quantile, upper_quantile, signal_strength, smoothing_window]
    - values: Signal input data
    - vwap_data: VWAP data for simulation
    - dates: Date data for grouping
    - use_gpu: Whether to use GPU acceleration
    """
    try:
        lower_q, upper_q, strength, smooth_win = params
        
        # Parameter bounds enforcement
        lower_q = np.clip(lower_q, 0.05, 0.45)
        upper_q = np.clip(upper_q, 0.55, 0.95)
        strength = np.clip(strength, 0.1, 3.0)
        smooth_win = max(1, int(smooth_win))
        
        # Ensure lower_q < upper_q
        if lower_q >= upper_q:
            return -1000  # Heavy penalty
        
        # Create optimized signal
        signal = create_optimized_signal(values, lower_q, upper_q, strength, smooth_win)
        
        # Simulate trades by date
        total_pnl = 0
        trade_count = 0
        
        df = pd.DataFrame({'vwap': vwap_data, 'signal': signal, 'date': dates})
        
        for date_val, group in df.groupby('date'):
            if len(group) < 2:
                continue
                
            group = group.sort_index()
            vwap_day = group['vwap'].tolist()
            signal_day = group['signal'].tolist()
            
            if len(vwap_day) != len(signal_day):
                continue
                
            if use_gpu:
                pnl, _ = simulate_trades_on_day_gpu(vwap_day, signal_day)
            else:
                pnl, _ = simulate_trades_on_day_enhanced(vwap_day, signal_day)
            total_pnl += pnl
            trade_count += 1
        
        if trade_count == 0:
            return -1000
            
        # Return negative for minimization (we want to maximize PnL)
        avg_pnl = total_pnl / trade_count
        
        # Add regularization to prevent overfitting
        complexity_penalty = abs(strength - 1.0) * 0.1 + abs(smooth_win - 1) * 0.05
        
        return -(avg_pnl - complexity_penalty)
        
    except Exception as e:
        return -1000  # Heavy penalty for errors

def simulated_annealing_optimize(values, vwap_data, dates, max_iterations=1000, use_gpu=True):
    """
    Simulated Annealing optimization for signal hyperparameters
    """
    # Initial parameters [lower_quantile, upper_quantile, signal_strength, smoothing_window]
    current_params = np.array([0.3, 0.7, 1.0, 1.0])
    current_score = objective_function(current_params, values, vwap_data, dates, use_gpu)
    
    best_params = current_params.copy()
    best_score = current_score
    
    # Annealing schedule
    initial_temp = 100.0
    final_temp = 0.01
    
    for iteration in range(max_iterations):
        # Calculate temperature
        temp = initial_temp * (final_temp / initial_temp) ** (iteration / max_iterations)
        
        # Generate candidate solution (small random perturbation)
        perturbation = np.random.normal(0, 0.05, 4)
        candidate_params = current_params + perturbation
        
        # Enforce bounds
        candidate_params[0] = np.clip(candidate_params[0], 0.05, 0.45)  # lower_quantile
        candidate_params[1] = np.clip(candidate_params[1], 0.55, 0.95)  # upper_quantile
        candidate_params[2] = np.clip(candidate_params[2], 0.1, 3.0)    # signal_strength
        candidate_params[3] = np.clip(candidate_params[3], 1.0, 10.0)   # smoothing_window
        
        # Evaluate candidate
        candidate_score = objective_function(candidate_params, values, vwap_data, dates, use_gpu)
        
        # Accept or reject candidate
        delta = candidate_score - current_score
        
        if delta < 0 or np.random.random() < np.exp(-delta / temp):
            current_params = candidate_params
            current_score = candidate_score
            
            # Update best solution
            if candidate_score < best_score:
                best_params = candidate_params.copy()
                best_score = candidate_score
    
    return best_params, -best_score  # Return positive PnL

def mcmc_optimize(values, vwap_data, dates, n_samples=1000, use_gpu=True):
    """
    MCMC optimization using Metropolis-Hastings algorithm
    """
    # Initial parameters
    current_params = np.array([0.3, 0.7, 1.0, 1.0])
    current_score = objective_function(current_params, values, vwap_data, dates, use_gpu)
    
    best_params = current_params.copy()
    best_score = current_score
    
    accepted_samples = []
    acceptance_count = 0
    
    for i in range(n_samples):
        # Propose new parameters
        proposal_std = max(0.01, 0.1 * (1 - i / n_samples))  # Decreasing std over time
        proposal = current_params + np.random.normal(0, proposal_std, 4)
        
        # Enforce bounds
        proposal[0] = np.clip(proposal[0], 0.05, 0.45)
        proposal[1] = np.clip(proposal[1], 0.55, 0.95)
        proposal[2] = np.clip(proposal[2], 0.1, 3.0)
        proposal[3] = np.clip(proposal[3], 1.0, 10.0)
        
        # Evaluate proposal
        proposal_score = objective_function(proposal, values, vwap_data, dates, use_gpu)
        
        # Acceptance probability (using exp for numerical stability)
        alpha = min(1.0, np.exp(current_score - proposal_score))
        
        if np.random.random() < alpha:
            current_params = proposal
            current_score = proposal_score
            acceptance_count += 1
            
            # Update best solution
            if proposal_score < best_score:
                best_params = proposal.copy()
                best_score = proposal_score
        
        accepted_samples.append(current_params.copy())
    
    print(f"MCMC acceptance rate: {acceptance_count / n_samples:.2%}")
    return best_params, -best_score

def optimize_single_signal_worker(args):
    """
    Worker function for parallel signal optimization
    """
    var, signalmultiplier, signal_data, vwap_data, dates, method, use_gpu = args
    
    try:
        adjusted_signal_data = signalmultiplier * signal_data
        
        # Optimize hyperparameters
        if method == 'simulated_annealing':
            best_params, best_pnl = simulated_annealing_optimize(
                adjusted_signal_data, vwap_data, dates, use_gpu=use_gpu
            )
        elif method == 'mcmc':
            best_params, best_pnl = mcmc_optimize(
                adjusted_signal_data, vwap_data, dates, use_gpu=use_gpu
            )
        else:
            raise ValueError("Method must be 'simulated_annealing' or 'mcmc'")
        
        return {
            'var': var,
            'signalmultiplier': signalmultiplier,
            'best_params': best_params,
            'best_pnl': best_pnl,
            'success': True
        }
    except Exception as e:
        return {
            'var': var,
            'signalmultiplier': signalmultiplier,
            'error': str(e),
            'success': False
        }

def optimize_signal_hyperparameters(values, vwap_data, dates, method='simulated_annealing', use_gpu=True):
    """
    Main function to optimize signal hyperparameters
    
    Returns:
    - best_params: Optimized hyperparameters
    - best_pnl: Best average PnL achieved
    """
    if method == 'simulated_annealing':
        return simulated_annealing_optimize(values, vwap_data, dates, use_gpu=use_gpu)
    elif method == 'mcmc':
        return mcmc_optimize(values, vwap_data, dates, use_gpu=use_gpu)
    else:
        raise ValueError("Method must be 'simulated_annealing' or 'mcmc'")

def generate_optimized_signals_for_dataframe(mldf, signalcolumns, method='simulated_annealing', 
                                           use_parallel=True, use_gpu=True, max_workers=None):
    """
    Generate optimized signals for all signal columns in a dataframe with GPU parallelism
    
    Parameters:
    - mldf: DataFrame with trading data
    - signalcolumns: List of signal column names to optimize
    - method: Optimization method ('simulated_annealing' or 'mcmc')
    - use_parallel: Whether to use parallel processing across signals
    - use_gpu: Whether to use GPU acceleration
    - max_workers: Maximum number of parallel workers (None = auto-detect)
    
    Returns:
    - optimized_signals: Dictionary with optimized parameters for each signal
    - enhanced_pnlframe: PnL results for optimized signals
    """
    optimized_signals = {}
    enhanced_pnlframe = pd.DataFrame()
    
    # Determine optimal number of workers
    if max_workers is None:
        max_workers = min(NUM_CORES, len(signalcolumns) * 2)  # 2 workers per signal (pos/neg multiplier)
    
    print(f"Optimizing {len(signalcolumns)} signals using {method}...")
    print(f"GPU acceleration: {use_gpu}, Parallel processing: {use_parallel}, Workers: {max_workers}")
    
    if use_parallel and len(signalcolumns) > 1:
        # Parallel processing across multiple signals
        optimization_tasks = []
        
        for var in signalcolumns:
            # Skip if insufficient data
            if var not in mldf.columns:
                continue
            
            signal_data = mldf[var].values
            vwap_data = mldf['vwap'].values
            dates = mldf['date'].values
            
            if len(signal_data) < 100 or len(np.unique(dates)) < 5:
                print(f"  Skipping {var}: insufficient data")
                continue
            
            # Add tasks for both positive and negative signal multipliers
            for signalmultiplier in [1, -1]:
                optimization_tasks.append((
                    var, signalmultiplier, signal_data, vwap_data, dates, method, use_gpu
                ))
        
        print(f"Created {len(optimization_tasks)} optimization tasks")
        
        # Execute parallel optimization
        if DEVICE == "cuda" and use_gpu:
            # Use ThreadPoolExecutor for GPU work (better for I/O bound GPU operations)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(optimize_single_signal_worker, optimization_tasks))
        else:
            # Use ProcessPoolExecutor for CPU work
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(optimize_single_signal_worker, optimization_tasks))
        
        # Process results
        for result in results:
            if result['success']:
                var = result['var']
                signalmultiplier = result['signalmultiplier']
                best_params = result['best_params']
                best_pnl = result['best_pnl']
                
                # Store optimized parameters
                opt_key = f"{var}_mult_{signalmultiplier}"
                optimized_signals[opt_key] = {
                    'var': var,
                    'signalmultiplier': signalmultiplier,
                    'lower_quantile': best_params[0],
                    'upper_quantile': best_params[1],
                    'signal_strength': best_params[2],
                    'smoothing_window': int(best_params[3]),
                    'optimized_pnl': best_pnl
                }
                
                print(f"  {opt_key}: PnL = {best_pnl:.4f}, params = {best_params}")
            else:
                print(f"  Error optimizing {result['var']}: {result.get('error', 'Unknown error')}")
        
        # Generate optimized signals and PnL for parallel results
        for opt_key, params in optimized_signals.items():
            var = params['var']
            signalmultiplier = params['signalmultiplier']
            
            # Generate optimized signal
            signal_data = mldf[var].values
            adjusted_signal_data = signalmultiplier * signal_data
            optimized_signal = create_optimized_signal(
                adjusted_signal_data,
                params['lower_quantile'], params['upper_quantile'], 
                params['signal_strength'], params['smoothing_window']
            )
            
            # Add optimized signal to dataframe
            opt_signal_col = f'opt_action_{var}_mult_{signalmultiplier}'
            mldf[opt_signal_col] = optimized_signal
            
            # Simulate trades and calculate PnL for each date
            results = []
            for date_val, group in mldf.groupby('date'):
                group = group.sort_values('t')
                if len(group) < 2:
                    continue
                
                if use_gpu:
                    pnl, position = simulate_trades_on_day_gpu(
                        group['vwap'].tolist(), 
                        group[opt_signal_col].tolist()
                    )
                else:
                    pnl, position = simulate_trades_on_day_enhanced(
                        group['vwap'].tolist(), 
                        group[opt_signal_col].tolist()
                    )
                results.append({
                    'date': date_val, 
                    'pnl': pnl, 
                    'position': position,
                    'var': var,
                    'signalmultiplier': signalmultiplier,
                    'optimized': True
                })
            
            # Add to enhanced_pnlframe
            if results:
                results_df = pd.DataFrame(results)
                enhanced_pnlframe = pd.concat([enhanced_pnlframe, results_df])
    
    else:
        # Sequential processing (original method)
        for i, var in enumerate(signalcolumns):
            print(f"Optimizing signal {i+1}/{len(signalcolumns)}: {var}")
            
            try:
                # Extract data for this signal
                signal_data = mldf[var].values
                vwap_data = mldf['vwap'].values
                dates = mldf['date'].values
                
                # Skip if insufficient data
                if len(signal_data) < 100 or len(np.unique(dates)) < 5:
                    print(f"  Skipping {var}: insufficient data")
                    continue
                
                # Optimize for both positive and negative signal multipliers
                for signalmultiplier in [1, -1]:
                    adjusted_signal_data = signalmultiplier * signal_data
                    
                    # Optimize hyperparameters
                    best_params, best_pnl = optimize_signal_hyperparameters(
                        adjusted_signal_data, vwap_data, dates, method=method, use_gpu=use_gpu
                    )
                    
                    # Store optimized parameters
                    opt_key = f"{var}_mult_{signalmultiplier}"
                    optimized_signals[opt_key] = {
                        'var': var,
                        'signalmultiplier': signalmultiplier,
                        'lower_quantile': best_params[0],
                        'upper_quantile': best_params[1],
                        'signal_strength': best_params[2],
                        'smoothing_window': int(best_params[3]),
                        'optimized_pnl': best_pnl
                    }
                    
                    # Generate optimized signal
                    optimized_signal = create_optimized_signal(
                        adjusted_signal_data,
                        best_params[0], best_params[1], 
                        best_params[2], int(best_params[3])
                    )
                    
                    # Add optimized signal to dataframe
                    opt_signal_col = f'opt_action_{var}_mult_{signalmultiplier}'
                    mldf[opt_signal_col] = optimized_signal
                    
                    # Simulate trades and calculate PnL for each date
                    results = []
                    for date_val, group in mldf.groupby('date'):
                        group = group.sort_values('t')
                        if len(group) < 2:
                            continue
                        
                        if use_gpu:
                            pnl, position = simulate_trades_on_day_gpu(
                                group['vwap'].tolist(), 
                                group[opt_signal_col].tolist()
                            )
                        else:
                            pnl, position = simulate_trades_on_day_enhanced(
                                group['vwap'].tolist(), 
                                group[opt_signal_col].tolist()
                            )
                        results.append({
                            'date': date_val, 
                            'pnl': pnl, 
                            'position': position,
                            'var': var,
                            'signalmultiplier': signalmultiplier,
                            'optimized': True
                        })
                    
                    # Add to enhanced_pnlframe
                    if results:
                        results_df = pd.DataFrame(results)
                        enhanced_pnlframe = pd.concat([enhanced_pnlframe, results_df])
                    
                    print(f"  {opt_key}: PnL = {best_pnl:.4f}, params = {best_params}")
                        
            except Exception as e:
                print(f"  Error optimizing {var}: {e}")
                continue
    
    return optimized_signals, enhanced_pnlframe

if __name__ == "__main__":
    # Test the optimization functions
    print("Testing optimized signal generator...")
    
    # Create synthetic test data
    np.random.seed(42)
    n_days = 10
    n_points_per_day = 50
    
    dates = np.repeat(pd.date_range('2023-01-01', periods=n_days), n_points_per_day)
    prices = 100 + np.cumsum(np.random.normal(0, 0.1, len(dates)))
    signal_values = np.random.normal(0, 1, len(dates))
    
    # Test optimization
    best_params, best_pnl = optimize_signal_hyperparameters(
        signal_values, prices, dates, method='simulated_annealing'
    )
    
    print(f"Best parameters: {best_params}")
    print(f"Best PnL: {best_pnl:.4f}")
    print("✅ Optimization test completed successfully!")
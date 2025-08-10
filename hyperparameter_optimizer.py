#!/usr/bin/env python3
"""
hyperparameter_optimizer.py - MCMC-based hyperparameter optimization

This module implements MCMC (Markov Chain Monte Carlo) optimization for hyperparameter tuning
with the following key features:

1. MCMC sampling with Metropolis-Hastings algorithm
2. Adaptive proposal distributions
3. Multi-objective optimization (reward maximization + stability)
4. Parallel evaluation support
5. Automatic convergence detection
6. Best hyperparameter tracking and persistence
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import base modules
basepath = '/Users/skumar81/Desktop/Personal/trading-final-stable'
os.chdir(basepath)

from parameters import *
from model_trainer import ModelTrainer

# Use initial values when scheduling is enabled
if 'USE_ENT_SCHEDULE' in globals() and USE_ENT_SCHEDULE:
    HYPEROPT_ENT_COEF = INITIAL_ENT_COEF
else:
    HYPEROPT_ENT_COEF = ENT_COEF

if 'USE_TARGET_KL_SCHEDULE' in globals() and USE_TARGET_KL_SCHEDULE:
    HYPEROPT_TARGET_KL = INITIAL_TARGET_KL
else:
    HYPEROPT_TARGET_KL = TARGET_KL


class HyperparameterSpace:
    """Define the hyperparameter search space with bounds and types"""
    
    def __init__(self):
        # Define hyperparameters to optimize from parameters.py
        # These are the parameters marked with "#To be Optimized" 
        self.param_space = {
            # Core PPO hyperparameters
            'ENT_COEF': {
                'type': 'continuous',
                'bounds': (0.001, 0.1),
                'current': HYPEROPT_ENT_COEF,
                'proposal_std': 0.01
            },
            'N_STEPS': {
                'type': 'discrete', 
                'bounds': [128, 256, 512, 1024],
                'current': N_STEPS,
                'proposal_std': 1
            },
            'N_EPOCHS': {
                'type': 'discrete',
                'bounds': [1, 2, 4, 8, 10],
                'current': N_EPOCHS,
                'proposal_std': 1
            },
            'BATCH_SIZE': {
                'type': 'discrete',
                'bounds': [64, 128, 256, 512],
                'current': BATCH_SIZE,
                'proposal_std': 1
            },
            'TARGET_KL': {
                'type': 'continuous',
                'bounds': (0.01, 0.2),
                'current': HYPEROPT_TARGET_KL,
                'proposal_std': 0.01
            },
            'GAE_LAMBDA': {
                'type': 'continuous',
                'bounds': (0.8, 1.0),
                'current': GAE_LAMBDA,
                'proposal_std': 0.05
            },
            'GLOBALLEARNINGRATE': {
                'type': 'continuous',
                'bounds': (1e-6, 1e-3),
                'current': GLOBALLEARNINGRATE,
                'proposal_std': 1e-5,
                'log_scale': True
            },
            'CLIP_RANGE': {
                'type': 'continuous',
                'bounds': (0.1, 0.4),
                'current': CLIP_RANGE,
                'proposal_std': 0.02
            },
            'CLIP_RANGE_VF': {
                'type': 'continuous',
                'bounds': (0.1, 0.4),
                'current': CLIP_RANGE_VF,
                'proposal_std': 0.02
            },
            'VF_COEF': {
                'type': 'continuous',
                'bounds': (0.1, 1.0),
                'current': VF_COEF,
                'proposal_std': 0.05
            },
            'USE_SDE': {
                'type': 'categorical',
                'bounds': [True, False],
                'current': USE_SDE
            },
            'SDE_SAMPLE_FREQ': {
                'type': 'discrete',
                'bounds': [1, 2, 4, 8, 16],
                'current': SDE_SAMPLE_FREQ,
                'proposal_std': 1
            }
        }
        
        # Additional parameters that can be optimized but are more experimental
        self.experimental_params = {
        }
        
        self.param_names = list(self.param_space.keys())
        self.current_params = {name: config['current'] for name, config in self.param_space.items()}
    
    def sample_proposal(self, current_params: Dict, temperature: float = 1.0) -> Dict:
        """Sample a new hyperparameter configuration using adaptive proposals"""
        new_params = current_params.copy()
        
        for param_name, config in self.param_space.items():
            if np.random.random() < 0.3:  # 30% chance to modify each parameter
                if config['type'] == 'continuous':
                    current_val = current_params[param_name]
                    std = config['proposal_std'] * temperature
                    
                    if config.get('log_scale', False):
                        # Log-scale sampling for learning rate
                        log_current = np.log(current_val)
                        log_proposal = np.random.normal(log_current, std)
                        proposal = np.exp(log_proposal)
                    else:
                        proposal = np.random.normal(current_val, std)
                    
                    # Clip to bounds
                    proposal = np.clip(proposal, config['bounds'][0], config['bounds'][1])
                    new_params[param_name] = proposal
                    
                elif config['type'] == 'discrete':
                    if isinstance(config['bounds'], list):
                        # Discrete choice from list
                        new_params[param_name] = np.random.choice(config['bounds'])
                    else:
                        # Integer range
                        current_val = current_params[param_name]
                        std = max(1, int(config['proposal_std'] * temperature))
                        proposal = int(np.random.normal(current_val, std))
                        proposal = np.clip(proposal, config['bounds'][0], config['bounds'][1])
                        new_params[param_name] = proposal
                        
                elif config['type'] == 'categorical':
                    new_params[param_name] = np.random.choice(config['bounds'])
        
        return new_params
    
    def get_random_params(self) -> Dict:
        """Generate completely random hyperparameters within bounds"""
        random_params = {}
        
        for param_name, config in self.param_space.items():
            if config['type'] == 'continuous':
                if config.get('log_scale', False):
                    log_min, log_max = np.log(config['bounds'][0]), np.log(config['bounds'][1])
                    random_params[param_name] = np.exp(np.random.uniform(log_min, log_max))
                else:
                    random_params[param_name] = np.random.uniform(config['bounds'][0], config['bounds'][1])
                    
            elif config['type'] == 'discrete':
                if isinstance(config['bounds'], list):
                    random_params[param_name] = np.random.choice(config['bounds'])
                else:
                    random_params[param_name] = np.random.randint(config['bounds'][0], config['bounds'][1] + 1)
                    
            elif config['type'] == 'categorical':
                random_params[param_name] = np.random.choice(config['bounds'])
        
        return random_params


class MCMCOptimizer:
    """MCMC-based hyperparameter optimizer with adaptive sampling"""
    
    def __init__(self, max_iterations: int = 15, burn_in: int = 5, 
                 temperature_schedule: str = 'linear', parallel_chains: int = 1):
        self.param_space = HyperparameterSpace()
        self.max_iterations = max_iterations
        self.burn_in = burn_in
        self.temperature_schedule = temperature_schedule
        self.parallel_chains = parallel_chains
        
        # Initialize model trainer
        self.trainer = ModelTrainer()
        
        # MCMC state
        self.chain_history = []
        self.reward_history = []
        self.acceptance_history = []
        self.best_params = None
        self.best_reward = -np.inf
        
        # Adaptive parameters
        self.acceptance_rate = 0.0
        self.target_acceptance = 0.23  # Optimal for continuous parameters
        self.adaptation_frequency = 5  # Reduced for fewer iterations
        
        # Results storage
        self.results_file = f"{basepath}/hyperparameter_results.json"
        self.chain_file = f"{basepath}/mcmc_chain.pkl"
        
        print(f"MCMC Optimizer initialized:")
        print(f"  - Max iterations: {max_iterations}")
        print(f"  - Burn-in: {burn_in}")
        print(f"  - Parameters to optimize: {len(self.param_space.param_names)}")
        print(f"  - Parameter names: {self.param_space.param_names}")
    
    def evaluate_hyperparameters(self, hyperparams: Dict) -> Tuple[float, Dict]:
        """Evaluate hyperparameters by training model and getting reward"""
        try:
            print(f"\nEvaluating hyperparameters:")
            for name, value in hyperparams.items():
                print(f"  {name}: {value}")
            
            # Update global parameters temporarily
            original_params = {}
            for name, value in hyperparams.items():
                if name in globals():
                    original_params[name] = globals()[name]
                    globals()[name] = value
            
            # Load data if not already loaded
            if not hasattr(self, 'data_loaded'):
                print("Loading historical data...")
                self.trainer.load_historical_data()
                self.trainer.extract_signals()
                self.data_loaded = True
            
            # Train model with current hyperparameters
            print("Training model with current hyperparameters...")
            reward = self.trainer.train_models_with_params(hyperparams)
            
            # Restore original parameters
            for name, value in original_params.items():
                globals()[name] = value
            
            # Handle case where modeltrain returns None
            if reward is None:
                print("Warning: modeltrain returned None, extracting reward from metrics file")
                reward = self._extract_reward_from_metrics()
            
            # Calculate additional metrics for multi-objective optimization
            stability_penalty = 0
            
            # Penalize extreme values
            if hyperparams.get('ENT_COEF', 0.01) > 0.05:
                stability_penalty += abs(hyperparams['ENT_COEF'] - 0.02) * 10
            
            if hyperparams.get('GLOBALLEARNINGRATE', 1e-4) > 5e-4:
                stability_penalty += abs(hyperparams['GLOBALLEARNINGRATE'] - 1e-4) * 1000
            
            # Penalize very large batch sizes that might cause memory issues
            if hyperparams.get('BATCH_SIZE', 256) > 512:
                stability_penalty += 5
            
            final_reward = reward - stability_penalty
            
            metrics = {
                'raw_reward': reward,
                'stability_penalty': stability_penalty,
                'final_reward': final_reward
            }
            
            print(f"Raw reward: {reward:.4f}, Stability penalty: {stability_penalty:.4f}, Final: {final_reward:.4f}")
            
            return final_reward, metrics
            
        except Exception as e:
            print(f"Error evaluating hyperparameters: {e}")
            import traceback
            traceback.print_exc()
            return -1000.0, {'error': str(e)}  # Large penalty for failed evaluations
    
    def _extract_reward_from_metrics(self) -> float:
        """Extract reward from metrics file or through model evaluation"""
        try:
            # First try metrics file
            metrics_file = f"{basepath}/tmp/sb3_log/custom_metrics.txt"
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    lines = f.readlines()
                
                best_reward = 0.0
                for line in lines:
                    if 'explained_variance:' in line:
                        try:
                            explained_variance = float(line.split('explained_variance:')[1].strip())
                            best_reward = max(best_reward, explained_variance)
                        except (ValueError, IndexError):
                            continue
                
                if best_reward > 0:
                    print(f"Extracted reward from metrics: {best_reward}")
                    return best_reward
            
            # If metrics file doesn't exist or gives 0, try model evaluation
            print("Metrics file unavailable or zero reward, using model evaluation...")
            return self._extract_reward_from_evaluation()
                
        except Exception as e:
            print(f"Error extracting reward from metrics: {e}")
            return self._extract_reward_from_evaluation()
    
    def _extract_reward_from_evaluation(self) -> float:
        """Extract reward by evaluating the trained model"""
        try:
            reward = self.trainer.extract_reward_from_evaluation()
            print(f"Extracted reward from evaluation: {reward}")
            return reward
        except Exception as e:
            print(f"Error in evaluation-based reward extraction: {e}")
            return 0.001
    
    def get_temperature(self, iteration: int) -> float:
        """Get temperature for simulated annealing schedule"""
        if self.temperature_schedule == 'linear':
            return max(0.1, 1.0 - (iteration / self.max_iterations))
        elif self.temperature_schedule == 'exponential':
            return 0.1 + 0.9 * np.exp(-3 * iteration / self.max_iterations) 
        else:
            return 1.0
    
    def accept_proposal(self, current_reward: float, proposed_reward: float, 
                       temperature: float) -> bool:
        """Metropolis-Hastings acceptance criterion"""
        if proposed_reward > current_reward:
            return True
        else:
            prob = np.exp((proposed_reward - current_reward) / temperature)
            return np.random.random() < prob
    
    def adapt_proposals(self, iteration: int):
        """Adapt proposal distributions based on acceptance rate"""
        if iteration > 0 and iteration % self.adaptation_frequency == 0:
            recent_acceptances = self.acceptance_history[-self.adaptation_frequency:]
            self.acceptance_rate = np.mean(recent_acceptances)
            
            print(f"Iteration {iteration}: Acceptance rate = {self.acceptance_rate:.3f}")
            
            # Adapt proposal standard deviations
            adaptation_factor = 1.0
            if self.acceptance_rate < 0.15:  # Too low acceptance
                adaptation_factor = 0.8
            elif self.acceptance_rate > 0.35:  # Too high acceptance
                adaptation_factor = 1.2
            
            # Update proposal standard deviations
            for param_name, config in self.param_space.param_space.items():
                if 'proposal_std' in config:
                    config['proposal_std'] *= adaptation_factor
            
            print(f"Adapted proposal distributions by factor {adaptation_factor:.2f}")
    
    def run_optimization(self) -> Dict:
        """Run MCMC optimization"""
        print(f"\nStarting MCMC optimization with {self.max_iterations} iterations...")
        
        # Initialize with current parameters
        current_params = self.param_space.current_params.copy()
        current_reward, current_metrics = self.evaluate_hyperparameters(current_params)
        
        self.best_params = current_params.copy()
        self.best_reward = current_reward
        
        # Initialize chain
        self.chain_history = [current_params.copy()]
        self.reward_history = [current_reward]
        self.acceptance_history = []
        
        print(f"Initial configuration reward: {current_reward:.4f}")
        
        # MCMC iterations
        for iteration in range(1, self.max_iterations + 1):
            print(f"\n{'='*60}")
            print(f"MCMC Iteration {iteration}/{self.max_iterations}")
            print(f"{'='*60}")
            
            # Get temperature for this iteration
            temperature = self.get_temperature(iteration)
            print(f"Temperature: {temperature:.3f}")
            
            # Sample proposal
            proposed_params = self.param_space.sample_proposal(current_params, temperature)
            
            # Evaluate proposal
            proposed_reward, proposed_metrics = self.evaluate_hyperparameters(proposed_params)
            
            # Accept or reject
            accept = self.accept_proposal(current_reward, proposed_reward, temperature)
            
            if accept:
                current_params = proposed_params.copy()
                current_reward = proposed_reward
                current_metrics = proposed_metrics
                print(f"✓ ACCEPTED: New reward = {current_reward:.4f}")
                
                # Update best if needed
                if current_reward > self.best_reward:
                    self.best_params = current_params.copy()
                    self.best_reward = current_reward
                    print(f"🎉 NEW BEST: {self.best_reward:.4f}")
                    self._save_best_params()
            else:
                print(f"✗ REJECTED: Kept reward = {current_reward:.4f}")
            
            # Record state
            self.chain_history.append(current_params.copy())
            self.reward_history.append(current_reward)
            self.acceptance_history.append(accept)
            
            # Adapt proposals periodically
            self.adapt_proposals(iteration)
            
            # Save chain periodically
            if iteration % 10 == 0:
                self._save_chain()
            
            # Print progress
            recent_rewards = self.reward_history[-10:]
            print(f"Current: {current_reward:.4f} | Best: {self.best_reward:.4f} | "
                  f"Recent avg: {np.mean(recent_rewards):.4f}")
        
        # Final results
        self._save_final_results()
        return self._get_optimization_results()
    
    def _save_best_params(self):
        """Save best parameters to file"""
        best_config = {
            'best_params': self._convert_to_serializable(self.best_params),
            'best_reward': float(self.best_reward),
            'timestamp': datetime.now().isoformat(),
            'iteration': int(len(self.chain_history))
        }
        
        with open(self.results_file, 'w') as f:
            json.dump(best_config, f, indent=2)
        
        print(f"Best parameters saved to {self.results_file}")
    
    def _save_chain(self):
        """Save MCMC chain to file"""
        chain_data = {
            'chain_history': self.chain_history,
            'reward_history': self.reward_history,
            'acceptance_history': self.acceptance_history,
            'best_params': self.best_params,
            'best_reward': self.best_reward
        }
        
        with open(self.chain_file, 'wb') as f:
            pickle.dump(chain_data, f)
    
    def _save_final_results(self):
        """Save comprehensive final results"""
        results = {
            'optimization_summary': {
                'total_iterations': int(len(self.chain_history) - 1),
                'burn_in_iterations': int(self.burn_in),
                'final_acceptance_rate': float(np.mean(self.acceptance_history[-20:]) if len(self.acceptance_history) >= 20 else np.mean(self.acceptance_history)),
                'best_reward': float(self.best_reward),
                'improvement': float(self.best_reward - self.reward_history[0])
            },
            'best_hyperparameters': self._convert_to_serializable(self.best_params),
            'parameter_statistics': self._compute_parameter_statistics(),
            'convergence_diagnostics': self._compute_convergence_diagnostics(),
            'timestamp': datetime.now().isoformat()
        }
        
        final_file = f"{basepath}/mcmc_optimization_results.json"
        with open(final_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Final results saved to {final_file}")
    
    def _convert_to_serializable(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    def _compute_parameter_statistics(self) -> Dict:
        """Compute statistics for each parameter across the chain"""
        if len(self.chain_history) <= self.burn_in:
            return {}
        
        # Use post-burn-in samples
        post_burnin = self.chain_history[self.burn_in:]
        
        stats = {}
        for param_name in self.param_space.param_names:
            values = [params[param_name] for params in post_burnin]
            stats[param_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'best_value': self._convert_to_serializable(self.best_params[param_name])
            }
        
        return stats
    
    def _compute_convergence_diagnostics(self) -> Dict:
        """Compute convergence diagnostics"""
        if len(self.reward_history) < 20:
            return {'status': 'insufficient_data'}
        
        # Compute running mean
        window_size = min(20, len(self.reward_history) // 4)
        running_mean = pd.Series(self.reward_history).rolling(window=window_size).mean()
        
        # Compute trend in recent iterations
        recent_rewards = self.reward_history[-20:]
        trend_slope = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]
        
        # Estimate effective sample size (simplified)
        post_burnin_rewards = self.reward_history[self.burn_in:]
        autocorr = np.corrcoef(post_burnin_rewards[:-1], post_burnin_rewards[1:])[0, 1] if len(post_burnin_rewards) > 1 else 0
        eff_sample_size = len(post_burnin_rewards) / (1 + 2 * max(0, autocorr))
        
        return {
            'trend_slope': float(trend_slope),
            'recent_volatility': float(np.std(recent_rewards)),
            'autocorrelation': float(autocorr),
            'effective_sample_size': float(eff_sample_size),
            'converged': bool(abs(trend_slope) < 0.001 and len(self.reward_history) > self.burn_in + 20)
        }
    
    def _get_optimization_results(self) -> Dict:
        """Get final optimization results"""
        return {
            'best_params': self.best_params,
            'best_reward': self.best_reward,
            'total_iterations': len(self.chain_history) - 1,
            'improvement': self.best_reward - self.reward_history[0],
            'final_acceptance_rate': np.mean(self.acceptance_history[-10:]) if len(self.acceptance_history) >= 10 else 0,
            'chain_history': self.chain_history,
            'reward_history': self.reward_history
        }
    
    def load_previous_chain(self) -> bool:
        """Load previous MCMC chain if available"""
        try:
            with open(self.chain_file, 'rb') as f:
                chain_data = pickle.load(f)
            
            self.chain_history = chain_data['chain_history']
            self.reward_history = chain_data['reward_history']
            self.acceptance_history = chain_data['acceptance_history']
            self.best_params = chain_data['best_params']
            self.best_reward = chain_data['best_reward']
            
            print(f"Loaded previous chain with {len(self.chain_history)} iterations")
            print(f"Previous best reward: {self.best_reward:.4f}")
            return True
            
        except FileNotFoundError:
            print("No previous chain found. Starting fresh.")
            return False
        except Exception as e:
            print(f"Error loading previous chain: {e}")
            return False


def run_hyperparameter_optimization(max_iterations: int = 15, 
                                   burn_in: int = 5,
                                   resume: bool = True) -> Dict:
    """
    Main function to run hyperparameter optimization
    
    Args:
        max_iterations: Maximum number of MCMC iterations
        burn_in: Number of burn-in iterations
        resume: Whether to resume from previous chain
    
    Returns:
        Dictionary with optimization results
    """
    print("="*80)
    print("HYPERPARAMETER OPTIMIZATION WITH MCMC")
    print("="*80)
    
    # Initialize optimizer
    optimizer = MCMCOptimizer(
        max_iterations=max_iterations,
        burn_in=burn_in,
        temperature_schedule='exponential'
    )
    
    # Load previous chain if resuming
    if resume:
        optimizer.load_previous_chain()
    
    # Run optimization
    results = optimizer.run_optimization()
    
    # Print final summary
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETED")
    print("="*80)
    print(f"Best reward achieved: {results['best_reward']:.4f}")
    print(f"Improvement over initial: {results['improvement']:.4f}")
    print(f"Total iterations: {results['total_iterations']}")
    print(f"Final acceptance rate: {results['final_acceptance_rate']:.3f}")
    
    print("\nBest hyperparameters:")
    for name, value in results['best_params'].items():
        print(f"  {name}: {value}")
    
    return results


if __name__ == "__main__":
    # Run hyperparameter optimization
    results = run_hyperparameter_optimization(max_iterations=15, burn_in=5)
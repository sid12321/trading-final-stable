#!/usr/bin/env python3
"""
model_trainer.py - Modular model training component

This module handles all model training functionality including:
1. Data preprocessing and model training
2. Model evaluation and validation
3. Posterior generation and plotting
"""

import os
import gc
import joblib
import pandas as pd
import numpy as np
import traceback
import time
from datetime import timedelta
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from StockTradingEnv2 import StockTradingEnv2
try:
    from StockTradingEnvOptimized import StockTradingEnvOptimized
    OPTIMIZED_ENV_AVAILABLE = True
except ImportError:
    OPTIMIZED_ENV_AVAILABLE = False
    print("Optimized environment not available, using standard environment")

# Import parameters and utilities
basepath = '/Users/skumar81/Desktop/Personal/trading-final-stable'
os.chdir(basepath)

from parameters import *
from lib import *
from common import *
import kitelogin

class ModelTrainer:
    """Modular model training class"""
    
    def __init__(self, symbols=None):
        self.symbols = symbols or TESTSYMBOLS
        self.rdflistp = {}
        self.lol = {}
        self.minparams = {}
        self.qtnorm = {}
        self.symposterior = {}
        self.timings = {}  # Store timing information
        
    def _time_section(self, section_name):
        """Context manager for timing code sections"""
        class Timer:
            def __init__(self, trainer, name):
                self.trainer = trainer
                self.name = name
                self.start_time = None
                
            def __enter__(self):
                self.start_time = time.time()
                return self
                
            def __exit__(self, *args):
                elapsed = time.time() - self.start_time
                self.trainer.timings[self.name] = elapsed
                print(f"[TIMING] {self.name}: {timedelta(seconds=elapsed)}")
                
        return Timer(self, section_name)
        
    def preprocess_data(self):
        """Preprocess and prepare training data"""
        with self._time_section("Data Preprocessing"):
            print("Preprocessing data")
            
            if not PREPROCESS:
                print("Preprocessing disabled in parameters")
                return None, None
            
            with self._time_section("Kite Login"):
                kite = kitelogin.login_to_kite()
            
            if kite:
                with self._time_section("Preprocess Execution"):
                    return preprocess(kite)
            else:
                print("Failed to login to Kite")
                return None, None

    def load_historical_data(self):
        """Load historical training data"""
        with self._time_section("Load Historical Data"):
            print("Reading historical data")
            
            for SYM in self.symbols:
                for prefix in ["final"]:
                    with self._time_section(f"Load {SYM} {prefix} data"):
                        print(f"{SYM} {prefix}")
                        rdf = pd.read_csv(f"{basepath}/traindata/{prefix}mldf{SYM}.csv")
                        rdf = rdf.drop(['t'], axis=1)
                        rdf = rdf.head(len(rdf) - 1)
                        self.rdflistp[SYM+prefix] = rdf

    def extract_signals(self):
        """Extract and validate trading signals"""
        with self._time_section("Extract Signals"):
            print("Reading strong signals")
            
            for SYM in self.symbols:
                for prefix in ["final"]:
                    with self._time_section(f"Extract signals for {SYM}"):
                        print(f'SYM: {SYM} prefix: {prefix}')
                        df = self.rdflistp[SYM+prefix]
                        if "Unnamed: 0" in df.columns:
                            df = df.drop(["Unnamed: 0"], axis=1)
                        df['currentt'] = pd.to_datetime(df['currentt'])
                        df['currentdate'] = df['currentt'].dt.date
                        finalsignalsp = df.columns[~df.columns.isin(['currentt', 'currento', 'currentdate', 'vwap2'])].tolist()
                        self.lol[SYM] = finalsignalsp
                        print(self.lol[SYM])
                        print(len(self.lol[SYM]))

            with self._time_section("Calculate Global Signals"):
                lolist = [self.lol[k] for k in self.lol]
                globalsignals = list(set.intersection(*[set(list) for list in lolist]))
                print("Global signals")
                print(globalsignals)
                return globalsignals

    def train_models_with_params(self, hyperparams=None, progress_tracker=None):
        """Train models with given hyperparameters"""
        with self._time_section("Train Models"):
            if hyperparams:
                # Update global parameters with hyperparameters
                globals().update(hyperparams)
            
            print("Training models with GPU optimization enabled")
            
            # Enable GPU optimizations if available
            if torch.cuda.is_available():
                # Optimize CUDA settings for maximum GPU utilization
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                if hasattr(torch.backends.cudnn, 'allow_tf32'):
                    torch.backends.cudnn.allow_tf32 = True
                
                # Set GPU memory management
                if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                    torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
                
                print(f"GPU optimization enabled: {torch.cuda.get_device_name(0)}")
                print(f"GPU memory available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            
            # modeltrain doesn't return a value, so we need to extract the reward differently
            for it in range(TRAINREPS):
                with self._time_section(f"Training iteration {it}"):
                    print(f"Training models - iteration: {it} (GPU optimized)")
                    DELMODELFLAG = True if it == 0 else False
                    
                    # Force GPU utilization by ensuring data is on GPU
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()  # Ensure previous operations complete
                    
                    modeltrain(self.rdflistp, NEWMODELFLAG, self.symbols, 
                              DELETEMODELS=DELMODELFLAG, SAVE_BEST_MODEL=False, lol=self.lol, 
                              progress_tracker=progress_tracker)
                    
                    # Memory cleanup between iterations
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    gc.collect()
            
            # Extract the reward from the metrics file since modeltrain doesn't return it
            with self._time_section("Extract Reward Metrics"):
                reward = self.extract_reward_from_metrics()
            
            # Final cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            
            return reward

    def extract_reward_from_metrics(self):
        """Extract the best reward from the metrics file"""
        try:
            # First try the standard metrics file
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
                    print(f"Extracted best reward from training: {best_reward}")
                    return best_reward
            
            # Try evaluation callback logs
            eval_reward = self.extract_reward_from_eval_callback()
            if eval_reward > 0:
                return eval_reward
                
            print("No metrics found, trying direct model evaluation")
            return self.extract_reward_from_evaluation()
                
        except Exception as e:
            print(f"Error extracting reward from metrics: {e}")
            return self.extract_reward_from_evaluation()

    def extract_reward_from_eval_callback(self):
        """Extract reward from evaluation callback results"""
        try:
            # Check for evaluation results saved by EvalCallback
            eval_file = f"{basepath}/tmp/sb3_log/evaluations.npz"
            if os.path.exists(eval_file):
                eval_data = np.load(eval_file)
                
                if 'results' in eval_data:
                    results = eval_data['results']
                    if len(results) > 0:
                        # Get the mean of the last evaluation
                        last_eval = results[-1]
                        mean_reward = np.mean(last_eval)
                        print(f"Extracted reward from evaluation callback: {mean_reward}")
                        return float(mean_reward)
            
            return 0.0
            
        except Exception as e:
            print(f"Error extracting from eval callback: {e}")
            return 0.0

    def extract_reward_from_evaluation(self):
        """Extract reward by evaluating the trained model on test environment"""
        try:
            with self._time_section("Model Evaluation for Reward"):
                print("Attempting to extract reward through model evaluation...")
            
            # Import required modules - already imported at top
            
            # Get the symbol we're training on (assuming symbols has at least one)
            if not self.symbols:
                print("No test symbols available for evaluation")
                return 0.001
                
            SYM = self.symbols[0]
            prefix = "final"
            
            # Check if model exists
            model_path = f"{basepath}/models/{SYM}localmodel.zip"
            if not os.path.exists(model_path):
                print(f"No trained model found at {model_path}")
                return 0.001
            
            # Load training data for evaluation environment
            if SYM + prefix not in self.rdflistp:
                print(f"No training data available for {SYM}")
                return 0.001
                
            df = self.rdflistp[SYM + prefix]
            
            # Use last 20% of data for evaluation (or minimum 50 rows)
            eval_size = max(50, int(len(df) * 0.2))
            eval_df = df.iloc[-eval_size:].reset_index(drop=True)
            
            # Get signal list
            if SYM not in self.lol:
                print(f"No signals available for {SYM}")
                return 0.001
                
            finalsignalsp = self.lol[SYM]
            
            # Create evaluation environment - already imported at top
            
            def make_eval_env():
                env_class = StockTradingEnvOptimized if OPTIMIZED_ENV_AVAILABLE else StockTradingEnv2
                return env_class(
                    eval_df, 
                    NLAGS, 
                    len(finalsignalsp), 
                    MAXIMUM_SHORT_VALUE,
                    finalsignalsp=finalsignalsp
                )
            
            eval_env = DummyVecEnv([make_eval_env])
            
            # Load normalization if available
            norm_path = f"{basepath}/models/{SYM}localmodel_vecnormalize.pkl"
            if os.path.exists(norm_path):
                eval_env = VecNormalize.load(norm_path, eval_env)
                eval_env.training = False
                eval_env.norm_reward = False
            else:
                eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True, training=False)
            
            # Load the trained model
            model = PPO.load(model_path, env=eval_env)
            
            # Evaluate the model
            print(f"Evaluating model on {eval_size} data points...")
            with self._time_section(f"Evaluate {SYM} model"):
                mean_reward, std_reward = evaluate_policy(
                    model, 
                    eval_env, 
                    n_eval_episodes=3,  # Multiple episodes for better estimate
                    deterministic=DETERMINISTIC,
                    render=False
                )
            
            print(f"Evaluation complete - Mean reward: {mean_reward:.4f} ± {std_reward:.4f}")
            
            # Clean up
            eval_env.close()
            
            # Ensure mean_reward is a scalar (not a list)
            if isinstance(mean_reward, (list, np.ndarray)):
                mean_reward = np.mean(mean_reward)
            return float(mean_reward)
            
        except Exception as e:
            print(f"Error in model evaluation: {e}")
            traceback.print_exc()
            return 0.001

    def load_trained_models(self):
        """Load all trained models and normalizations"""
        with self._time_section("Load Trained Models"):
            print("Loading all models and data normalizations")
            
            with self._time_section("Load All Models"):
                allmodels = loadallmodels(self.symbols)
            
            for SYM in self.symbols:
                with self._time_section(f"Load normalization for {SYM}"):
                    self.minparams[SYM] = len(self.lol[SYM])
                    try:
                        self.qtnorm[SYM] = joblib.load(f'{basepath}/models/{SYM}qt.joblib')
                        print(f"Loaded normalization for {SYM}")
                    except:
                        print("Normalization doesn't exist, model may not be available")
            
            # Clean up memory after loading models
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return allmodels

    def generate_posterior_analysis(self):
        """Generate posterior distributions and plots"""
        if not (TRAINMODEL and GENERATEPOSTERIOR):
            return
        
        with self._time_section("Generate Posterior Analysis"):
            print("Generating posterior distributions for the returns")
            with self._time_section("Generate Posterior Distributions"):
                df_test_actions_list = generateposterior(self.rdflistp, self.qtnorm, self.symbols, lol=self.lol)
            
            if POSTERIORPLOTS:
                with self._time_section("Generate Posterior Plots"):
                    print("Plotting posterior trading plots")
                    for i in range(100):
                        print(f"Index {i}")
                        try:
                            with self._time_section(f"Plot index {i}"):
                                self.symposterior = posteriorplots(df_test_actions_list, self.symbols, i)
                        except:
                            break
            
            return df_test_actions_list

    def run_full_training_pipeline(self):
        """Run the complete training pipeline"""
        pipeline_start = time.time()
        print("Starting full training pipeline...")
        print("=" * 60)
        
        # Clean TensorBoard logs before starting
        from common import clean_tensorboard_logs
        clean_tensorboard_logs()
        
        # Initialize multi-symbol progress tracker
        from progress_tracker import MultiSymbolProgressTracker
        progress_tracker = MultiSymbolProgressTracker(self.symbols)
        progress_tracker.start_training()
        
        # Step 1: Preprocess data if needed
        if PREPROCESS:
            self.preprocess_data()
        
        # Step 2: Load historical data
        self.load_historical_data()
        
        # Step 3: Extract signals
        globalsignals = self.extract_signals()
        
        # Step 4: Train models
        if TRAINMODEL:
            reward = self.train_models_with_params(progress_tracker=progress_tracker)
            print(f"Training completed with reward: {reward}")
        
        # Step 5: Load trained models
        allmodels = self.load_trained_models()
        
        # Step 6: Generate posterior analysis
        self.generate_posterior_analysis()
        
        # Calculate total time
        total_time = time.time() - pipeline_start
        self.timings['Total Pipeline'] = total_time
        
        # Finish progress tracking
        progress_tracker.finish_training()
        
        # Print timing summary
        self._print_timing_summary()
        
        print("\nTraining pipeline completed successfully!")
        return allmodels, globalsignals, self.lol, self.qtnorm
    
    def _print_timing_summary(self):
        """Print a summary of all timing measurements"""
        print("\n" + "=" * 60)
        print("EXECUTION TIME SUMMARY")
        print("=" * 60)
        
        # Sort timings by duration (longest first)
        sorted_timings = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)
        
        # Find the longest name for formatting
        max_name_len = max(len(name) for name, _ in sorted_timings) if sorted_timings else 0
        
        # Print each timing
        for name, duration in sorted_timings:
            # Skip sub-timings in the main summary
            if any(parent in name for parent in ['Load ', 'Extract signals for ', 'Training iteration ', 'Plot index ']):
                continue
            print(f"{name:<{max_name_len + 2}} : {timedelta(seconds=duration)}")
        
        # Print total time specially
        if 'Total Pipeline' in self.timings:
            print("=" * 60)
            print(f"{'TOTAL TIME':<{max_name_len + 2}} : {timedelta(seconds=self.timings['Total Pipeline'])}")
        
        # Print detailed breakdown if there are many timings
        detailed_timings = [t for t in sorted_timings if any(parent in t[0] for parent in ['Load ', 'Extract signals for ', 'Training iteration ', 'Plot index '])]
        if detailed_timings:
            print("\n" + "-" * 60)
            print("DETAILED BREAKDOWN")
            print("-" * 60)
            for name, duration in detailed_timings:
                print(f"  {name:<{max_name_len}} : {timedelta(seconds=duration)}")

if __name__ == "__main__":
    # Run the full training pipeline
    trainer = ModelTrainer()
    try:
        trainer.run_full_training_pipeline()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        trainer._print_timing_summary()
    except Exception as e:
        print(f"\n\nError during training: {e}")
        traceback.print_exc()
        trainer._print_timing_summary()
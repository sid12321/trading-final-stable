#!/usr/bin/env python3
"""
Progress tracking utilities for trading system training
"""

import time
import os
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure


class TrainingProgressCallback(BaseCallback):
    """
    Custom callback for tracking training progress with detailed metrics
    """
    
    def __init__(self, total_timesteps, symbol="Unknown", check_freq=1000, verbose=0):
        super(TrainingProgressCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.symbol = symbol
        self.check_freq = check_freq
        self.start_time = None
        self.pbar = None
        self.episode_rewards = []
        self.episode_lengths = []
        self.last_update_time = None
        self.best_mean_reward = -float('inf')
        self.episodes_completed = 0
        
    def _on_training_start(self) -> None:
        """
        Initialize progress tracking when training starts
        """
        self.start_time = time.time()
        self.last_update_time = self.start_time
        
        # Create progress bar
        self.pbar = tqdm(
            total=self.total_timesteps,
            desc=f"Training {self.symbol}",
            unit="steps",
            leave=True,
            ncols=120,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        )
        
        print(f"\n🚀 Starting training for {self.symbol}")
        print(f"📊 Total timesteps: {self.total_timesteps:,}")
        print(f"🔄 Progress updates every {self.check_freq:,} steps")
        print("=" * 80)
        
    def _on_step(self) -> bool:
        """
        Update progress on each step
        """
        # Update progress bar to match actual timesteps completed
        # self.num_timesteps already includes all environments
        if self.pbar is not None:
            # Set progress bar position to actual timesteps
            self.pbar.n = min(self.num_timesteps, self.total_timesteps)
            self.pbar.refresh()
            
        # Update progress bar postfix with current metrics
        if self.num_timesteps % self.check_freq == 0:
            self._update_progress_info()
            
        return True
    
    def _update_progress_info(self):
        """
        Update progress bar with detailed training metrics
        """
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        steps_per_sec = self.num_timesteps / elapsed_time if elapsed_time > 0 else 0
        
        # Calculate ETA
        remaining_steps = self.total_timesteps - self.num_timesteps
        eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
        eta_str = str(timedelta(seconds=int(eta_seconds)))
        
        # Get recent metrics from logger if available
        postfix_dict = {
            'SPS': f"{steps_per_sec:.0f}",
            'Episodes': self.episodes_completed,
            'ETA': eta_str
        }
        
        # Add reward info if available
        if len(self.episode_rewards) > 0:
            recent_rewards = self.episode_rewards[-10:]  # Last 10 episodes
            mean_reward = np.mean(recent_rewards)
            postfix_dict['Reward'] = f"{mean_reward:.2f}"
            
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                postfix_dict['Best'] = f"{self.best_mean_reward:.2f}"
        
        self.pbar.set_postfix(postfix_dict)
        
        # Print detailed progress every 50k steps
        if self.num_timesteps % (self.check_freq * 50) == 0:
            self._print_detailed_progress()
    
    def _print_detailed_progress(self):
        """
        Print detailed progress information
        """
        elapsed_time = time.time() - self.start_time
        progress_pct = (self.num_timesteps / self.total_timesteps) * 100
        
        print(f"\n📈 Progress Update - {self.symbol}")
        print(f"   Steps: {self.num_timesteps:,} / {self.total_timesteps:,} ({progress_pct:.1f}%)")
        print(f"   Time Elapsed: {str(timedelta(seconds=int(elapsed_time)))}")
        print(f"   Episodes Completed: {self.episodes_completed}")
        
        if len(self.episode_rewards) > 0:
            recent_rewards = self.episode_rewards[-10:]
            print(f"   Recent Mean Reward: {np.mean(recent_rewards):.3f}")
            print(f"   Best Mean Reward: {self.best_mean_reward:.3f}")
            print(f"   Recent Episode Length: {np.mean(self.episode_lengths[-10:]):.1f}")
        
        print("=" * 60)
    
    def _on_rollout_end(self) -> None:
        """
        Called at the end of each rollout
        """
        # Extract episode info if available
        if 'episode' in self.locals:
            episode_info = self.locals['episode']
            if len(episode_info) > 0:
                for ep in episode_info:
                    if 'r' in ep and 'l' in ep:
                        self.episode_rewards.append(ep['r'])
                        self.episode_lengths.append(ep['l'])
                        self.episodes_completed += 1
    
    def _on_training_end(self) -> None:
        """
        Clean up when training ends
        """
        if self.pbar is not None:
            self.pbar.close()
        
        total_time = time.time() - self.start_time
        print(f"\n✅ Training completed for {self.symbol}!")
        print(f"📊 Final Statistics:")
        print(f"   Total Time: {str(timedelta(seconds=int(total_time)))}")
        print(f"   Total Steps: {self.num_timesteps:,}")
        print(f"   Total Episodes: {self.episodes_completed}")
        print(f"   Average SPS: {self.num_timesteps / total_time:.0f}")
        
        if len(self.episode_rewards) > 0:
            print(f"   Final Mean Reward: {np.mean(self.episode_rewards[-10:]):.3f}")
            print(f"   Best Mean Reward: {self.best_mean_reward:.3f}")
        
        print("=" * 80)


class MultiSymbolProgressTracker:
    """
    Track progress across multiple symbols in the training pipeline
    """
    
    def __init__(self, symbols):
        self.symbols = symbols
        self.total_symbols = len(symbols)
        self.current_symbol_index = 0
        self.start_time = None
        self.symbol_times = {}
        
    def start_training(self):
        """Start the overall training process"""
        self.start_time = time.time()
        print(f"\n🎯 Starting Multi-Symbol Training Pipeline")
        print(f"📈 Training {self.total_symbols} symbols: {', '.join(self.symbols)}")
        print(f"🕒 Started at: {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 80)
        
    def start_symbol(self, symbol):
        """Start training for a specific symbol"""
        self.current_symbol_index = next(i for i, s in enumerate(self.symbols) if s == symbol)
        symbol_start = time.time()
        self.symbol_times[symbol] = {'start': symbol_start}
        
        progress_pct = (self.current_symbol_index / self.total_symbols) * 100
        
        print(f"\n🔄 Starting Symbol {self.current_symbol_index + 1}/{self.total_symbols}: {symbol}")
        print(f"📊 Overall Progress: {progress_pct:.1f}%")
        
        if self.current_symbol_index > 0:
            elapsed = time.time() - self.start_time
            avg_time_per_symbol = elapsed / self.current_symbol_index
            remaining_symbols = self.total_symbols - self.current_symbol_index
            eta = remaining_symbols * avg_time_per_symbol
            print(f"🕒 Estimated Time Remaining: {str(timedelta(seconds=int(eta)))}")
        
        print("=" * 60)
        
    def finish_symbol(self, symbol):
        """Finish training for a specific symbol"""
        if symbol in self.symbol_times:
            self.symbol_times[symbol]['end'] = time.time()
            duration = self.symbol_times[symbol]['end'] - self.symbol_times[symbol]['start']
            
            print(f"\n✅ Completed {symbol} in {str(timedelta(seconds=int(duration)))}")
            
            # Calculate running average time per symbol
            completed_symbols = self.current_symbol_index + 1
            total_elapsed = time.time() - self.start_time
            avg_time = total_elapsed / completed_symbols
            
            remaining_symbols = self.total_symbols - completed_symbols
            eta = remaining_symbols * avg_time
            
            if remaining_symbols > 0:
                print(f"🕒 Estimated time for remaining {remaining_symbols} symbols: {str(timedelta(seconds=int(eta)))}")
            
            print("=" * 60)
    
    def finish_training(self):
        """Finish the overall training process"""
        total_time = time.time() - self.start_time
        
        print(f"\n🎉 Multi-Symbol Training Pipeline Completed!")
        print(f"📊 Final Summary:")
        print(f"   Total Symbols: {self.total_symbols}")
        print(f"   Total Time: {str(timedelta(seconds=int(total_time)))}")
        print(f"   Average Time per Symbol: {str(timedelta(seconds=int(total_time / self.total_symbols)))}")
        
        print(f"\n📈 Symbol-by-Symbol Breakdown:")
        for symbol in self.symbols:
            if symbol in self.symbol_times and 'end' in self.symbol_times[symbol]:
                duration = self.symbol_times[symbol]['end'] - self.symbol_times[symbol]['start']
                print(f"   {symbol}: {str(timedelta(seconds=int(duration)))}")
        
        print(f"\n🕒 Completed at: {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 80)


def create_progress_callback(total_timesteps, symbol="Unknown"):
    """
    Factory function to create a progress callback
    """
    return TrainingProgressCallback(
        total_timesteps=total_timesteps,
        symbol=symbol,
        check_freq=max(1000, total_timesteps // 100),  # Update 100 times during training
        verbose=1
    )
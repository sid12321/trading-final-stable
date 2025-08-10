"""
Scheduling utilities for learning rate, entropy coefficient, and target KL.

This module provides schedule functions that can be used with Stable Baselines3
to gradually adjust hyperparameters during training.
"""

import numpy as np
from typing import Callable


def linear_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    """
    Linear interpolation between initial and final values.
    
    Args:
        initial_value: Starting value
        final_value: Ending value
        
    Returns:
        Schedule function that takes progress remaining (1.0 to 0.0) and returns interpolated value
    """
    def func(progress_remaining: float) -> float:
        """
        Progress remaining goes from 1 (start) to 0 (end)
        """
        return final_value + (initial_value - final_value) * progress_remaining
    
    return func


def exponential_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    """
    Exponential decay from initial to final value.
    
    Args:
        initial_value: Starting value
        final_value: Ending value
        
    Returns:
        Schedule function with exponential decay
    """
    def func(progress_remaining: float) -> float:
        """
        Progress remaining goes from 1 (start) to 0 (end)
        """
        # Use exponential decay
        decay_rate = np.log(final_value / initial_value)
        return initial_value * np.exp(decay_rate * (1 - progress_remaining))
    
    return func


def cosine_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    """
    Cosine annealing schedule from initial to final value.
    
    Args:
        initial_value: Starting value
        final_value: Ending value
        
    Returns:
        Schedule function with cosine annealing
    """
    def func(progress_remaining: float) -> float:
        """
        Progress remaining goes from 1 (start) to 0 (end)
        """
        # Cosine annealing
        cosine_factor = 0.5 * (1 + np.cos(np.pi * (1 - progress_remaining)))
        return final_value + (initial_value - final_value) * cosine_factor
    
    return func


def get_schedule(schedule_type: str, initial_value: float, final_value: float) -> Callable[[float], float]:
    """
    Get a schedule function based on the specified type.
    
    Args:
        schedule_type: Type of schedule ("linear", "exponential", or "cosine")
        initial_value: Starting value
        final_value: Ending value
        
    Returns:
        Schedule function
    """
    schedule_map = {
        "linear": linear_schedule,
        "exponential": exponential_schedule,
        "cosine": cosine_schedule
    }
    
    if schedule_type not in schedule_map:
        print(f"Unknown schedule type: {schedule_type}. Using linear schedule.")
        schedule_type = "linear"
    
    return schedule_map[schedule_type](initial_value, final_value)


# Create schedule functions for entropy coefficient and target KL that can be called manually
def get_entropy_coef_at_progress(progress_remaining: float, initial_ent: float, final_ent: float) -> float:
    """
    Get entropy coefficient value at given progress.
    
    Args:
        progress_remaining: Training progress remaining (1.0 to 0.0)
        initial_ent: Initial entropy coefficient
        final_ent: Final entropy coefficient
        
    Returns:
        Entropy coefficient value
    """
    return final_ent + (initial_ent - final_ent) * progress_remaining


def get_target_kl_at_progress(progress_remaining: float, initial_kl: float, final_kl: float) -> float:
    """
    Get target KL value at given progress.
    
    Args:
        progress_remaining: Training progress remaining (1.0 to 0.0)
        initial_kl: Initial target KL
        final_kl: Final target KL
        
    Returns:
        Target KL value
    """
    return final_kl + (initial_kl - final_kl) * progress_remaining
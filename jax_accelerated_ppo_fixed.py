"""
Fixed JAX-accelerated training components for PPO
This provides JAX acceleration for critical computational bottlenecks
"""

import warnings
warnings.filterwarnings('ignore')

# Configure JAX before importing
import os
import platform

# Detect platform and configure accordingly
is_mac = platform.system() == 'Darwin'

if is_mac:
    # Mac-specific JAX configuration for Metal
    os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # Metal support in JAX is experimental, use CPU for stability
    os.environ['JAX_ENABLE_X64'] = 'False'  # Keep False for performance
    os.environ['JAX_ENABLE_CHECKS'] = 'false'  # Disable runtime checks for performance
    os.environ['JAX_LOG_COMPILES'] = 'false'  # Reduce logging overhead
    # Note: JAX Metal support is still experimental. Using CPU backend for stability.
else:
    # Linux/Windows configuration
    os.environ['JAX_ENABLE_X64'] = 'False'  # Keep False for performance
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # Don't preallocate GPU memory
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.10'  # Use only 10% of GPU memory for JAX
    os.environ['JAX_ENABLE_CHECKS'] = 'false'  # Disable runtime checks for performance
    os.environ['JAX_LOG_COMPILES'] = 'false'  # Reduce logging overhead
    os.environ['JAX_PLATFORMS'] = 'cpu'  # Force CPU to avoid GPU memory conflicts with PyTorch

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    import numpy as np
    
    # Initialize JAX - let it auto-detect GPU
    # jax.config.update('jax_platform_name', 'cpu')  # Removed to allow GPU
    jax.config.update('jax_enable_x64', False)  # Keep False for performance
    
    # Test JAX works
    test_array = jnp.array([1, 2, 3])
    _ = jnp.sum(test_array)
    
    JAX_AVAILABLE = True
    if os.environ.get('JAX_INIT_PRINTED') != '1':
        # Check which backend JAX is using
        backend = jax.default_backend()
        print(f"JAX acceleration enabled (backend: {backend})")
        os.environ['JAX_INIT_PRINTED'] = '1'
    
except Exception as e:
    # If GPU fails, try CPU as fallback
    try:
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
        import jax
        import jax.numpy as jnp
        from jax import jit, vmap
        
        # Test JAX works on CPU
        test_array = jnp.array([1, 2, 3])
        _ = jnp.sum(test_array)
        
        JAX_AVAILABLE = True
        if os.environ.get('JAX_INIT_PRINTED') != '1':
            print(f"JAX acceleration enabled (backend: cpu, GPU failed with {type(e).__name__})")
            os.environ['JAX_INIT_PRINTED'] = '1'
    except:
        JAX_AVAILABLE = False
        jnp = np  # Fallback to numpy
        if os.environ.get('JAX_INIT_PRINTED') != '1':
            print(f"JAX disabled due to error: {type(e).__name__}")
            os.environ['JAX_INIT_PRINTED'] = '1'

# Define functions that work with either JAX or NumPy
def compute_advantages_jax(values, rewards, dones, gamma=0.99, gae_lambda=0.95):
    """Compute advantages using GAE (JAX or NumPy)"""
    if JAX_AVAILABLE and len(values) > 50:  # Use JAX for smaller computations to maximize GPU usage
        try:
            # Convert to JAX arrays
            values_jax = jnp.array(values)
            rewards_jax = jnp.array(rewards)
            dones_jax = jnp.array(dones)
            
            # Simple advantage computation (more stable than scan)
            advantages = jnp.zeros_like(rewards_jax[:-1])
            last_advantage = 0.0
            
            # Manual loop (more stable than scan for this case)
            adv_list = []
            for t in reversed(range(len(rewards_jax) - 1)):
                delta = rewards_jax[t] + gamma * values_jax[t + 1] * (1 - dones_jax[t]) - values_jax[t]
                advantage = delta + gamma * gae_lambda * (1 - dones_jax[t]) * last_advantage
                adv_list.append(advantage)
                last_advantage = advantage
            
            advantages = jnp.array(list(reversed(adv_list)))
            return np.array(advantages)  # Convert back to numpy
            
        except Exception as e:
            print(f"JAX advantage computation failed, using NumPy: {e}")
            # Fall through to NumPy implementation
    
    # NumPy implementation (fallback or for small arrays)
    advantages = np.zeros(len(rewards) - 1)
    last_advantage = 0
    
    for t in reversed(range(len(rewards) - 1)):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        advantages[t] = delta + gamma * gae_lambda * (1 - dones[t]) * last_advantage
        last_advantage = advantages[t]
        
    return advantages

def compute_policy_loss_jax(log_probs, old_log_probs, advantages, clip_range=0.2):
    """Compute PPO policy loss (JAX or NumPy)"""
    if JAX_AVAILABLE and len(log_probs) > 32:  # Use JAX for most batch sizes to maximize GPU usage
        try:
            log_probs_jax = jnp.array(log_probs)
            old_log_probs_jax = jnp.array(old_log_probs)
            advantages_jax = jnp.array(advantages)
            
            ratio = jnp.exp(log_probs_jax - old_log_probs_jax)
            surr1 = ratio * advantages_jax
            surr2 = jnp.clip(ratio, 1 - clip_range, 1 + clip_range) * advantages_jax
            return float(-jnp.mean(jnp.minimum(surr1, surr2)))
        except Exception:
            pass  # Fall through to NumPy
    
    # NumPy fallback
    ratio = np.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = np.clip(ratio, 1 - clip_range, 1 + clip_range) * advantages
    return -np.mean(np.minimum(surr1, surr2))

def compute_value_loss_jax(values, returns, old_values, clip_range=0.2):
    """Compute clipped value loss (JAX or NumPy)"""
    if JAX_AVAILABLE and len(values) > 32:  # Use JAX for most batch sizes to maximize GPU usage
        try:
            values_jax = jnp.array(values)
            returns_jax = jnp.array(returns)
            old_values_jax = jnp.array(old_values)
            
            value_pred_clipped = old_values_jax + jnp.clip(values_jax - old_values_jax, -clip_range, clip_range)
            value_loss1 = (values_jax - returns_jax) ** 2
            value_loss2 = (value_pred_clipped - returns_jax) ** 2
            return float(0.5 * jnp.mean(jnp.maximum(value_loss1, value_loss2)))
        except Exception:
            pass  # Fall through to NumPy
    
    # NumPy fallback
    value_pred_clipped = old_values + np.clip(values - old_values, -clip_range, clip_range)
    value_loss1 = (values - returns) ** 2
    value_loss2 = (value_pred_clipped - returns) ** 2
    return 0.5 * np.mean(np.maximum(value_loss1, value_loss2))

def compute_explained_variance_jax(y_pred, y_true):
    """Compute explained variance (JAX or NumPy)"""
    if JAX_AVAILABLE and len(y_pred) > 100:  # Use JAX for most calculations to maximize GPU usage
        try:
            y_pred_jax = jnp.array(y_pred)
            y_true_jax = jnp.array(y_true)
            
            var_y = jnp.var(y_true_jax)
            exp_var = 1 - jnp.var(y_true_jax - y_pred_jax) / (var_y + 1e-8)
            return float(exp_var)
        except Exception:
            pass  # Fall through to NumPy
    
    # NumPy fallback
    var_y = np.var(y_true)
    return 1 - np.var(y_true - y_pred) / (var_y + 1e-8)

def normalize_observations_jax(obs, mean, std, epsilon=1e-8):
    """Normalize observations (JAX or NumPy)"""
    if JAX_AVAILABLE and obs.size > 50:  # Use JAX for most normalizations to maximize GPU usage
        try:
            obs_jax = jnp.array(obs)
            mean_jax = jnp.array(mean)
            std_jax = jnp.array(std)
            result = (obs_jax - mean_jax) / (std_jax + epsilon)
            return np.array(result)
        except Exception:
            pass  # Fall through to NumPy
    
    # NumPy fallback
    return (obs - mean) / (std + epsilon)

# Utility functions
def to_jax_if_available(arr):
    """Convert numpy array to JAX array if available and beneficial"""
    if JAX_AVAILABLE and isinstance(arr, np.ndarray) and arr.size > 100:
        try:
            return jnp.array(arr)
        except Exception:
            pass
    return arr

def to_numpy_if_jax(arr):
    """Convert JAX array to numpy array if needed"""
    if JAX_AVAILABLE and hasattr(arr, 'device'):
        try:
            return np.array(arr)
        except Exception:
            pass
    return arr

# Export functions
__all__ = [
    'JAX_AVAILABLE',
    'compute_advantages_jax',
    'compute_policy_loss_jax', 
    'compute_value_loss_jax',
    'normalize_observations_jax',
    'compute_explained_variance_jax',
    'to_jax_if_available',
    'to_numpy_if_jax'
]
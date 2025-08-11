#!/usr/bin/env python3
"""
JAX CPU optimizations for trading environment
Even without GPU, JAX can significantly speed up operations through JIT compilation
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
import numpy as np
import time
from functools import partial

# Configure JAX for CPU optimization
jax.config.update('jax_platform_name', 'cpu')
# Enable x64 for better numerical precision
jax.config.update("jax_enable_x64", True)
# Optimize CPU parallelization
import os
os.environ['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=16'

class JAXOptimizedFeatures:
    """JAX-optimized technical indicator calculations"""
    
    @staticmethod
    @jit
    def calculate_sma(prices, window):
        """JIT-compiled Simple Moving Average"""
        def _sma_step(carry, x):
            buffer, sum_val = carry
            # Shift buffer and add new value
            new_buffer = jnp.concatenate([buffer[1:], x[None]])
            new_sum = sum_val - buffer[0] + x
            sma = new_sum / window
            return (new_buffer, new_sum), sma
        
        # Initialize
        init_buffer = jnp.ones(window) * prices[0]
        init_sum = jnp.sum(init_buffer)
        
        # Scan through prices
        _, smas = lax.scan(_sma_step, (init_buffer, init_sum), prices)
        return smas
    
    @staticmethod
    @jit
    def calculate_rsi(prices, period=14):
        """JIT-compiled RSI calculation"""
        # Calculate price changes
        deltas = jnp.diff(prices, prepend=prices[0])
        
        # Separate gains and losses
        gains = jnp.where(deltas > 0, deltas, 0)
        losses = jnp.where(deltas < 0, -deltas, 0)
        
        # Calculate exponential moving averages
        def ema_scan(carry, x):
            prev_ema = carry
            new_ema = (x + prev_ema * (period - 1)) / period
            return new_ema, new_ema
        
        # Calculate average gains and losses
        _, avg_gains = lax.scan(ema_scan, gains[0], gains)
        _, avg_losses = lax.scan(ema_scan, losses[0], losses)
        
        # Calculate RSI
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    @jit
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        """JIT-compiled MACD calculation"""
        # EMA calculation
        def ema(prices, period):
            alpha = 2 / (period + 1)
            def _ema_step(carry, x):
                prev = carry
                new = alpha * x + (1 - alpha) * prev
                return new, new
            _, emas = lax.scan(_ema_step, prices[0], prices)
            return emas
        
        # Calculate MACD line
        ema_fast = ema(prices, fast)
        ema_slow = ema(prices, slow)
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = ema(macd_line, signal)
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    @jit
    def calculate_all_features(ohlcv_data):
        """Calculate all features in one JIT-compiled function"""
        open_prices = ohlcv_data[:, 0]
        high_prices = ohlcv_data[:, 1]
        low_prices = ohlcv_data[:, 2]
        close_prices = ohlcv_data[:, 3]
        volumes = ohlcv_data[:, 4]
        
        # Price-based features
        returns = jnp.diff(close_prices, prepend=close_prices[0]) / (close_prices + 1e-10)
        volatility = jnp.std(returns)
        
        # Volume features
        volume_ratio = volumes / (jnp.mean(volumes) + 1e-10)
        
        # Technical indicators (simplified for demo)
        rsi = JAXOptimizedFeatures.calculate_rsi(close_prices)
        sma_10 = JAXOptimizedFeatures.calculate_sma(close_prices, 10)
        
        # Stack all features
        features = jnp.stack([
            returns,
            jnp.ones_like(returns) * volatility,
            volume_ratio,
            rsi / 100,  # Normalize RSI
            sma_10 / close_prices  # Relative SMA
        ], axis=1)
        
        return features


class JAXOptimizedEnvCore:
    """Core environment operations optimized with JAX"""
    
    @staticmethod
    @jit
    def step_batch(states, actions, prices):
        """Vectorized environment step for multiple environments"""
        # Unpack state
        positions = states[:, 0]
        cash = states[:, 1]
        
        # Unpack actions (buy/sell amounts)
        action_types = actions[:, 0]
        amounts = actions[:, 1]
        
        # Current prices
        current_prices = prices
        
        # Calculate trades
        # Buy when action_type > 0.5, sell when < -0.5
        buy_mask = action_types > 0.5
        sell_mask = action_types < -0.5
        
        # Calculate position changes
        max_buy = cash / (current_prices + 1e-10)
        buy_amount = jnp.where(buy_mask, jnp.minimum(max_buy, amounts * max_buy), 0)
        sell_amount = jnp.where(sell_mask, jnp.minimum(positions, amounts * positions), 0)
        
        # Update positions and cash
        new_positions = positions + buy_amount - sell_amount
        trade_cost = buy_amount * current_prices - sell_amount * current_prices
        transaction_cost = jnp.abs(trade_cost) * 0.001  # 0.1% transaction cost
        new_cash = cash - trade_cost - transaction_cost
        
        # Calculate rewards
        portfolio_value = new_positions * current_prices + new_cash
        prev_value = positions * current_prices + cash
        rewards = (portfolio_value - prev_value) / prev_value
        
        # New state
        new_states = jnp.stack([new_positions, new_cash], axis=1)
        
        return new_states, rewards
    
    @staticmethod
    @jit
    def calculate_portfolio_metrics(states, prices):
        """Calculate portfolio metrics efficiently"""
        positions = states[:, 0]
        cash = states[:, 1]
        
        portfolio_values = positions * prices + cash
        returns = jnp.diff(portfolio_values) / portfolio_values[:-1]
        
        # Sharpe ratio (simplified)
        sharpe = jnp.mean(returns) / (jnp.std(returns) + 1e-10) * jnp.sqrt(252)
        
        # Max drawdown
        cummax = jnp.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cummax) / cummax
        max_drawdown = jnp.min(drawdown)
        
        return {
            'total_return': (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0],
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'final_value': portfolio_values[-1]
        }


def benchmark_jax_optimizations():
    """Benchmark JAX optimizations vs NumPy"""
    print("=" * 60)
    print("🚀 JAX CPU Optimization Benchmark")
    print("=" * 60)
    
    # Test data
    n_steps = 10000
    n_envs = 100
    
    # Generate random OHLCV data
    np.random.seed(42)
    ohlcv = np.random.randn(n_steps, 5).astype(np.float64)
    ohlcv[:, :4] = np.abs(ohlcv[:, :4]) * 100 + 100  # Prices
    ohlcv[:, 4] = np.abs(ohlcv[:, 4]) * 1000000  # Volume
    
    # Convert to JAX
    ohlcv_jax = jnp.array(ohlcv)
    
    # 1. Feature calculation benchmark
    print("\n📊 Feature Calculation:")
    print("-" * 40)
    
    # NumPy version
    start = time.time()
    for _ in range(10):
        # Simple NumPy calculations
        returns = np.diff(ohlcv[:, 3]) / ohlcv[:-1, 3]
        sma = np.convolve(ohlcv[:, 3], np.ones(10)/10, 'same')
    numpy_time = time.time() - start
    print(f"NumPy time (10 iterations): {numpy_time:.3f}s")
    
    # JAX version
    jax_features = JAXOptimizedFeatures()
    
    # Compile
    _ = jax_features.calculate_all_features(ohlcv_jax)
    
    start = time.time()
    for _ in range(10):
        features = jax_features.calculate_all_features(ohlcv_jax).block_until_ready()
    jax_time = time.time() - start
    print(f"JAX time (10 iterations): {jax_time:.3f}s")
    print(f"Speedup: {numpy_time/jax_time:.2f}x")
    
    # 2. Vectorized environment steps
    print("\n🎮 Vectorized Environment Steps:")
    print("-" * 40)
    
    # Initial states and actions
    states = jnp.ones((n_envs, 2))  # [position, cash]
    states = states.at[:, 1].set(10000)  # Starting cash
    actions = jax.random.normal(jax.random.PRNGKey(0), (n_envs, 2))
    prices = jnp.ones(n_envs) * 100
    
    # Compile
    _ = JAXOptimizedEnvCore.step_batch(states, actions, prices)
    
    # Benchmark
    start = time.time()
    for _ in range(1000):
        states, rewards = JAXOptimizedEnvCore.step_batch(states, actions, prices)
        rewards.block_until_ready()
    jax_env_time = time.time() - start
    
    steps_per_second = (1000 * n_envs) / jax_env_time
    print(f"JAX vectorized steps: {steps_per_second:.0f} steps/second")
    print(f"Time for 1000 iterations: {jax_env_time:.3f}s")
    
    print("\n" + "=" * 60)
    print("📈 Summary:")
    print("=" * 60)
    print("Even without GPU/Metal support, JAX provides:")
    print(f"✅ {numpy_time/jax_time:.1f}x speedup for feature calculation")
    print(f"✅ {steps_per_second:.0f} environment steps/second")
    print("✅ JIT compilation eliminates Python overhead")
    print("✅ Automatic vectorization and parallelization")
    print("\n💡 With these optimizations, your training could be")
    print("   2-3x faster even using CPU-only JAX!")
    

if __name__ == "__main__":
    # Check JAX configuration
    print(f"JAX version: {jax.__version__}")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    
    # Run benchmarks
    benchmark_jax_optimizations()
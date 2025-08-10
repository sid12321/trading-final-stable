"""
Custom PPO implementation with bounded entropy loss

This module provides a PPO implementation that restricts entropy loss 
to an absolute magnitude of 1, preventing extreme entropy values that
could destabilize training.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo.ppo import PPO as BasePPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import explained_variance
from gymnasium import spaces
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union
import warnings

# Import JAX acceleration if available
try:
    from jax_accelerated_ppo_fixed import (
        JAX_AVAILABLE, compute_advantages_jax, compute_policy_loss_jax,
        compute_value_loss_jax, compute_explained_variance_jax,
        to_jax_if_available, to_numpy_if_jax
    )
    if JAX_AVAILABLE:
        if os.environ.get('JAX_PPO_INIT_PRINTED') != '1':
            print("JAX acceleration enabled for PPO computations")
            os.environ['JAX_PPO_INIT_PRINTED'] = '1'
except (ImportError, Exception) as e:
    JAX_AVAILABLE = False
    print(f"JAX acceleration disabled due to error: {type(e).__name__}")
from parameters import (
    GLOBALLEARNINGRATE, N_STEPS, BATCH_SIZE, N_EPOCHS, GAMMA, GAE_LAMBDA, CLIP_RANGE, CLIP_RANGE_VF, NORAD, ENT_COEF, VF_COEF, MAX_GRAD_NORM, USE_SDE, SDE_SAMPLE_FREQ, TARGET_KL, STATS_WINDOW_SIZE, ENTROPY_BOUND, VALUE_LOSS_BOUND, VERBOSITY, USE_MIXED_PRECISION, DEVICE,
    USE_LR_SCHEDULE, INITIAL_LR, FINAL_LR, LR_SCHEDULE_TYPE,
    USE_ENT_SCHEDULE, INITIAL_ENT_COEF, FINAL_ENT_COEF,
    USE_TARGET_KL_SCHEDULE, INITIAL_TARGET_KL, FINAL_TARGET_KL
)
from schedule_utils import get_schedule, get_entropy_coef_at_progress, get_target_kl_at_progress

class BoundedEntropyActorCriticPolicy(MlpPolicy):
    """
    Custom Actor-Critic policy that applies bounds to entropy loss calculation.
    
    This policy modifies the entropy calculation to ensure the entropy loss
    stays within [-1, 1] bounds, preventing training instability from extreme
    entropy values. Also supports mixed precision training for better GPU utilization.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store the entropy bound
        self.entropy_bound = 1.0
        # Enable mixed precision if available and requested
        self.use_mixed_precision = USE_MIXED_PRECISION if 'USE_MIXED_PRECISION' in globals() else False
        if self.use_mixed_precision and DEVICE == "cuda":
            # Initialize GradScaler for mixed precision
            self.grad_scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions according to the current policy,
        given the observations.
        
        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Call parent method to get normal values
        values, log_prob, entropy = super().evaluate_actions(obs, actions)
        
        # Apply bounds to entropy: clamp to [-entropy_bound, entropy_bound]
        bounded_entropy = torch.clamp(entropy, -self.entropy_bound, self.entropy_bound)
        
        return values, log_prob, bounded_entropy


class BoundedEntropyPPO(BasePPO):
    """
    PPO implementation with bounded entropy loss.
    
    This class extends the standard PPO algorithm to apply bounds on the 
    entropy loss component, restricting it to [-1, 1] to prevent training
    instability caused by extreme entropy values.
    
    The entropy bound is applied in the policy's evaluate_actions method,
    ensuring that the entropy contribution to the total loss remains
    within reasonable bounds.
    """
    
    policy_aliases: ClassVar[Dict[str, Type[ActorCriticPolicy]]] = {
        "MlpPolicy": BoundedEntropyActorCriticPolicy,
    }
    
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = GLOBALLEARNINGRATE,
        n_steps: int = N_STEPS,
        batch_size: int = BATCH_SIZE,
        n_epochs: int = N_EPOCHS,
        gamma: float = GAMMA,
        gae_lambda: float = GAE_LAMBDA,
        clip_range: Union[float, Schedule] = CLIP_RANGE,
        clip_range_vf: Union[None, float, Schedule] = CLIP_RANGE_VF,
        normalize_advantage: bool = NORAD,
        ent_coef: float = ENT_COEF,
        vf_coef: float = VF_COEF,
        max_grad_norm: float = MAX_GRAD_NORM,
        use_sde: bool = USE_SDE,
        sde_sample_freq: int = SDE_SAMPLE_FREQ,
        target_kl: Optional[float] = TARGET_KL,
        stats_window_size: int = STATS_WINDOW_SIZE,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = VERBOSITY,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
        entropy_bound: float = ENTROPY_BOUND,
        value_loss_bound: float = VALUE_LOSS_BOUND,
    ):
        # Store bounds parameters
        self.entropy_bound = entropy_bound
        self.value_loss_bound = value_loss_bound
        
        # MPS gradient accumulation for effective larger batch sizes
        self.gradient_accumulation_steps = 3 if DEVICE == "mps" else 1
        
        # Handle learning rate scheduling
        if USE_LR_SCHEDULE:
            learning_rate = get_schedule(LR_SCHEDULE_TYPE, INITIAL_LR, FINAL_LR)
            if verbose > 0:
                print(f"Using {LR_SCHEDULE_TYPE} learning rate schedule: {INITIAL_LR} -> {FINAL_LR}")
        
        # Store initial values for scheduling
        self.use_ent_schedule = USE_ENT_SCHEDULE
        self.initial_ent_coef = INITIAL_ENT_COEF if USE_ENT_SCHEDULE else ent_coef
        self.final_ent_coef = FINAL_ENT_COEF if USE_ENT_SCHEDULE else ent_coef
        
        self.use_target_kl_schedule = USE_TARGET_KL_SCHEDULE
        self.initial_target_kl = INITIAL_TARGET_KL if USE_TARGET_KL_SCHEDULE else target_kl
        self.final_target_kl = FINAL_TARGET_KL if USE_TARGET_KL_SCHEDULE else target_kl
        
        # Initialize parent class
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )
        
        # Set entropy bound in policy if it has the attribute
        if hasattr(self.policy, 'entropy_bound'):
            self.policy.entropy_bound = entropy_bound
    
    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        
        This method includes the bounded entropy loss calculation.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        
        # Update entropy coefficient if scheduling is enabled
        if self.use_ent_schedule:
            self.ent_coef = get_entropy_coef_at_progress(
                self._current_progress_remaining, 
                self.initial_ent_coef, 
                self.final_ent_coef
            )
        
        # Update target KL if scheduling is enabled
        if self.use_target_kl_schedule:
            self.target_kl = get_target_kl_at_progress(
                self._current_progress_remaining,
                self.initial_target_kl,
                self.final_target_kl
            )
        
        # Log scheduling info occasionally (first update and every 10%)
        if self._n_updates % max(1, int(self.num_timesteps / (self.n_steps * 10))) == 0:
            current_lr = self.policy.optimizer.param_groups[0]['lr']
            if self.verbose > 0:
                print(f"\nScheduling Update - Progress: {(1-self._current_progress_remaining)*100:.1f}%")
                print(f"  Learning Rate: {current_lr:.2e}")
                if self.use_ent_schedule:
                    print(f"  Entropy Coef: {self.ent_coef:.4f}")
                if self.use_target_kl_schedule:
                    print(f"  Target KL: {self.target_kl:.4f}")
        
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # Initialize mixed precision if available
        use_amp = USE_MIXED_PRECISION if 'USE_MIXED_PRECISION' in globals() else False
        use_amp = use_amp and (torch.cuda.is_available() and DEVICE == "cuda" or 
                              torch.backends.mps.is_available() and DEVICE == "mps")
        
        if use_amp:
            if DEVICE == "cuda":
                scaler = torch.amp.GradScaler('cuda')
            elif DEVICE == "mps":
                # MPS doesn't need GradScaler, it handles mixed precision differently
                scaler = None
        
        # Initialize gradient accumulation counter
        self._current_batch_idx = 0
        self.policy.optimizer.zero_grad()
        
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                # Use mixed precision for forward pass if available
                if use_amp:
                    if DEVICE == "cuda":
                        with torch.amp.autocast('cuda'):
                            values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                    elif DEVICE == "mps":
                        with torch.amp.autocast('cpu', dtype=torch.float16):
                            values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                else:
                    values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                
                # Apply bounds to value loss
                value_loss = torch.clamp(value_loss, -self.value_loss_bound, self.value_loss_bound)
                
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                # NOTE: entropy is already bounded in the policy's evaluate_actions method
                entropy_loss = -torch.mean(entropy)
                
                # Apply additional bounds to the entropy loss itself (safety measure)
                entropy_loss = torch.clamp(entropy_loss, -self.entropy_bound, self.entropy_bound)
                
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Gradient accumulation for MPS optimization
                # Scale loss by accumulation steps to maintain same effective learning rate
                loss = loss / self.gradient_accumulation_steps
                
                if use_amp and DEVICE == "cuda" and scaler is not None:
                    # CUDA mixed precision backward pass
                    scaler.scale(loss).backward()
                else:
                    # Standard backward pass (includes MPS mixed precision)
                    loss.backward()
                
                # Only update parameters every gradient_accumulation_steps
                batch_idx = getattr(self, '_current_batch_idx', 0)
                self._current_batch_idx = (batch_idx + 1) % self.gradient_accumulation_steps
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if use_amp and DEVICE == "cuda" and scaler is not None:
                        # CUDA mixed precision optimizer step
                        scaler.unscale_(self.policy.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                        scaler.step(self.policy.optimizer)
                        scaler.update()
                    else:
                        # Standard optimizer step (includes MPS)
                        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                        self.policy.optimizer.step()
                    
                    # Reset gradients after accumulation step
                    self.policy.optimizer.zero_grad()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        
        # Use JAX for explained variance if available and data is large enough
        buffer_values_flat = self.rollout_buffer.values.flatten()
        buffer_returns_flat = self.rollout_buffer.returns.flatten()
        
        if JAX_AVAILABLE and len(pg_losses) > 0 and len(buffer_values_flat) > 1000:
            # Only use JAX for larger datasets where compilation overhead is worth it
            buffer_values = to_jax_if_available(buffer_values_flat)
            buffer_returns = to_jax_if_available(buffer_returns_flat)
            explained_var = float(to_numpy_if_jax(compute_explained_variance_jax(buffer_values, buffer_returns)))
        else:
            explained_var = explained_variance(buffer_values_flat, buffer_returns_flat)

        # Logs - handle empty lists
        self.logger.record("train/entropy_loss", np.mean(entropy_losses) if entropy_losses else 0.0)
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses) if pg_losses else 0.0)
        self.logger.record("train/value_loss", np.mean(value_losses) if value_losses else 0.0)
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs) if approx_kl_divs else 0.0)
        self.logger.record("train/clip_fraction", np.mean(clip_fractions) if clip_fractions else 0.0)
        self.logger.record("train/loss", loss.item() if 'loss' in locals() else 0.0)
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/entropy_bound", self.entropy_bound)  # Log the entropy bound
        self.logger.record("train/value_loss_bound", self.value_loss_bound)  # Log the value loss bound
        
        # Log scheduled hyperparameters
        if self.use_ent_schedule:
            self.logger.record("train/entropy_coefficient", self.ent_coef)
        if self.use_target_kl_schedule:
            self.logger.record("train/target_kl", self.target_kl)
        
        # Log current learning rate
        self.logger.record("train/learning_rate", self.policy.optimizer.param_groups[0]['lr'])
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


# Additional imports moved to top of file

# Ensure backward compatibility with the original PPO interface
__all__ = ["BoundedEntropyPPO", "BoundedEntropyActorCriticPolicy"]
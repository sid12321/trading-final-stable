# Optimized Stock Trading Environment for MPS/GPU performance
import warnings
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL')

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from parameters import INITIAL_ACCOUNT_BALANCE, COST_PER_TRADE, BUYTHRESHOLD, SELLTHRESHOLD

class StockTradingEnvOptimized(gym.Env):
    """Optimized stock trading environment with pre-computed features for MPS/GPU acceleration"""
    
    # Class variables for tracking actions across all instances
    action_dict = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
    amount_dict = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
    
    def __init__(self, df, NLAGS=5, NUMVARS=4, MAXIMUM_SHORT_VALUE=INITIAL_ACCOUNT_BALANCE,
                 INITIAL_ACCOUNT_BALANCE=INITIAL_ACCOUNT_BALANCE, MAX_STEPS=20000,
                 finalsignalsp=[], INITIAL_NET_WORTH=INITIAL_ACCOUNT_BALANCE, 
                 INITIAL_SHARES_HELD=0, COST_PER_TRADE=COST_PER_TRADE,
                 BUYTHRESHOLD=BUYTHRESHOLD, SELLTHRESHOLD=SELLTHRESHOLD):
        super(StockTradingEnvOptimized, self).__init__()
        
        # Store parameters
        self.df = df
        self.NLAGS = NLAGS
        self.NUMVARS = NUMVARS
        self.MAXIMUM_SHORT_VALUE = MAXIMUM_SHORT_VALUE
        self.INITIAL_ACCOUNT_BALANCE = INITIAL_ACCOUNT_BALANCE
        self.INITIAL_NET_WORTH = INITIAL_NET_WORTH
        self.INITIAL_SHARES_HELD = INITIAL_SHARES_HELD
        self.MAX_STEPS = MAX_STEPS
        self.finalsignalsp = finalsignalsp
        self.COST_PER_TRADE = COST_PER_TRADE
        self.BUYTHRESHOLD = BUYTHRESHOLD
        self.SELLTHRESHOLD = SELLTHRESHOLD
        
        # Track current date for daily position liquidation
        self.current_date = None
        self.previous_date = None
        
        # Track daily performance for end-of-day rewards
        self.daily_start_net_worth = INITIAL_NET_WORTH
        self.daily_liquidation_reward = 0.0
        
        # Pre-compute all observations for efficiency
        self._precompute_observations()
        
        # Pre-extract frequently accessed columns as numpy arrays
        self.vwap2_array = self.df['vwap2'].values.astype(np.float32)
        self.currentdate_array = self.df['currentdate'].values if 'currentdate' in self.df.columns else None
        
        # Pre-allocate observation buffer
        self.obs_buffer = np.zeros((self.NUMVARS, self.NLAGS+1), dtype=np.float32)
        
        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.float32(np.array([-1, 0])), 
            high=np.float32(np.array([1, 1])), 
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=np.float32(0), 
            high=np.float32(1), 
            shape=(self.NUMVARS, self.NLAGS+1), 
            dtype=np.float32
        )
    
    def _precompute_observations(self):
        """Pre-compute all observations for the entire dataset"""
        print("Pre-computing observations for optimized performance...")
        
        # Create observation matrix for all timesteps
        max_steps = len(self.df) - self.NLAGS
        self.precomputed_obs = np.zeros((max_steps, self.NUMVARS, self.NLAGS+1), dtype=np.float32)
        
        # Convert signals to numpy array once
        if self.finalsignalsp:
            signals_df = self.df[self.finalsignalsp]
            signals_array = signals_df.values.astype(np.float32)
            
            # Compute observations for each valid timestep
            for step in range(self.NLAGS, len(self.df)):
                idx = step - self.NLAGS
                # Extract window of data
                window = signals_array[(step-self.NLAGS):(step+1), :]
                # Transpose to get (NUMVARS, NLAGS+1)
                self.precomputed_obs[idx] = window.T
        
        print(f"Pre-computed {len(self.precomputed_obs)} observations")
    
    def _next_observation(self):
        """Optimized observation retrieval using pre-computed data"""
        # Direct array indexing - no DataFrame operations
        idx = self.current_step - self.NLAGS
        if idx >= 0 and idx < len(self.precomputed_obs):
            return self.precomputed_obs[idx].copy()
        else:
            # Fallback for edge cases
            return self._next_observation_fallback()
    
    def _next_observation_fallback(self):
        """Fallback observation method for edge cases"""
        data_slices = []
        for sig in self.finalsignalsp:
            data_slices.append(self.df.loc[(self.current_step-self.NLAGS): self.current_step, sig])
        
        if data_slices:
            sigframe = pd.concat(data_slices, axis=1, keys=self.finalsignalsp)
        else:
            sigframe = pd.DataFrame()
        
        obs = sigframe.T.to_numpy().astype(np.float32)
        return obs
    
    def _get_current_price(self):
        """Optimized price retrieval using pre-extracted array"""
        return self.vwap2_array[self.current_step]
    
    def _get_previous_price(self):
        """Optimized previous price retrieval"""
        if self.current_step > 0:
            return self.vwap2_array[self.current_step - 1]
        return self.vwap2_array[self.current_step]
    
    def _get_current_date(self):
        """Optimized date retrieval"""
        if self.currentdate_array is not None:
            return self.currentdate_array[self.current_step]
        return None
    
    def _liquidate_daily_positions(self):
        """Liquidate all positions at the end of the trading day and calculate daily reward bonus"""
        daily_pnl_before_liquidation = self.net_worth - self.daily_start_net_worth
        
        if self.shares_held == 0:
            # No positions to liquidate, but still calculate daily performance reward
            self._calculate_daily_liquidation_reward(daily_pnl_before_liquidation, 0)
            return
        
        current_price = self._get_current_price()
        transaction_cost_rate = self.COST_PER_TRADE
        liquidation_pnl = 0
        
        # Calculate liquidation
        if self.shares_held > 0:
            # Liquidate long position
            revenue = self.shares_held * current_price
            transaction_cost = revenue * transaction_cost_rate
            net_revenue = revenue - transaction_cost
            liquidation_pnl = net_revenue - (self.shares_held * current_price)
            self.balance += net_revenue
            print(f"Day-end liquidation: Sold {self.shares_held} shares at {current_price:.2f} for net revenue {net_revenue:.2f}")
        else:
            # Cover short position
            cost = abs(self.shares_held) * current_price
            transaction_cost = cost * transaction_cost_rate
            total_cost = cost + transaction_cost
            liquidation_pnl = -(total_cost - (abs(self.shares_held) * current_price))
            self.balance -= total_cost
            print(f"Day-end liquidation: Covered {abs(self.shares_held)} short shares at {current_price:.2f} for total cost {total_cost:.2f}")
        
        # Reset position to zero
        self.shares_held = 0
        
        # Update net worth
        self.net_worth = self.balance + self.shares_held * current_price
        
        # Calculate total daily P&L including liquidation
        total_daily_pnl = self.net_worth - self.daily_start_net_worth
        
        # Calculate daily liquidation reward bonus
        self._calculate_daily_liquidation_reward(total_daily_pnl, liquidation_pnl)
        
        # Reset daily tracking for next day
        self.daily_start_net_worth = self.net_worth
    
    def _calculate_daily_liquidation_reward(self, total_daily_pnl, liquidation_pnl):
        """Calculate reward bonus for end-of-day liquidation based on daily performance"""
        daily_return_pct = total_daily_pnl / self.INITIAL_ACCOUNT_BALANCE
        
        # Base daily performance reward
        if daily_return_pct > 0:
            base_reward = daily_return_pct * 500
        else:
            base_reward = daily_return_pct * 200
        
        # Liquidation efficiency bonus
        liquidation_efficiency = 0
        if abs(liquidation_pnl) > 0:
            liquidation_return = liquidation_pnl / self.INITIAL_ACCOUNT_BALANCE
            if liquidation_return > 0:
                liquidation_efficiency = liquidation_return * 300
            else:
                liquidation_efficiency = liquidation_return * 100
        
        # Risk-adjusted reward
        consistency_bonus = 0
        if hasattr(self, 'daily_returns_history'):
            self.daily_returns_history.append(daily_return_pct)
            if len(self.daily_returns_history) > 10:
                self.daily_returns_history = self.daily_returns_history[-10:]
            
            if len(self.daily_returns_history) >= 5:
                daily_sharpe = np.mean(self.daily_returns_history) / (np.std(self.daily_returns_history) + 1e-8)
                consistency_bonus = daily_sharpe * 50
        else:
            self.daily_returns_history = [daily_return_pct]
        
        # Combine all daily reward components
        self.daily_liquidation_reward = base_reward + liquidation_efficiency + consistency_bonus
        
        print(f"Daily liquidation reward: {self.daily_liquidation_reward:.2f} (Daily P&L: {total_daily_pnl:.2f}, Return: {daily_return_pct*100:.2f}%)")
    
    def _take_action(self, action):
        """Optimized action execution"""
        current_price = self._get_current_price()
        prev_net_worth = self.net_worth
        
        action_type = action[0]
        amount = max(0.1, min(1.0, action[1]))  # Clamp amount between 0.1 and 1.0
        
        # Calculate transaction costs
        transaction_cost_rate = self.COST_PER_TRADE
        
        if action_type >= self.BUYTHRESHOLD:  
            max_affordable = self.balance / current_price
            shares_to_buy = int(max_affordable * amount)
            
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                transaction_cost = cost * transaction_cost_rate
                total_cost = cost + transaction_cost
                
                if total_cost <= self.balance:
                    self.balance -= total_cost
                    self.shares_held += shares_to_buy
                    StockTradingEnvOptimized.action_dict['BUY'] += 1
                    StockTradingEnvOptimized.amount_dict['BUY'] += amount
                    
        elif action_type <= self.SELLTHRESHOLD:
            max_sellable = self.shares_held + (self.MAXIMUM_SHORT_VALUE / current_price)
            shares_to_sell = int(max_sellable * amount)
            
            if shares_to_sell > 0:
                revenue = shares_to_sell * current_price
                transaction_cost = revenue * transaction_cost_rate
                net_revenue = revenue - transaction_cost
                
                self.balance += net_revenue
                self.shares_held -= shares_to_sell
                StockTradingEnvOptimized.action_dict['SELL'] += 1
                StockTradingEnvOptimized.amount_dict['SELL'] += amount
        else:
            StockTradingEnvOptimized.action_dict['HOLD'] += 1
            StockTradingEnvOptimized.amount_dict['HOLD'] += amount
        
        # Update net worth
        self.net_worth = self.balance + self.shares_held * current_price
        
        # Store action info for reward calculation
        self.prev_net_worth = prev_net_worth
        self.action_taken = action_type
        
        # Track action history for over-activity penalty
        if not hasattr(self, 'action_history'):
            self.action_history = []
        self.action_history.append(abs(action_type))
        if len(self.action_history) > 20:
            self.action_history = self.action_history[-20:]
    
    def step(self, action):
        """Optimized step function"""
        noisy = len(action) == 1
        if noisy:
            action = action[0]
        
        # Check for date change and liquidate positions if needed
        daily_liquidation_occurred = False
        if self.currentdate_array is not None:
            current_date = self._get_current_date()
            if self.current_date is not None and current_date != self.current_date:
                print(f"Date changed from {self.current_date} to {current_date}, liquidating positions")
                self._liquidate_daily_positions()
                daily_liquidation_occurred = True
            self.previous_date = self.current_date
            self.current_date = current_date
        
        current_price = self._get_current_price()
        prev_price = self._get_previous_price()
        
        self._take_action(action)
        profit = self.net_worth - self.INITIAL_ACCOUNT_BALANCE
        profit_pct = profit / self.INITIAL_ACCOUNT_BALANCE
        
        # Reward components
        profit_reward = profit_pct * 100
        step_pnl = self.net_worth - self.prev_net_worth
        step_reward = step_pnl / self.INITIAL_ACCOUNT_BALANCE * 1000
        price_change = (current_price - prev_price) / prev_price
        
        # Market trend calculation - optimized with array slicing
        if self.current_step >= 10:
            recent_prices = self.vwap2_array[max(0, self.current_step-10):self.current_step+1]
            market_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        else:
            market_trend = 0
        
        # Calculate action and position rewards
        action_reward = self._calculate_action_reward(price_change, market_trend)
        position_reward = self._calculate_position_reward(market_trend, current_price)
        
        # Risk and activity penalties
        portfolio_value = abs(self.shares_held * current_price)
        leverage_ratio = portfolio_value / self.INITIAL_ACCOUNT_BALANCE
        risk_penalty = -max(0, (leverage_ratio - 2.0) * 10)
        
        activity_penalty = self._calculate_activity_penalty()
        
        # Sharpe-like reward
        sharpe_reward = self._calculate_sharpe_reward(step_pnl)
        
        # Action balance reward
        balance_reward = self._calculate_balance_reward()
        
        # Daily liquidation component
        daily_liquidation_component = 0.0
        if daily_liquidation_occurred and hasattr(self, 'daily_liquidation_reward'):
            daily_liquidation_component = self.daily_liquidation_reward
            self.daily_liquidation_reward = 0.0
        
        # Scale and combine reward components
        reward = self._combine_rewards(
            profit_reward, step_reward, action_reward, position_reward,
            risk_penalty, activity_penalty, sharpe_reward, balance_reward,
            daily_liquidation_component
        )
        
        self.current_step += 1
        
        # Episode termination conditions
        done = (self.net_worth <= self.INITIAL_ACCOUNT_BALANCE * 0.5 or 
                self.current_step >= min(self.MAX_STEPS, len(self.df)-1) or 
                self.net_worth <= 0)
        
        obs = self._next_observation()
        
        # Update previous observation for signal change tracking
        self.previous_obs = obs.copy()
        
        info = self._create_info_dict(
            profit, profit_pct, step_pnl, action_reward, position_reward,
            risk_penalty, activity_penalty, sharpe_reward, balance_reward,
            daily_liquidation_component, market_trend
        )
        
        # Episode reward and length when done
        if done:
            episode_profit_pct = profit / self.INITIAL_ACCOUNT_BALANCE
            scaled_episode_reward = np.clip(np.tanh(episode_profit_pct * 5) * 1, -1, 1)
            info['r'] = scaled_episode_reward
            info['l'] = self.current_step
        
        if noisy:
            print(f'Step: {self.current_step}, Position: {self.shares_held}, Profit: {profit:.2f}, '
                  f'Balance: {self.balance:.2f}, NetWorth: {self.net_worth:.2f}, '
                  f'Price: {current_price:.2f}, Reward: {reward:.4f}')
        
        # Gymnasium format
        terminated = done and self.current_step >= len(self.df) - 1
        truncated = done and not terminated
        return obs, reward, terminated, truncated, info
    
    def _calculate_action_reward(self, price_change, market_trend):
        """Calculate reward based on action taken"""
        if not hasattr(self, 'action_taken'):
            return 0
        
        if self.action_taken > self.BUYTHRESHOLD:  # Buy action
            if price_change > 0:
                return 5
            elif market_trend > 0.002:
                return 3
            else:
                return -1
        elif self.action_taken < self.SELLTHRESHOLD:  # Sell action
            if price_change < 0:
                return 5
            elif market_trend < -0.002:
                return 3
            else:
                return -1
        else:  # Hold action
            if abs(price_change) < 0.002:
                return 2
            else:
                return 0
    
    def _calculate_position_reward(self, market_trend, current_price):
        """Calculate reward for holding positions"""
        position_reward = 0
        
        if self.shares_held < 0 and market_trend < -0.002:  # Short position in downtrend
            position_reward = abs(self.shares_held) * current_price * abs(market_trend) * 15
            if hasattr(self, 'position_hold_time'):
                self.position_hold_time += 1
                if self.position_hold_time > 5:
                    position_reward += 1
            else:
                self.position_hold_time = 1
        elif self.shares_held > 0 and market_trend > 0.002:  # Long position in uptrend
            position_reward = self.shares_held * current_price * abs(market_trend) * 15
            if hasattr(self, 'position_hold_time'):
                self.position_hold_time += 1
                if self.position_hold_time > 5:
                    position_reward += 1
            else:
                self.position_hold_time = 1
        else:
            self.position_hold_time = 0
        
        return position_reward
    
    def _calculate_activity_penalty(self):
        """Calculate penalty for over-trading"""
        activity_penalty = 0
        
        if hasattr(self, 'action_history') and len(self.action_history) >= 10:
            recent_activity = np.mean(np.abs(self.action_history[-10:]))
            if recent_activity > 0.25:
                activity_penalty = -(recent_activity - 0.25) * 60
            
            if len(self.action_history) >= 5:
                action_changes = sum(1 for i in range(1, 5) 
                                   if abs(self.action_history[-i] - self.action_history[-i-1]) > 0.3)
                if action_changes >= 2:
                    activity_penalty -= 20
            
            consecutive_trades = 0
            for i in range(min(5, len(self.action_history))):
                if abs(self.action_history[-(i+1)]) > self.BUYTHRESHOLD:
                    consecutive_trades += 1
                else:
                    break
            if consecutive_trades >= 2:
                activity_penalty -= consecutive_trades * 10
        
        return activity_penalty
    
    def _calculate_sharpe_reward(self, step_pnl):
        """Calculate Sharpe ratio based reward"""
        returns_list = getattr(self, 'returns_history', [])
        returns_list.append(step_pnl / self.INITIAL_ACCOUNT_BALANCE)
        if len(returns_list) > 50:
            returns_list = returns_list[-50:]
        self.returns_history = returns_list
        
        sharpe_reward = 0
        if len(returns_list) >= 10:
            mean_return = np.mean(returns_list)
            std_return = np.std(returns_list) + 1e-8
            sharpe_reward = (mean_return / std_return) * 10
        
        return sharpe_reward
    
    def _calculate_balance_reward(self):
        """Calculate reward for balanced action distribution"""
        balance_reward = 0
        
        if hasattr(self, 'action_history') and len(self.action_history) >= 20:
            recent_actions = self.action_history[-20:]
            buy_actions = sum(1 for a in recent_actions if a > self.BUYTHRESHOLD)
            sell_actions = sum(1 for a in recent_actions if a < self.SELLTHRESHOLD)
            
            total_actions = len(recent_actions)
            if total_actions > 0:
                buy_ratio = buy_actions / total_actions
                sell_ratio = sell_actions / total_actions
                hold_ratio = 1 - buy_ratio - sell_ratio
                
                # Ideal ratios: 30% buy, 30% sell, 40% hold
                ideal_buy, ideal_sell, ideal_hold = 0.3, 0.3, 0.4
                balance_score = 1.0 - (abs(buy_ratio - ideal_buy) + 
                                      abs(sell_ratio - ideal_sell) + 
                                      abs(hold_ratio - ideal_hold)) / 2
                balance_reward = balance_score * 3
        
        return balance_reward
    
    def _combine_rewards(self, profit_reward, step_reward, action_reward, position_reward,
                        risk_penalty, activity_penalty, sharpe_reward, balance_reward,
                        daily_liquidation_component):
        """Combine all reward components with scaling"""
        
        def scale_reward_component(value, scale_factor=1.0):
            if abs(value) <= 1e-8:
                return 0.0
            scaled = np.tanh(value * scale_factor)
            return np.clip(scaled, -1.0, 1.0)
        
        def scale_penalty_component(value, scale_factor=1.0):
            if value >= 0:
                return 0.0
            scaled = np.tanh(value * scale_factor)
            return np.clip(scaled, -1.0, 1.0)
        
        
        # Scale components
        scaled_profit = scale_reward_component(profit_reward, scale_factor=0.25)
        scaled_step = scale_reward_component(step_reward, scale_factor=0.15)
        scaled_action = scale_reward_component(action_reward, scale_factor=0.1)
        scaled_position = scale_reward_component(position_reward, scale_factor=0.05)
        scaled_risk = scale_penalty_component(risk_penalty, scale_factor=0.05)
        scaled_activity = scale_penalty_component(activity_penalty, scale_factor=0.1)
        scaled_sharpe = scale_reward_component(sharpe_reward, scale_factor=0.2)
        scaled_balance = scale_reward_component(balance_reward, scale_factor=0.05)
        scaled_daily_liquidation = scale_reward_component(daily_liquidation_component, scale_factor=0.05)
        
        #SID: Ideally need a GA to optimize the hyperparameters

        # Combine with weights
        reward = (scaled_profit * 3 + scaled_step * 0.1 + scaled_action * 1 +
                 scaled_position * 0.5 + scaled_risk * 1 + scaled_activity * 0.5 +
                 scaled_sharpe * 1.5 + scaled_balance * 0.5 + scaled_daily_liquidation * 0.5)/8.6
        
        # Scale to [-0.1, 0.1] range
        return float(np.clip(reward * 0.1, -0.1, 0.1))
    
    def _create_info_dict(self, profit, profit_pct, step_pnl, action_reward, position_reward,
                         risk_penalty, activity_penalty, sharpe_reward, balance_reward,
                         daily_liquidation_component, market_trend):
        """Create info dictionary for step return"""
        return {
            'profit': profit,
            'profit_pct': profit_pct,
            'step_pnl': step_pnl,
            'net_worth': self.net_worth,
            'shares_held': self.shares_held,
            'balance': self.balance,
            'action_reward': action_reward,
            'position_reward': position_reward,
            'risk_penalty': risk_penalty,
            'activity_penalty': activity_penalty,
            'sharpe_reward': sharpe_reward,
            'balance_reward': balance_reward,
            'daily_liquidation_reward': daily_liquidation_component,
            'market_trend': market_trend,
            'recent_activity': np.mean(self.action_history[-10:]) if hasattr(self, 'action_history') and len(self.action_history) >= 10 else 0
        }
    
    def reset(self, seed=None, **kwargs):
        """Reset environment to initial state"""
        if seed is not None:
            np.random.seed(seed)
        
        self.balance = self.INITIAL_ACCOUNT_BALANCE
        self.net_worth = self.INITIAL_NET_WORTH
        self.shares_held = self.INITIAL_SHARES_HELD
        self.current_step = self.NLAGS
        
        # Initialize date tracking
        if self.currentdate_array is not None:
            self.current_date = self._get_current_date()
            self.previous_date = None
        else:
            self.current_date = None
            self.previous_date = None
        
        # Reset tracking variables
        self.prev_net_worth = self.INITIAL_ACCOUNT_BALANCE
        self.action_taken = 0
        self.returns_history = []
        # Initialize action history with balanced actions
        self.action_history = [self.BUYTHRESHOLD+0.1, self.SELLTHRESHOLD-0.1, 0.0, 
                              self.BUYTHRESHOLD+0.1, self.SELLTHRESHOLD-0.1, 0.0] * 3
        self.market_direction_history = []
        self.position_hold_time = 0
        self.previous_obs = None
        self.previous_signals = None
        
        # Reset daily tracking variables
        self.daily_start_net_worth = self.INITIAL_NET_WORTH
        self.daily_liquidation_reward = 0.0
        self.daily_returns_history = []
        
        observation = self._next_observation()
        info = {}
        return observation, info
    
    def seed(self, seed=None):
        """Set random seed"""
        np.random.seed(seed)
        return [seed]
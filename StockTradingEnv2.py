# Import only necessary modules to avoid circular imports
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from parameters import INITIAL_ACCOUNT_BALANCE, COST_PER_TRADE, BUYTHRESHOLD, SELLTHRESHOLD

class StockTradingEnv2(gym.Env):
    """A stock trading environment for OpenAI gym""" 
    
    # Class variables for tracking actions across all instances
    action_dict = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
    amount_dict = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
    
    def __init__(self, df,NLAGS = 5,NUMVARS = 4,MAXIMUM_SHORT_VALUE = INITIAL_ACCOUNT_BALANCE,INITIAL_ACCOUNT_BALANCE=INITIAL_ACCOUNT_BALANCE,MAX_STEPS=20000,finalsignalsp=[],INITIAL_NET_WORTH=INITIAL_ACCOUNT_BALANCE, INITIAL_SHARES_HELD=0,COST_PER_TRADE=COST_PER_TRADE,BUYTHRESHOLD=BUYTHRESHOLD,SELLTHRESHOLD=SELLTHRESHOLD):
        super(StockTradingEnv2, self).__init__()
        self.df = df
        self.NLAGS = NLAGS
        self.NUMVARS = NUMVARS
        self.MAXIMUM_SHORT_VALUE = MAXIMUM_SHORT_VALUE
        self.INITIAL_ACCOUNT_BALANCE = INITIAL_ACCOUNT_BALANCE
        self.INITIAL_NET_WORTH=INITIAL_NET_WORTH
        self.INITIAL_SHARES_HELD=INITIAL_SHARES_HELD
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

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.float32(np.array([-1, 0])), high=np.float32(np.array([1, 1])), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.float32(0), high=np.float32(1), shape=(self.NUMVARS, self.NLAGS+1), dtype=np.float32)

    def _next_observation(self):
        # Get the stock data points for the last NLAGS periods
        data_slices = []
        for sig in self.finalsignalsp:
            data_slices.append(self.df.loc[(self.current_step-self.NLAGS): self.current_step, sig])
        
        if data_slices:
            sigframe = pd.concat(data_slices, axis=1, keys=self.finalsignalsp)
        else:
            sigframe = pd.DataFrame()
        
        obs = sigframe.T.to_numpy().astype(np.float32)
        
        return obs

    def _liquidate_daily_positions(self):
        """Liquidate all positions at the end of the trading day and calculate daily reward bonus"""
        daily_pnl_before_liquidation = self.net_worth - self.daily_start_net_worth
        
        if self.shares_held == 0:
            # No positions to liquidate, but still calculate daily performance reward
            self._calculate_daily_liquidation_reward(daily_pnl_before_liquidation, 0)
            return
        
        current_price = self.df.loc[self.current_step, "vwap2"]
        transaction_cost_rate = self.COST_PER_TRADE
        liquidation_pnl = 0
        
        # Calculate liquidation
        if self.shares_held > 0:
            # Liquidate long position
            revenue = self.shares_held * current_price
            transaction_cost = revenue * transaction_cost_rate
            net_revenue = revenue - transaction_cost
            liquidation_pnl = net_revenue - (self.shares_held * current_price)  # Account for unrealized gains/losses
            self.balance += net_revenue
            print(f"Day-end liquidation: Sold {self.shares_held} shares at {current_price:.2f} for net revenue {net_revenue:.2f}")
        else:
            # Cover short position
            cost = abs(self.shares_held) * current_price
            transaction_cost = cost * transaction_cost_rate
            total_cost = cost + transaction_cost
            liquidation_pnl = -(total_cost - (abs(self.shares_held) * current_price))  # Account for short position gains/losses
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
        
        # Base daily performance reward - stronger for positive returns
        if daily_return_pct > 0:
            base_reward = daily_return_pct * 500  # Amplified reward for profitable days
        else:
            base_reward = daily_return_pct * 200  # Reduced penalty for losing days
        
        # Liquidation efficiency bonus - reward for closing positions profitably
        liquidation_efficiency = 0
        if abs(liquidation_pnl) > 0:  # Only if there was actual liquidation
            liquidation_return = liquidation_pnl / self.INITIAL_ACCOUNT_BALANCE
            if liquidation_return > 0:
                liquidation_efficiency = liquidation_return * 300  # Bonus for profitable liquidation
            else:
                liquidation_efficiency = liquidation_return * 100  # Smaller penalty for unprofitable liquidation
        
        # Risk-adjusted reward - favor consistent daily gains
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
        current_price = self.df.loc[self.current_step, "vwap2"]
        prev_net_worth = self.net_worth
        
        action_type = action[0]
        amount = max(0.1, min(1.0, action[1]))  # Clamp amount between 0.1 and 1.0
        
        # Signal change threshold - only trade if signals have changed significantly
        # signal_change_threshold = 0.25
        # 
        # if hasattr(self, 'previous_obs') and self.previous_obs is not None and self.current_step > self.NLAGS + 5:
        #     # Calculate signal change from recent observations
        #     current_obs = self._next_observation()
        #     signal_change = np.mean(np.abs(current_obs.flatten() - self.previous_obs.flatten()))
        #     if signal_change < signal_change_threshold and abs(action_type) > BUYTHRESHOLD:
        #         action_type = 0  # Force hold if signals haven't changed significantly
        
        # Calculate transaction costs (0.1% per trade)
        transaction_cost_rate = self.COST_PER_TRADE
        
        if action_type >= BUYTHRESHOLD:  
          max_affordable = self.balance / current_price
          shares_to_buy = int(max_affordable * amount)
          
          if shares_to_buy > 0:
              cost = shares_to_buy * current_price
              transaction_cost = cost * transaction_cost_rate
              total_cost = cost + transaction_cost
              
              if total_cost <= self.balance:
                  self.balance -= total_cost
                  self.shares_held += shares_to_buy
                  StockTradingEnv2.action_dict['BUY'] += 1
                  StockTradingEnv2.amount_dict['BUY'] += amount
                  
        elif action_type <= SELLTHRESHOLD:  # Sell threshold
          max_sellable = self.shares_held + (self.MAXIMUM_SHORT_VALUE / current_price)
          shares_to_sell = int(max_sellable * amount)
          
          if shares_to_sell > 0:
              revenue = shares_to_sell * current_price
              transaction_cost = revenue * transaction_cost_rate
              net_revenue = revenue - transaction_cost
              
              self.balance += net_revenue
              self.shares_held -= shares_to_sell
              StockTradingEnv2.action_dict['SELL'] += 1
              StockTradingEnv2.amount_dict['SELL'] += amount
        else:
          StockTradingEnv2.action_dict['HOLD'] += 1
          StockTradingEnv2.amount_dict['HOLD'] += amount
          
        # Update net worth
        self.net_worth = self.balance + self.shares_held * current_price
        
        # Store action info for reward calculation
        self.prev_net_worth = prev_net_worth
        self.action_taken = action_type
        
        # Track action history for over-activity penalty
        if not hasattr(self, 'action_history'):
            self.action_history = []
        self.action_history.append(abs(action_type))
        if len(self.action_history) > 20:  # Keep last 20 actions
            self.action_history = self.action_history[-20:] 

    def step(self, action):
        noisy = len(action) == 1
        if noisy:
            action = action[0]

        # Check for date change and liquidate positions if needed
        daily_liquidation_occurred = False
        if 'currentdate' in self.df.columns:
            current_date = self.df.loc[self.current_step, 'currentdate']
            if self.current_date is not None and current_date != self.current_date:
                # New day started, liquidate all positions from previous day
                print(f"Date changed from {self.current_date} to {current_date}, liquidating positions")
                self._liquidate_daily_positions()
                daily_liquidation_occurred = True
            self.previous_date = self.current_date
            self.current_date = current_date

        current_price = self.df.loc[self.current_step, "vwap2"]
        prev_price = self.df.loc[self.current_step-1, "vwap2"] if self.current_step > 0 else current_price
        
        self._take_action(action)
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
        profit_pct = profit / INITIAL_ACCOUNT_BALANCE
        
        # Reward components
        profit_reward = profit_pct * 100
        step_pnl = self.net_worth - self.prev_net_worth
        step_reward = step_pnl / INITIAL_ACCOUNT_BALANCE * 1000
        price_change = (current_price - prev_price) / prev_price
        
        # Market trend calculation
        if self.current_step >= 10:
            recent_prices = [self.df.loc[i, "vwap2"] for i in range(max(0, self.current_step-10), self.current_step+1)]
            market_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        else:
            market_trend = 0
        
        action_reward = 0
        position_reward = 0
        
        if hasattr(self, 'action_taken'):
            if self.action_taken > BUYTHRESHOLD:  # Buy action
                if price_change > 0:
                    action_reward = 5
                elif market_trend > 0.002:
                    action_reward = 3
                else:
                    action_reward = -1
            elif self.action_taken < SELLTHRESHOLD:  # Sell action
                if price_change < 0:
                    action_reward = 5
                elif market_trend < -0.002:
                    action_reward = 3
                else:
                    action_reward = -1
            else:  # Hold action
                if abs(price_change) < 0.002:
                    action_reward = 2
                else:
                    action_reward = 0
            
            # Balanced position holding rewards - equal treatment for long/short
            if self.shares_held < 0 and market_trend < -0.002:  # Short position in downtrend
                position_reward = abs(self.shares_held) * current_price * abs(market_trend) * 15  # Reduced from 30
                if hasattr(self, 'position_hold_time'):
                    self.position_hold_time += 1
                    if self.position_hold_time > 5:  # Reduced from 8
                        position_reward += 1  # Reduced from 2
                else:
                    self.position_hold_time = 1
            elif self.shares_held > 0 and market_trend > 0.002:  # Long position in uptrend
                position_reward = self.shares_held * current_price * abs(market_trend) * 15  # Now uses abs() like shorts
                if hasattr(self, 'position_hold_time'):
                    self.position_hold_time += 1
                    if self.position_hold_time > 5:  # Reduced from 8
                        position_reward += 1  # Reduced from 2
                else:
                    self.position_hold_time = 1
            else:
                self.position_hold_time = 0
        
        # Risk penalty
        portfolio_value = abs(self.shares_held * current_price)
        leverage_ratio = portfolio_value / INITIAL_ACCOUNT_BALANCE
        risk_penalty = -max(0, (leverage_ratio - 2.0) * 10)
        # Over-activity penalty
        activity_penalty = 0
        if hasattr(self, 'action_history') and len(self.action_history) >= 10:
            recent_activity = np.mean(np.abs(self.action_history[-10:]))
            if recent_activity > 0.25:
                activity_penalty = -(recent_activity - 0.25) * 60

            if len(self.action_history) >= 5:
                action_changes = sum(1 for i in range(1, 5) if abs(self.action_history[-i] - self.action_history[-i-1]) > 0.3)
                if action_changes >= 2:
                    activity_penalty -= 20

            consecutive_trades = 0
            for i in range(min(5, len(self.action_history))):
                if abs(self.action_history[-(i+1)]) > BUYTHRESHOLD:
                    consecutive_trades += 1
                else:
                    break
            if consecutive_trades >= 2:
                activity_penalty -= consecutive_trades * 10
        
        # Sharpe-like reward
        returns_list = getattr(self, 'returns_history', [])
        returns_list.append(step_pnl / INITIAL_ACCOUNT_BALANCE)
        if len(returns_list) > 50:
            returns_list = returns_list[-50:]
        self.returns_history = returns_list
        
        sharpe_reward = 0
        if len(returns_list) >= 10:
            mean_return = np.mean(returns_list)
            std_return = np.std(returns_list) + 1e-8
            sharpe_reward = (mean_return / std_return) * 10
        
        # Action balance reward - encourage variety in trading actions
        balance_reward = 0
        if hasattr(self, 'action_history') and len(self.action_history) >= 20:
            recent_actions = self.action_history[-20:]
            buy_actions = sum(1 for a in recent_actions if a > BUYTHRESHOLD)
            sell_actions = sum(1 for a in recent_actions if a < SELLTHRESHOLD)
            hold_actions = sum(1 for a in recent_actions if abs(a) <= BUYTHRESHOLD)
            
            # Reward balanced action distribution
            total_actions = len(recent_actions)
            if total_actions > 0:
                buy_ratio = buy_actions / total_actions
                sell_ratio = sell_actions / total_actions
                hold_ratio = hold_actions / total_actions
                
                # Ideal ratios: 30% buy, 30% sell, 40% hold
                ideal_buy, ideal_sell, ideal_hold = 0.3, 0.3, 0.4
                balance_score = 1.0 - (abs(buy_ratio - ideal_buy) + abs(sell_ratio - ideal_sell) + abs(hold_ratio - ideal_hold)) / 2
                balance_reward = balance_score * 3  # Scale the reward
        
        # Reward scaling functions
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
        
        # Add daily liquidation reward component
        daily_liquidation_component = 0.0
        if daily_liquidation_occurred and hasattr(self, 'daily_liquidation_reward'):
            daily_liquidation_component = self.daily_liquidation_reward
            # Reset the daily liquidation reward after using it
            self.daily_liquidation_reward = 0.0
        
        # Scale and combine reward components
        scaled_profit = scale_reward_component(profit_reward, scale_factor=0.25)  # Increased from 0.1
        scaled_step = scale_reward_component(step_reward, scale_factor=0.15)  # Reduced from 0.5
        scaled_action = scale_reward_component(action_reward, scale_factor=0.1)
        scaled_position = scale_reward_component(position_reward, scale_factor=0.05)
        scaled_risk = scale_penalty_component(risk_penalty, scale_factor=0.05)  # Slightly increased
        scaled_activity = scale_penalty_component(activity_penalty, scale_factor=0.1)
        scaled_sharpe = scale_reward_component(sharpe_reward, scale_factor=0.2)  # Slightly increased
        scaled_balance = scale_reward_component(balance_reward, scale_factor=0.05)
        scaled_daily_liquidation = scale_reward_component(daily_liquidation_component, scale_factor=0.05)
        
        reward = (scaled_profit * 0.3 + scaled_step * 0.1 + scaled_action * 0.1 +
                 scaled_position * 0.05 + scaled_risk * 0.05 + scaled_activity * 0.05 +
                 scaled_sharpe * 0.25 + scaled_balance * 0.05 + scaled_daily_liquidation * 0.05)
        
        #Scale to [-0.1, 0.1] range and ensure bounds
        reward = float(np.clip(reward * 0.1, -0.1, 0.1))
        
        self.current_step += 1
        
        # Episode termination conditions
        done = (self.net_worth <= INITIAL_ACCOUNT_BALANCE * 0.5 or 
                self.current_step >= min(self.MAX_STEPS, self.df.shape[0]-1) or 
                self.net_worth <= 0)
        
        obs = self._next_observation()
        
        # Update previous observation for signal change tracking
        self.previous_obs = obs.copy()
        
        info = {
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
            'market_trend': market_trend if 'market_trend' in locals() else 0,
            'recent_activity': np.mean(self.action_history[-10:]) if hasattr(self, 'action_history') and len(self.action_history) >= 10 else 0
        }
        
        # Episode reward and length when done
        if done:
            episode_profit_pct = profit / INITIAL_ACCOUNT_BALANCE
            scaled_episode_reward = np.clip(np.tanh(episode_profit_pct * 5) * 1, -1, 1)
            info['r'] = scaled_episode_reward
            info['l'] = self.current_step
        
        if noisy:
            print(f'Step: {self.current_step}, Position: {self.shares_held}, Profit: {profit:.2f}, '
                  f'Balance: {self.balance:.2f}, NetWorth: {self.net_worth:.2f}, '
                  f'Price: {current_price:.2f}, Reward: {reward:.4f}')
                  
        # Gymnasium format: observation, reward, terminated, truncated, info
        terminated = done and self.current_step >= len(self.df) - 1  # Episode ended naturally
        truncated = done and not terminated  # Episode ended due to other conditions
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            np.random.seed(seed)
            
        self.balance = self.INITIAL_ACCOUNT_BALANCE
        self.net_worth = self.INITIAL_NET_WORTH
        self.shares_held = self.INITIAL_SHARES_HELD
        self.current_step = self.NLAGS
        
        # Initialize date tracking
        if 'currentdate' in self.df.columns:
            self.current_date = self.df.loc[self.current_step, 'currentdate']
            self.previous_date = None
        else:
            self.current_date = None
            self.previous_date = None
        
        # Reset tracking variables
        self.prev_net_worth = self.INITIAL_ACCOUNT_BALANCE
        self.action_taken = 0
        self.returns_history = []
        # Initialize action history with balanced actions to prevent initial bias
        self.action_history = [self.BUYTHRESHOLD+0.1, self.SELLTHRESHOLD-0.1, 0.0, self.BUYTHRESHOLD+0.1, self.SELLTHRESHOLD-0.1, 0.0] * 3  # Mix of buy, sell, hold
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
        np.random.seed(seed)
        return [seed]

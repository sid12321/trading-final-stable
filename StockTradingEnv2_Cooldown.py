# Import only necessary modules to avoid circular imports
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from parameters import INITIAL_ACCOUNT_BALANCE, COST_PER_TRADE, BUYTHRESHOLD, SELLTHRESHOLD

class StockTradingEnv2Cooldown(gym.Env):
    """A stock trading environment with trading cooldown period to focus on regime changes"""
    
    # Class variables for tracking actions across all instances
    action_dict = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
    amount_dict = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
    
    def __init__(self, df, NLAGS=5, NUMVARS=4, MAXIMUM_SHORT_VALUE=INITIAL_ACCOUNT_BALANCE,
                 INITIAL_ACCOUNT_BALANCE=INITIAL_ACCOUNT_BALANCE, MAX_STEPS=20000, 
                 finalsignalsp=[], INITIAL_NET_WORTH=INITIAL_ACCOUNT_BALANCE, 
                 INITIAL_SHARES_HELD=0, COST_PER_TRADE=COST_PER_TRADE,
                 BUYTHRESHOLD=BUYTHRESHOLD, SELLTHRESHOLD=SELLTHRESHOLD,
                 COOLDOWN_PERIOD=5):  # New parameter for cooldown period
        super(StockTradingEnv2Cooldown, self).__init__()
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
        
        # Trading cooldown mechanism
        self.COOLDOWN_PERIOD = COOLDOWN_PERIOD  # Steps to wait after a trade
        self.steps_since_last_trade = COOLDOWN_PERIOD  # Start ready to trade
        self.trades_blocked_by_cooldown = 0  # Track how many trades were blocked
        self.regime_trades = 0  # Track trades that happen after cooldown
        
        # Track current date for daily position liquidation
        self.current_date = None
        self.previous_date = None
        
        # Track daily performance for end-of-day rewards
        self.daily_start_net_worth = INITIAL_NET_WORTH
        self.daily_liquidation_reward = 0.0
        
        # Track price trends for regime detection
        self.price_history = []
        self.regime_change_detected = False

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.float32(np.array([-1, 0])), high=np.float32(np.array([1, 1])), dtype=np.float32)

        # Add cooldown indicator to observation space (NUMVARS+1 to include cooldown status)
        self.observation_space = spaces.Box(
            low=np.float32(0), high=np.float32(1), shape=(self.NUMVARS+1, self.NLAGS+1), dtype=np.float32)

    def _next_observation(self):
        # Get the stock data points for the last NLAGS periods
        data_slices = []
        for sig in self.finalsignalsp:
            if sig in self.df.columns:
                data_slices.append(self.df.loc[(self.current_step-self.NLAGS): self.current_step, sig].values)
            else:
                # If signal not found, use zeros
                data_slices.append(np.zeros(self.NLAGS + 1))
        
        # Stack all signals
        if len(data_slices) > 0:
            obs = np.stack(data_slices)
        else:
            # Fallback if no signals
            obs = np.zeros((self.NUMVARS, self.NLAGS + 1))
        
        # Add cooldown status as an additional feature
        # Normalize cooldown to [0, 1] where 1 means ready to trade
        cooldown_status = min(self.steps_since_last_trade / self.COOLDOWN_PERIOD, 1.0)
        cooldown_feature = np.full((1, self.NLAGS+1), cooldown_status)
        
        # Combine original observations with cooldown feature
        obs = np.vstack([obs, cooldown_feature])
        
        return obs

    def _detect_regime_change(self):
        """Detect if we're in a regime change (significant price movement)"""
        if len(self.price_history) < 10:
            return False
        
        # Calculate short-term and long-term trends
        recent_prices = self.price_history[-5:]
        older_prices = self.price_history[-10:-5]
        
        recent_mean = np.mean(recent_prices)
        older_mean = np.mean(older_prices)
        
        # Calculate percentage change
        pct_change = abs((recent_mean - older_mean) / older_mean)
        
        # Regime change if price moved more than 2%
        return pct_change > 0.02

    def _take_action(self, action):
        current_price = self.df.loc[self.current_step, "vwap2"]
        prev_net_worth = self.net_worth
        
        # Track price for regime detection
        self.price_history.append(current_price)
        if len(self.price_history) > 20:
            self.price_history = self.price_history[-20:]  # Keep last 20 prices
        
        # Check for regime change
        self.regime_change_detected = self._detect_regime_change()
        
        action_type = action[0]
        amount = max(0.1, min(1.0, action[1]))  # Clamp amount between 0.1 and 1.0
        
        # Check if we're in cooldown period
        can_trade = self.steps_since_last_trade >= self.COOLDOWN_PERIOD
        
        # If trying to trade during cooldown, block it
        if not can_trade and (action_type >= BUYTHRESHOLD or action_type <= SELLTHRESHOLD):
            self.trades_blocked_by_cooldown += 1
            action_type = 0  # Force HOLD during cooldown
            # Small penalty for trying to trade during cooldown
            cooldown_penalty = -0.01
        else:
            cooldown_penalty = 0
        
        # Calculate transaction costs (0.1% per trade)
        transaction_cost_rate = self.COST_PER_TRADE
        
        trade_executed = False
        
        if action_type >= BUYTHRESHOLD and can_trade:  
            max_affordable = self.balance / current_price
            shares_to_buy = int(max_affordable * amount)
            
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                transaction_cost = cost * transaction_cost_rate
                total_cost = cost + transaction_cost
                
                if total_cost <= self.balance:
                    self.balance -= total_cost
                    self.shares_held += shares_to_buy
                    StockTradingEnv2Cooldown.action_dict['BUY'] += 1
                    StockTradingEnv2Cooldown.amount_dict['BUY'] += amount
                    trade_executed = True
                    
                    # Track if this was a regime trade
                    if self.regime_change_detected:
                        self.regime_trades += 1
                    
        elif action_type <= SELLTHRESHOLD and can_trade:  # Sell threshold
            max_sellable = self.shares_held + (self.MAXIMUM_SHORT_VALUE / current_price)
            shares_to_sell = int(max_sellable * amount)
            
            if shares_to_sell > 0:
                revenue = shares_to_sell * current_price
                transaction_cost = revenue * transaction_cost_rate
                net_revenue = revenue - transaction_cost
                
                self.balance += net_revenue
                self.shares_held -= shares_to_sell
                StockTradingEnv2Cooldown.action_dict['SELL'] += 1
                StockTradingEnv2Cooldown.amount_dict['SELL'] += amount
                trade_executed = True
                
                # Track if this was a regime trade
                if self.regime_change_detected:
                    self.regime_trades += 1
        else:
            StockTradingEnv2Cooldown.action_dict['HOLD'] += 1
            StockTradingEnv2Cooldown.amount_dict['HOLD'] += amount
        
        # Update cooldown counter
        if trade_executed:
            self.steps_since_last_trade = 0  # Reset cooldown
        else:
            self.steps_since_last_trade += 1
        
        # Update net worth
        self.net_worth = self.balance + self.shares_held * current_price
        
        # Calculate position details for display
        position_value = self.shares_held * current_price
        position_pct = (position_value / self.net_worth) * 100 if self.net_worth > 0 else 0
        cash_pct = (self.balance / self.net_worth) * 100 if self.net_worth > 0 else 0
        
        # Calculate step reward with cooldown and regime bonuses
        step_pnl = self.net_worth - prev_net_worth
        step_return = step_pnl / self.INITIAL_ACCOUNT_BALANCE
        
        # Base reward
        base_reward = step_return * 100
        
        # Regime trading bonus - reward trades during regime changes
        regime_bonus = 0
        if trade_executed and self.regime_change_detected:
            regime_bonus = abs(step_return) * 50  # Extra reward for trading during regime changes
        
        # Patience bonus - small reward for waiting during non-regime periods
        patience_bonus = 0
        if not trade_executed and not self.regime_change_detected and self.steps_since_last_trade < self.COOLDOWN_PERIOD:
            patience_bonus = 0.02  # Small reward for being patient
        
        self.reward = base_reward + regime_bonus + patience_bonus + cooldown_penalty
        
        # Print detailed step info with cooldown status
        cooldown_status = f"Ready" if can_trade else f"Cooldown: {self.COOLDOWN_PERIOD - self.steps_since_last_trade} steps left"
        regime_status = "REGIME CHANGE" if self.regime_change_detected else "Normal"
        
        print(f"Step: {self.current_step}, Price: {current_price:.2f}, "
              f"Action: {action_type:.2f}, Amount: {amount:.2f}, "
              f"Shares: {self.shares_held:.0f}, Balance: {self.balance:.2f}, "
              f"Net Worth: {self.net_worth:.2f}, Position: {position_pct:.1f}%, "
              f"Cash: {cash_pct:.1f}%, Reward: {self.reward:.4f}, "
              f"Cooldown: {cooldown_status}, Market: {regime_status}")

    def _daily_liquidation(self):
        """Force liquidate all positions at 15:15 (end of trading day)"""
        current_price = self.df.loc[self.current_step, "vwap2"]
        
        # Calculate P&L from liquidation
        liquidation_pnl = 0
        if self.shares_held != 0:
            liquidation_value = self.shares_held * current_price
            self.balance += liquidation_value
            liquidation_pnl = liquidation_value if self.shares_held > 0 else -liquidation_value
            
            print(f"Daily liquidation at 15:15: Closed {self.shares_held:.0f} shares at {current_price:.2f}, P&L: {liquidation_pnl:.2f}")
        
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
        
        # Reset cooldown for new day
        self.steps_since_last_trade = self.COOLDOWN_PERIOD  # Start fresh next day
    
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
        
        print(f"Daily liquidation reward: {self.daily_liquidation_reward:.2f} "
              f"(Daily P&L: {total_daily_pnl:.2f}, Return: {daily_return_pct*100:.2f}%, "
              f"Regime trades today: {self.regime_trades})")

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        # Check if we need to liquidate at end of day
        current_time = pd.to_datetime(self.df.loc[self.current_step, "currentt"])
        self.current_date = current_time.date()
        
        # Check for daily liquidation at 15:15 or later (accounting for possible gaps)
        if current_time.hour == 15 and current_time.minute >= 15:
            if self.previous_date is None or self.current_date != self.previous_date:
                self._daily_liquidation()
        
        self.previous_date = self.current_date
        
        self.current_step += 1
        
        # Add daily liquidation reward to step reward
        total_reward = self.reward + self.daily_liquidation_reward
        self.daily_liquidation_reward = 0  # Reset after adding
        
        done = self.net_worth <= 0 or self.current_step >= len(self.df) - 1 or self.current_step >= self.MAX_STEPS
        
        obs = self._next_observation()
        
        # Enhanced info with cooldown stats
        info = {
            'net_worth': self.net_worth,
            'shares_held': self.shares_held,
            'balance': self.balance,
            'steps_since_trade': self.steps_since_last_trade,
            'trades_blocked': self.trades_blocked_by_cooldown,
            'regime_trades': self.regime_trades,
            'can_trade': self.steps_since_last_trade >= self.COOLDOWN_PERIOD
        }
        
        return obs, total_reward, done, False, info

    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        super().reset(seed=seed)
        
        self.balance = self.INITIAL_ACCOUNT_BALANCE
        self.net_worth = self.INITIAL_NET_WORTH
        self.shares_held = self.INITIAL_SHARES_HELD
        self.reward = 0
        
        # Reset cooldown tracking
        self.steps_since_last_trade = self.COOLDOWN_PERIOD  # Start ready to trade
        self.trades_blocked_by_cooldown = 0
        self.regime_trades = 0
        self.price_history = []
        self.regime_change_detected = False
        
        # Reset date tracking
        self.current_date = None
        self.previous_date = None
        
        # Reset daily performance tracking
        self.daily_start_net_worth = self.INITIAL_NET_WORTH
        self.daily_liquidation_reward = 0.0
        self.daily_returns_history = []
        
        # Set the current step to a random point within the data frame
        self.current_step = np.random.randint(self.NLAGS, min(self.MAX_STEPS, len(self.df) - 1))

        print(f"\nEnvironment reset. Starting at step {self.current_step}")
        print(f"Trading cooldown period: {self.COOLDOWN_PERIOD} steps")
        print("Focus: Trading regime changes, not noise")
        
        return self._next_observation(), {}

    def render(self, mode='human'):
        # Render the environment to the screen
        profit = self.net_worth - self.INITIAL_ACCOUNT_BALANCE
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held}')
        print(f'Net worth: {self.net_worth}')
        print(f'Profit: {profit}')
        print(f'Cooldown status: {self.steps_since_last_trade}/{self.COOLDOWN_PERIOD}')
        print(f'Trades blocked by cooldown: {self.trades_blocked_by_cooldown}')
        print(f'Regime trades executed: {self.regime_trades}')
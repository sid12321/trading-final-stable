#!/usr/bin/env python3
"""
trader.py - Modular trading component

This module handles all trading functionality including:
1. Real-time tick data collection and processing
2. Trading logic and execution
3. Portfolio management and P&L tracking
"""

import os
import sys
import time
import threading
from threading import Thread, Lock
from collections import defaultdict, deque
from datetime import datetime
import numpy as np
import pandas as pd
import gc
import joblib

# Import parameters and utilities
basepath = '/Users/skumar81/Desktop/Personal/trading-final-stable'
os.chdir(basepath)

from parameters import *
from lib import *
from common import *
import kitelogin
from kiteconnect import KiteTicker
from model_trainer import ModelTrainer

class TickDataProcessor:
    """Advanced tick data processor with real-time OHLCV generation"""
    
    def __init__(self):
        self.tick_data = defaultdict(list)
        self.candle_data = defaultdict(lambda: {
            'open': [], 'high': [], 'low': [], 'close': [], 
            'volume': [], 'timestamp': [], 'vwap': []
        })
        self.current_minute_data = defaultdict(dict)
        self.lock = Lock()
        self.tick_counts = defaultdict(int)
        self.last_candle_time = {}
        self.ready = defaultdict(bool)
        self.symbol_token_map = {}
        
        # Enhanced aggregation for technical indicators
        self.technical_cache = defaultdict(dict)
        self.price_history = defaultdict(lambda: deque(maxlen=1000))
        
    def register_symbol(self, symbol, instrument_token):
        """Register symbol-token mapping"""
        self.symbol_token_map[instrument_token] = symbol
        print(f"Registered {symbol} with token {instrument_token}")
        
    def process_tick(self, tick):
        """Process individual tick and update candle data"""
        instrument_token = tick['instrument_token']
        
        with self.lock:
            # Store enhanced tick data
            tick_data = {
                'price': tick['last_price'],
                'quantity': tick['last_quantity'],
                'timestamp': datetime.fromtimestamp(tick['timestamp']),
                'volume': tick.get('volume_traded', tick.get('volume', 0)),
                'oi': tick.get('oi', 0),
                'change': tick.get('change', 0)
            }
            
            self.tick_data[instrument_token].append(tick_data)
            self.tick_counts[instrument_token] += 1
            
            # Update price history for technical calculations
            self.price_history[instrument_token].append(tick['last_price'])
            
            # Get current minute for candle aggregation
            tick_minute = datetime.fromtimestamp(tick['timestamp']).replace(second=0, microsecond=0)
            
            # Initialize or update current minute data
            if (instrument_token not in self.current_minute_data or 
                'minute' not in self.current_minute_data[instrument_token]):
                self._initialize_minute_data(instrument_token, tick_minute, tick)
            elif self.current_minute_data[instrument_token]['minute'] == tick_minute:
                self._update_minute_data(instrument_token, tick)
            else:
                # New minute - generate candle for previous minute
                self._create_candle(instrument_token)
                self._initialize_minute_data(instrument_token, tick_minute, tick)
    
    def _initialize_minute_data(self, instrument_token, tick_minute, tick):
        """Initialize minute data for new minute"""
        self.current_minute_data[instrument_token] = {
            'minute': tick_minute,
            'open': tick['last_price'],
            'high': tick['last_price'], 
            'low': tick['last_price'],
            'close': tick['last_price'],
            'volume': tick['last_quantity'],
            'vwap_numerator': tick['last_price'] * tick['last_quantity'],
            'vwap_denominator': tick['last_quantity'],
            'tick_count': 1,
            'price_sum': tick['last_price'],
            'trades': 1
        }
    
    def _update_minute_data(self, instrument_token, tick):
        """Update existing minute data with new tick"""
        minute_data = self.current_minute_data[instrument_token]
        
        minute_data['high'] = max(minute_data['high'], tick['last_price'])
        minute_data['low'] = min(minute_data['low'], tick['last_price'])
        minute_data['close'] = tick['last_price']
        minute_data['volume'] += tick['last_quantity']
        minute_data['vwap_numerator'] += tick['last_price'] * tick['last_quantity']
        minute_data['vwap_denominator'] += tick['last_quantity']
        minute_data['tick_count'] += 1
        minute_data['price_sum'] += tick['last_price']
        minute_data['trades'] += 1
    
    def _create_candle(self, instrument_token):
        """Create enhanced candle from accumulated tick data"""
        if instrument_token not in self.current_minute_data:
            return
            
        candle = self.current_minute_data[instrument_token]
        
        if candle.get('tick_count', 0) < 1:
            return
        
        # Calculate VWAP for the minute
        vwap = (candle['vwap_numerator'] / candle['vwap_denominator'] 
                if candle['vwap_denominator'] > 0 else candle['close'])
        
        # Append enhanced candle data
        self.candle_data[instrument_token]['open'].append(candle['open'])
        self.candle_data[instrument_token]['high'].append(candle['high'])
        self.candle_data[instrument_token]['low'].append(candle['low'])
        self.candle_data[instrument_token]['close'].append(candle['close'])
        self.candle_data[instrument_token]['volume'].append(candle['volume'])
        self.candle_data[instrument_token]['vwap'].append(vwap)
        self.candle_data[instrument_token]['timestamp'].append(candle['minute'])
        
        # Update technical indicators cache
        self._update_technical_cache(instrument_token)
        
        # Update status
        self.last_candle_time[instrument_token] = candle['minute']
        
        # Mark as ready if we have enough candles
        if len(self.candle_data[instrument_token]['open']) >= MINHORIZON:
            self.ready[instrument_token] = True
            
        # Log candle generation
        symbol = self.symbol_token_map.get(instrument_token, f"Token{instrument_token}")
        print(f"Generated candle for {symbol}: OHLCV({candle['open']:.2f}, {candle['high']:.2f}, "
              f"{candle['low']:.2f}, {candle['close']:.2f}, {candle['volume']}) VWAP: {vwap:.2f}")
    
    def _update_technical_cache(self, instrument_token):
        """Update cached technical indicators"""
        candles = self.candle_data[instrument_token]
        
        if len(candles['close']) < 5:
            return
            
        # Cache commonly used technical indicators
        closes = np.array(candles['close'])
        highs = np.array(candles['high'])
        lows = np.array(candles['low'])
        volumes = np.array(candles['volume'])
        
        self.technical_cache[instrument_token] = {
            'sma_5': np.mean(closes[-5:]) if len(closes) >= 5 else closes[-1],
            'sma_10': np.mean(closes[-10:]) if len(closes) >= 10 else closes[-1],
            'sma_20': np.mean(closes[-20:]) if len(closes) >= 20 else closes[-1],
            'volatility': np.std(closes[-20:]) if len(closes) >= 20 else 0,
            'hl_ratio': (highs[-1] - lows[-1]) / lows[-1] if lows[-1] > 0 else 0,
            'volume_avg': np.mean(volumes[-10:]) if len(volumes) >= 10 else volumes[-1]
        }
    
    def force_candle_creation(self):
        """Force creation of candles for all instruments"""
        with self.lock:
            for instrument_token in list(self.current_minute_data.keys()):
                if 'minute' in self.current_minute_data[instrument_token]:
                    self._create_candle(instrument_token)
    
    def get_candle_dataframe(self, instrument_token):
        """Get enhanced candle data as DataFrame with all technical indicators"""
        with self.lock:
            if instrument_token not in self.candle_data:
                return None
                
            if len(self.candle_data[instrument_token]['open']) < 1:
                return None
            
            # Create enhanced DataFrame with all required technical indicators
            df = pd.DataFrame({
                't': self.candle_data[instrument_token]['timestamp'],
                'o': self.candle_data[instrument_token]['open'],
                'h': self.candle_data[instrument_token]['high'],
                'l': self.candle_data[instrument_token]['low'],
                'c': self.candle_data[instrument_token]['close'],
                'v': self.candle_data[instrument_token]['volume'],
                'vwap': self.candle_data[instrument_token]['vwap']
            })
            
            # Add all derived technical indicators needed by the model
            df['date'] = df['t'].dt.date
            df['vwap2'] = df['vwap']  # Required by model
            df['co'] = (df['c'] - df['o']) / df['o']
            df['dv'] = df['vwap'] * df['v']
            df['scco'] = (df['c'] - df['o']) / (df['h'] - df['l'] + 1e-10)
            df['vscco'] = df['scco'] * df['v']
            df['dvscco'] = df['vwap'] * df['vscco']
            df['hl'] = (df['h'] - df['l']) / df['l']
            df['vhl'] = df['hl'] * df['v']
            df['opc'] = df['o'] / df['c'].shift(1) - 1
            df['dvwap'] = df['vwap'].diff()
            df['d2vwap'] = df['dvwap'].diff()
            df['ddv'] = df['dv'].diff()
            df['d2dv'] = df['ddv'].diff()
            
            # Add lagged returns
            for lag in [1, 2, 3, 5, 7, 13, 17, 19, 23]:
                df[f'lret{lag}'] = df['c'].pct_change(lag)
            
            # Add moving averages and derived features
            for window in [5, 10, 20, 50]:
                df[f'sma{window}'] = df['c'].rolling(window=window).mean()
                df[f'price_vs_sma{window}'] = df['c'] / df[f'sma{window}'] - 1
            
            # Add comprehensive technical indicators
            df['rsi'] = self._calculate_rsi(df['c'])
            df['macd'], df['macd_signal'], df['macd_histogram'] = self._calculate_macd(df['c'])
            df['bb_middle'], df['bb_upper'], df['bb_lower'] = self._calculate_bollinger_bands(df['c'])
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['c'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['volatility'] = df['c'].rolling(window=20).std()
            df['atr'] = self._calculate_atr(df)
            df['williams_r'] = self._calculate_williams_r(df)
            df['momentum'] = df['c'] / df['c'].shift(10) - 1
            df['rate_of_change'] = df['c'].pct_change(10)
            
            # Volume indicators
            df['volume_sma'] = df['v'].rolling(window=20).mean()
            df['volume_ratio'] = df['v'] / df['volume_sma']
            df['price_volume'] = df['c'] * df['v']
            
            return df
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    def _calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return sma, upper, lower
    
    def _calculate_atr(self, df, window=14):
        """Calculate Average True Range"""
        high_low = df['h'] - df['l']
        high_close = np.abs(df['h'] - df['c'].shift())
        low_close = np.abs(df['l'] - df['c'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=window).mean()
    
    def _calculate_williams_r(self, df, window=14):
        """Calculate Williams %R"""
        highest_high = df['h'].rolling(window=window).max()
        lowest_low = df['l'].rolling(window=window).min()
        return -100 * (highest_high - df['c']) / (highest_high - lowest_low)
    
    def is_ready(self, instrument_token):
        """Check if instrument has enough data for trading"""
        with self.lock:
            return self.ready.get(instrument_token, False)
    
    def get_tick_count(self, instrument_token):
        """Get number of ticks received for instrument"""
        with self.lock:
            return self.tick_counts.get(instrument_token, 0)


class TickerHandler:
    """Enhanced ticker handler with robust error handling"""
    
    def __init__(self, api_key, access_token):
        self.api_key = api_key
        self.access_token = access_token
        self.ticker = None
        self.processor = TickDataProcessor()
        self.instrument_tokens = set()
        self.running = False
        self.ticker_thread = None
        self.candle_thread = None
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        
    def register_symbols(self, symbol_token_map):
        """Register symbol to token mappings"""
        for symbol, token in symbol_token_map.items():
            self.processor.register_symbol(symbol, token)
    
    def start(self, instrument_tokens):
        """Start the enhanced ticker"""
        if self.running:
            return
            
        self.instrument_tokens = set(instrument_tokens)
        self._initialize_ticker()
        
        # Start ticker and candle threads
        self.running = True
        self.ticker_thread = Thread(target=self._run_ticker)
        self.ticker_thread.daemon = True
        self.ticker_thread.start()
        
        self.candle_thread = Thread(target=self._run_candle_creator)
        self.candle_thread.daemon = True
        self.candle_thread.start()
        
        print(f"Enhanced ticker started with {len(instrument_tokens)} instruments")
    
    def _initialize_ticker(self):
        """Initialize KiteTicker with enhanced callbacks"""
        self.ticker = KiteTicker(self.api_key, self.access_token)
        
        self.ticker.on_ticks = self.on_ticks
        self.ticker.on_connect = self.on_connect
        self.ticker.on_close = self.on_close
        self.ticker.on_error = self.on_error
        self.ticker.on_reconnect = self.on_reconnect
        self.ticker.on_noreconnect = self.on_noreconnect
    
    def _run_ticker(self):
        """Run ticker with automatic reconnection"""
        while self.running:
            try:
                self.ticker.connect()
            except Exception as e:
                print(f"Ticker connection error: {e}")
                if self.reconnect_attempts < self.max_reconnect_attempts:
                    self.reconnect_attempts += 1
                    print(f"Attempting reconnection {self.reconnect_attempts}/{self.max_reconnect_attempts}")
                    time.sleep(5)
                    self._initialize_ticker()
                else:
                    print("Max reconnection attempts reached. Stopping ticker.")
                    self.running = False
                    break
    
    def _run_candle_creator(self):
        """Enhanced candle creation with adaptive timing"""
        while self.running:
            time.sleep(10)  # Force candle creation every 10 seconds
            try:
                self.processor.force_candle_creation()
            except Exception as e:
                print(f"Error in candle creation: {e}")
    
    def stop(self):
        """Stop the ticker gracefully"""
        if not self.running:
            return
            
        self.running = False
        
        if self.ticker:
            try:
                self.ticker.close()
            except:
                pass
        
        # Join threads with timeout
        for thread in [self.ticker_thread, self.candle_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=2.0)
        
        print("Ticker stopped")
    
    def on_ticks(self, ws, ticks):
        """Enhanced tick processing with error handling"""
        try:
            for tick in ticks:
                self.processor.process_tick(tick)
        except Exception as e:
            print(f"Error processing ticks: {e}")
    
    def on_connect(self, ws, response):
        """Enhanced connection handler"""
        print(f"Ticker connected successfully: {response}")
        try:
            ws.set_mode(ws.MODE_FULL, list(self.instrument_tokens))
            self.reconnect_attempts = 0  # Reset on successful connection
        except Exception as e:
            print(f"Error setting ticker mode: {e}")
    
    def on_close(self, ws, code, reason):
        """Enhanced close handler"""
        print(f"Ticker connection closed: {reason} (Code: {code})")
    
    def on_error(self, ws, code, reason):
        """Enhanced error handler"""
        print(f"Ticker error: {reason} (Code: {code})")
    
    def on_reconnect(self, ws, attempts_count):
        """Enhanced reconnection handler"""
        print(f"Ticker reconnecting: attempt {attempts_count}")
    
    def on_noreconnect(self, ws):
        """Enhanced no-reconnection handler"""
        print("Ticker reconnection failed permanently")
        self.running = False
    
    def get_candle_dataframe(self, instrument_token):
        """Get candle data for specific instrument"""
        return self.processor.get_candle_dataframe(instrument_token)
    
    def is_ready(self, instrument_token):
        """Check if instrument has enough data"""
        return self.processor.is_ready(instrument_token)
    
    def get_tick_count(self, instrument_token):
        """Get tick count for instrument"""
        return self.processor.get_tick_count(instrument_token)


class Trader:
    """Modular trading class"""
    
    def __init__(self, symbols=None, use_trained_models=True):
        self.symbols = symbols or TESTSYMBOLS
        self.use_trained_models = use_trained_models
        
        # Trading data structures
        self.datalist = {}
        self.portfolio = {}
        self.cashposition = {}
        self.positivetrades = {}
        self.negativetrades = {}
        self.symbolsignals = {}
        
        # Tick data handling
        self.ticker_handler = None
        self.instrument_token_map = {}
        self.symbol_token_map = {}
        
        # Model data - will be loaded from trainer if needed
        self.allmodels = {}
        self.lol = {}
        self.qtnorm = {}
        self.rdflistp = {}
        
        # Login data
        self.kite = None
        self.APIKEY = None
        self.access_token_kite_p = None
        self.nse = None
        self.nfo = None
        
    def initialize_login(self):
        """Initialize Kite login and verify connectivity"""
        print("Initializing login...")
        
        # Verify internet
        if not internet():
            sys.exit("No internet")
        
        # Login to Kite
        try:
            self.kite = kitelogin.login_to_kite()
            self.APIKEY = kitelogin.API_KEY
            self.access_token_kite_p = self.kite.access_token
            
            # Test connection
            curllisttry = [self.APIKEY+':'+self.access_token_kite_p, 'https://api.kite.trade/instruments/historical/134657/minute?from=2025-04-21+9:15:00&to=2025-04-21+9:45:00']
            fetchdata, statuscode = getdataforme(curllisttry)
            if statuscode == 200:
                print("Logged in successfully")
            else:
                sys.exit("Login verification failed")
                
        except Exception as e:
            sys.exit(f"Could not login: {e}")
        
        # Load instrument data
        self.nse = pd.DataFrame(self.kite.instruments("NSE"))
        self.nfo = pd.DataFrame(self.kite.instruments("NFO"))
    
    def load_models_and_data(self):
        """Load trained models and required data"""
        if self.use_trained_models:
            print("Loading trained models and data...")
            
            # Load models
            self.allmodels = loadallmodels(self.symbols)
            
            # Load normalizations and other data
            for SYM in self.symbols:
                try:
                    self.qtnorm[SYM] = joblib.load(f'{basepath}/models/{SYM}qt.joblib')
                    print(f"Loaded normalization for {SYM}")
                except:
                    print(f"Normalization doesn't exist for {SYM}, model may not be available")
            
            # Load training data for signal extraction (if needed)
            for SYM in self.symbols:
                for prefix in ["final"]:
                    try:
                        rdf = pd.read_csv(f"{basepath}/traindata/{prefix}mldf{SYM}.csv")
                        rdf = rdf.drop(['t'], axis=1)
                        rdf = rdf.head(len(rdf) - 1)
                        self.rdflistp[SYM+prefix] = rdf
                        
                        # Extract signals
                        df = self.rdflistp[SYM+prefix]
                        if "Unnamed: 0" in df.columns:
                            df = df.drop(["Unnamed: 0"], axis=1)
                        df['currentt'] = pd.to_datetime(df['currentt'])
                        df['currentdate'] = df['currentt'].dt.date
                        finalsignalsp = df.columns[~df.columns.isin(['currentt', 'currento', 'currentdate', 'vwap2'])].tolist()
                        self.lol[SYM] = finalsignalsp
                        
                    except Exception as e:
                        print(f"Error loading data for {SYM}: {e}")
            
            gc.collect()
    
    def setup_instrument_mapping(self):
        """Setup instrument token mappings for tick data"""
        print("Setting up instrument token mappings...")
        
        for SYM in self.symbols:
            try:
                instrument_data = self.nse[self.nse['tradingsymbol'] == SYM]
                if not instrument_data.empty:
                    token = instrument_data['instrument_token'].iloc[0]
                    self.instrument_token_map[SYM] = token
                    self.symbol_token_map[token] = SYM
                    print(f"Mapped {SYM} to token {token}")
                else:
                    print(f"Warning: No instrument found for {SYM}")
            except Exception as e:
                print(f"Error mapping {SYM}: {e}")
        
        print(f"Successfully mapped {len(self.instrument_token_map)} instruments")
    
    def setup_tick_data_handler(self):
        """Setup the tick data handler"""
        if not self.instrument_token_map:
            print("No instrument tokens available. Cannot setup tick handler.")
            return False
        
        try:
            self.ticker_handler = TickerHandler(self.APIKEY, self.access_token_kite_p)
            self.ticker_handler.register_symbols(self.instrument_token_map)
            self.ticker_handler.start(list(self.instrument_token_map.values()))
            return True
        except Exception as e:
            print(f"Error setting up tick handler: {e}")
            return False
    
    def wait_for_tick_data(self):
        """Wait for sufficient tick data before starting trading"""
        print("Waiting for initial tick data...")
        
        max_wait_cycles = 30  # Wait up to 5 minutes
        wait_cycle = 0
        
        while wait_cycle < max_wait_cycles:
            ready_count = 0
            total_ticks = 0
            
            for SYM in self.symbols:
                token = self.instrument_token_map.get(SYM)
                if token and self.ticker_handler.is_ready(token):
                    ready_count += 1
                if token:
                    total_ticks += self.ticker_handler.get_tick_count(token)
            
            print(f"Tick data status: {ready_count}/{len(self.symbols)} symbols ready, "
                  f"{total_ticks} total ticks received")
            
            if ready_count == len(self.symbols):
                print("All symbols have sufficient tick data. Starting trading!")
                return True
            
            time.sleep(10)
            wait_cycle += 1
        
        print(f"Warning: Only {ready_count}/{len(self.symbols)} symbols ready after waiting")
        return ready_count > 0
    
    def process_symbol_tick_data(self, SYM):
        """Process a symbol using real-time tick data"""
        token = self.instrument_token_map.get(SYM)
        if not token:
            print(f"No instrument token for {SYM}")
            return False
        
        # Check if we have enough tick data
        if not self.ticker_handler.is_ready(token):
            print(f"Not enough tick data for {SYM} yet (ticks: {self.ticker_handler.get_tick_count(token)})")
            return False
        
        # Get real-time candle data from tick processor
        df = self.ticker_handler.get_candle_dataframe(token)
        if df is None or len(df) < MINHORIZON:
            print(f"Insufficient candle data for {SYM}: {len(df) if df is not None else 0} candles")
            return False
        
        # Store the tick-derived data
        self.datalist[SYM] = df
        
        print(f"✓ {SYM}: {len(df)} candles from {self.ticker_handler.get_tick_count(token)} ticks")
        return True
    
    def initialize_trading_session(self):
        """Initialize trading session with positions and market setup"""
        print("Initializing trading session...")
        
        # Reset positions
        self.portfolio, self.cashposition, self.datalist, self.startingnetvalue, self.positivetrades, self.negativetrades = resetpositions()
        
        # Setup market timing
        if FAKE:
            FROMDATE, LATESTDATE, prevtime, curtime = setupformarketfake(YR=2025, MO=4, DA=21)
        else:
            FROMDATE, LATESTDATE, prevtime, curtime = setupformarket()
        
        self.marketstart, self.marketend = getstartendvalid(curtime, MINHORIZON)
        
        print("Trading session initialized")
    
    def run_trading_loop(self):
        """Main trading loop"""
        print("Starting tick-based trading loop...")
        
        try:
            while True:
                current_time = datetime.now()
                print(f"Trading cycle: {current_time.strftime('%H:%M:%S')}")
                
                # Check market hours
                if ((current_time.hour >= 16) or 
                    ((current_time.hour == 15) and (current_time.minute >= 28))):
                    print("Market closed. Exiting trading loop.")
                    break
                
                # Process each symbol with tick data
                successful_processing = 0
                
                for SYM in self.symbols:
                    try:
                        print(f"Processing {SYM}...")
                        
                        # Use tick-derived data
                        if not self.process_symbol_tick_data(SYM):
                            continue
                        
                        successful_processing += 1
                        
                        # Generate technical features and trade
                        if self.datalist[SYM].shape[0] >= MINHORIZON:
                            prefix = "final"
                            retframe = createallvars(SYM, self.datalist[SYM], self.rdflistp[SYM+prefix])
                            intendedprice = float(self.datalist[SYM].iloc[-1]["c"])
                            
                            # Model prediction
                            action, quantity, trade_size = modelscore(
                                retframe, SYM, prefix, self.rdflistp[SYM+prefix], self.lol[SYM],
                                self.allmodels[SYM+prefix], self.qtnorm[SYM], self.cashposition[SYM],
                                self.cashposition[SYM] + self.portfolio[SYM] * intendedprice, self.portfolio[SYM]
                            )
                            
                            # Trade validation and execution
                            if trade_size > 0:
                                trade_size = np.floor(trade_size)
                            elif trade_size < 0:
                                trade_size = np.floor(trade_size) + 1
                            else:
                                trade_size = 0
                            
                            checkifvalid = validation(self.portfolio, self.cashposition, trade_size, intendedprice, SYM)
                            if (not checkifvalid) and (trade_size != 0):
                                trade_size = correcttillvalid(self.portfolio, self.cashposition, trade_size, intendedprice, SYM)
                            
                            if trade_size != 0:
                                if trade_size > 0:
                                    self.positivetrades[SYM] += 1
                                else:
                                    self.negativetrades[SYM] += 1
                                
                                if FAKE:
                                    response = transmitactionfake(SYM, self.APIKEY, self.access_token_kite_p, trade_size, intendedprice)
                                else:
                                    response = transmitaction(SYM, self.APIKEY, self.access_token_kite_p, trade_size, intendedprice)
                                
                                print("Updating portfolio for " + SYM)
                                self.portfolio, self.cashposition = updateportfolio(SYM, response, self.portfolio, self.cashposition)
                                
                                # Enhanced logging with tick count
                                token = self.instrument_token_map.get(SYM)
                                tick_count = self.ticker_handler.get_tick_count(token) if token else 0
                                print(f"Trade executed for {SYM}: {trade_size} @ {intendedprice:.2f} "
                                      f"(from {tick_count} ticks)")
                        
                    except Exception as e:
                        print(f"Error processing {SYM}: {e}")
                        continue
                
                # Update P&L calculation
                try:
                    finalprices = {}
                    total_ticks = 0
                    
                    for SS in self.symbols:
                        if SS in self.datalist.keys() and len(self.datalist[SS]) > 0:
                            finalprices[SS] = self.datalist[SS].iloc[-1]['c']
                            token = self.instrument_token_map.get(SS)
                            if token:
                                total_ticks += self.ticker_handler.get_tick_count(token)
                    
                    if finalprices:
                        realtimepnl = (
                            sum([self.cashposition[SS] * (1 if (self.positivetrades[SS] + self.negativetrades[SS]) > 0 else 0) 
                                 for SS in finalprices.keys()]) +
                            sum([finalprices[SS] * self.portfolio[SS] for SS in finalprices.keys()]) -
                            self.startingnetvalue * len(finalprices) / len(self.portfolio)
                        )
                        
                        print(f"PNL: {realtimepnl:.2f} | Processed: {successful_processing}/{len(self.symbols)} | "
                              f"Total ticks: {total_ticks}")
                    
                except Exception as e:
                    print(f"Error calculating P&L: {e}")
                
                # Adaptive sleep based on market activity
                time.sleep(5 if successful_processing > 0 else 10)

        except KeyboardInterrupt:
            print("Trading interrupted by user")
        except Exception as e:
            print(f"Critical error in trading loop: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Cleanup tick data handler
            print("Cleaning up tick data handler...")
            if self.ticker_handler:
                self.ticker_handler.stop()
    
    def finalize_trading_session(self):
        """Finalize trading session and generate reports"""
        print("Finalizing trading session...")
        
        # Kill positions if not fake
        if not FAKE:
            curtime = datetime.now()
            if (curtime.hour == 15 and curtime.minute >= 28):
                killpositions(self.symbols, self.portfolio, self.cashposition)
        
        # Generate final reports
        try:
            finalprices = {SS: self.datalist[SS].iloc[-1]['c'] for SS in self.symbols if SS in self.datalist.keys()}
            prettyprintpnl(finalprices, self.portfolio, self.cashposition, self.startingnetvalue, self.positivetrades, self.negativetrades)
            
            # Additional tick data statistics
            print("\n" + "="*60)
            print("TICK DATA STATISTICS")
            print("="*60)
            
            for SYM in self.symbols:
                token = self.instrument_token_map.get(SYM)
                if token and self.ticker_handler:
                    tick_count = self.ticker_handler.get_tick_count(token)
                    ready_status = "✓" if self.ticker_handler.is_ready(token) else "✗"
                    candle_count = len(self.datalist[SYM]) if SYM in self.datalist else 0
                    print(f"{SYM:12} | Ticks: {tick_count:6} | Candles: {candle_count:3} | Ready: {ready_status}")
            
            print("="*60)
            
        except Exception as e:
            print(f"Error in final reporting: {e}")
        
        print("Trading session completed!")
    
    def run_full_trading_session(self):
        """Run complete trading session"""
        print("Starting full trading session...")
        
        # Step 1: Initialize login
        self.initialize_login()
        
        # Step 2: Load models and data
        self.load_models_and_data()
        
        # Step 3: Initialize trading session
        self.initialize_trading_session()
        
        # Step 4: Setup tick data infrastructure
        self.setup_instrument_mapping()
        
        if not self.setup_tick_data_handler():
            print("Failed to setup tick data handler. Exiting.")
            sys.exit("Tick data setup failed")
        
        # Step 5: Wait for initial tick data
        if not self.wait_for_tick_data():
            print("Insufficient tick data received. Continuing with available data...")
        
        # Step 6: Run trading loop
        self.run_trading_loop()
        
        # Step 7: Finalize session
        self.finalize_trading_session()

if __name__ == "__main__":
    # Run the full trading session
    trader = Trader()
    trader.run_full_trading_session()
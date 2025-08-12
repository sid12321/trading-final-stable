#SID: Implement a learning rate schedule 

# Suppress warnings before importing torch
import warnings
import os
warnings.filterwarnings('ignore', message='CUDA initialization: Unexpected error')
warnings.filterwarnings('ignore', category=UserWarning, module='torch.cuda')
# Enable CUDA - comment out to disable
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Fix CUDA library path for JAX (Linux-specific, commented out for Mac)
# if 'LD_LIBRARY_PATH' in os.environ:
#     os.environ['LD_LIBRARY_PATH'] = f"/usr/lib/x86_64-linux-gnu:{os.environ['LD_LIBRARY_PATH']}"
# else:
#     os.environ['LD_LIBRARY_PATH'] = "/usr/lib/x86_64-linux-gnu"

basepath = '/Users/skumar81/Desktop/Personal/trading-final-stable'
tmp_path = basepath+"/tmp/sb3_log/"
check_path = basepath+"/tmp/checkpoints/"
tensorboard_log_path = basepath+"/tmp/tensorboard_logs/"
METRICS_FILE = 'custom_metrics.txt'
METRICS_FILE_PATH = basepath + '/tmp/sb3_log/' + METRICS_FILE
CHECKPOINT_DIR = basepath + '/tmp/checkpoints/'

import torch
import multiprocessing as mp

# Manual device selection flag (set to "cuda", "mps", "cpu", or None for automatic)
# Examples:
# FORCE_DEVICE = "cuda"  # Force CUDA GPU usage
# FORCE_DEVICE = "mps"   # Force Apple Metal usage
# FORCE_DEVICE = "cpu"   # Force CPU usage
# FORCE_DEVICE = None    # Automatic detection (default)
FORCE_DEVICE = 'cpu'  # Set to "cuda", "mps", "cpu" to override automatic detection

# Detect optimal device and core count
if FORCE_DEVICE is not None:
    # Manual device selection
    DEVICE = FORCE_DEVICE.lower()
    if DEVICE == "cuda":
        if torch.cuda.is_available():
            N_GPUS = torch.cuda.device_count()
            GPU_NAME = torch.cuda.get_device_name(0)
            GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / 1e9
            if os.environ.get('TRADING_INIT_PRINTED') != '1':
                print(f"FORCE_DEVICE: Using CUDA - {N_GPUS}x {GPU_NAME} ({GPU_MEMORY:.1f}GB each)")
        else:
            print("WARNING: CUDA forced but not available, falling back to CPU")
            DEVICE = "cpu"
            N_GPUS = 0
    elif DEVICE == "mps":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            N_GPUS = 1
            import platform
            import subprocess
            try:
                chip_info = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                         capture_output=True, text=True).stdout.strip()
                GPU_NAME = f"Apple Metal ({chip_info})"
            except:
                GPU_NAME = "Apple Metal Performance Shaders"
            try:
                mem_bytes = int(subprocess.run(['sysctl', '-n', 'hw.memsize'], 
                                              capture_output=True, text=True).stdout.strip())
                GPU_MEMORY = mem_bytes / 1e9
            except:
                GPU_MEMORY = 0
            if os.environ.get('TRADING_INIT_PRINTED') != '1':
                print(f"FORCE_DEVICE: Using Metal - {GPU_NAME} ({GPU_MEMORY:.1f}GB unified memory)")
        else:
            print("WARNING: MPS forced but not available, falling back to CPU")
            DEVICE = "cpu"
            N_GPUS = 0
    elif DEVICE == "cpu":
        N_GPUS = 0
        if os.environ.get('TRADING_INIT_PRINTED') != '1':
            print("FORCE_DEVICE: Using CPU (manual override)")
    else:
        print(f"WARNING: Unknown FORCE_DEVICE value '{FORCE_DEVICE}', using automatic detection")
        FORCE_DEVICE = None
        
# Automatic detection if FORCE_DEVICE is None or invalid
if FORCE_DEVICE is None:
    if torch.cuda.is_available():
        DEVICE = "cuda"
        N_GPUS = torch.cuda.device_count()
        GPU_NAME = torch.cuda.get_device_name(0)
        GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / 1e9
        # Only print once to avoid spam in multiprocessing
        if os.environ.get('TRADING_INIT_PRINTED') != '1':
            print(f"CUDA GPU(s) detected: {N_GPUS}x {GPU_NAME} ({GPU_MEMORY:.1f}GB each)")
            print("Optimized for RTX 4080 and similar high-end GPUs")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        DEVICE = "mps"
        N_GPUS = 1  # MPS uses unified memory, count as 1 device
        # Get Mac system info
        import platform
        import subprocess
        try:
            # Get chip info
            chip_info = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                     capture_output=True, text=True).stdout.strip()
            GPU_NAME = f"Apple Metal ({chip_info})"
        except:
            GPU_NAME = "Apple Metal Performance Shaders"
        
        # MPS uses unified memory, report system memory
        try:
            mem_bytes = int(subprocess.run(['sysctl', '-n', 'hw.memsize'], 
                                          capture_output=True, text=True).stdout.strip())
            GPU_MEMORY = mem_bytes / 1e9
        except:
            GPU_MEMORY = 0
        
        if os.environ.get('TRADING_INIT_PRINTED') != '1':
            print(f"Metal GPU detected: {GPU_NAME} ({GPU_MEMORY:.1f}GB unified memory)")
            print("Optimized for Apple Silicon (M1/M2/M3)")
    else:
        DEVICE = "cpu"
        N_GPUS = 0
        if os.environ.get('TRADING_INIT_PRINTED') != '1':
            print("Using CPU training")

# Multi-core settings optimized for different platforms
if DEVICE == "cuda":
    # Simple, effective settings for trading workloads
    N_CORES = min(mp.cpu_count(), 12)  # Apple Silicon has performance/efficiency cores
    N_ENVS = min(mp.cpu_count(), 12)  # Match core count for Metal
    SIGNAL_OPTIMIZATION_WORKERS = min(N_CORES, 12)  # Match environment count
elif DEVICE == "mps":
    # Apple Silicon optimized settings
    N_CORES = min(mp.cpu_count(), 1)  # Apple Silicon has performance/efficiency cores
    N_ENVS = min(mp.cpu_count(), 1)  # Match core count for Metal
    SIGNAL_OPTIMIZATION_WORKERS = min(N_CORES, 12)  # Optimize for unified memory
else:
    # CPU optimized settings - increased for 32-core system with 61GB RAM
    N_CORES = min(mp.cpu_count(), 12)  # Optimize cores for stability
    N_ENVS = 12  # Balance environments for performance vs stability
    SIGNAL_OPTIMIZATION_WORKERS = min(N_CORES, 12)  # Increase workers

OMP_NUM_THREADS = N_CORES  # OpenMP threads

# Set PyTorch optimizations
torch.set_num_threads(N_CORES)

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True  # Optimize for RTX GPUs
    torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for RTX 30/40 series
    torch.backends.cudnn.allow_tf32 = True  # Enable TF32 for cuDNN operations
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True  # Better FP16 performance
    # Enable mixed precision autocast for better GPU utilization
    torch.backends.cuda.enable_flash_sdp(True)  # Flash attention if available
elif DEVICE == "mps":
    # MPS-specific optimizations
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.8'  # Memory management
    os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.3'
    os.environ['PYTORCH_MPS_ALLOCATOR_POLICY'] = 'garbage_collection'
    # MPS doesn't support all CUDA features yet, but it's improving

# Signal optimization parameters
OPTIMIZATION_METHOD = 'simulated_annealing'  # or 'mcmc'
USE_GPU_ACCELERATION = True
USE_PARALLEL_OPTIMIZATION = True
MAX_OPTIMIZATION_ITERATIONS = 2000 if DEVICE in ["cuda", "mps"] else 1000
MCMC_SAMPLES = 2000 if DEVICE in ["cuda", "mps"] else 1000

# Only print once to avoid spam in multiprocessing
if os.environ.get('TRADING_INIT_PRINTED') != '1':
    print(f"Multi-core setup: {N_CORES} cores, {N_ENVS} environments, Device: {DEVICE}")
    print(f"Signal optimization: {SIGNAL_OPTIMIZATION_WORKERS} workers, GPU acceleration: {USE_GPU_ACCELERATION}")
    os.environ['TRADING_INIT_PRINTED'] = '1'

POLICY_KWARGS = {
    'activation_fn': 'ReLU',  # Will be converted to torch.nn.ReLU in modeltrain
    'net_arch': {
        'pi': [256, 256],  # OPTIMIZED network architecture
        'vf': [256, 256]   # OPTIMIZED network architecture
    },
    'ortho_init': True  # Orthogonal initialization for better training
}

#Symbols
SYMLIST = ["BPCL","HDFCLIFE","BRITANNIA","HEROMOTOCO","INDUSINDBK","APOLLOHOSP","WIPRO","TATASTEEL","BHARTIARTL","ITC","HINDUNILVR","POWERGRID"]
TESTSYMBOLS = SYMLIST[:1] 

# Training Pipeline Control Flags
# These flags control which parts of the pipeline to execute

# PREPROCESS: Download and process new market data from Kite API
# Set to False to skip data preprocessing and use existing historical data
# Useful for training-only mode or when using cached data
PREPROCESS = os.environ.get('PREPROCESS', 'True').lower() in ('true', '1', 'yes')

# TRAINMODEL: Train new PPO models using historical data
# Set to False to skip model training (e.g., for data preprocessing only)
TRAINMODEL = os.environ.get('TRAINMODEL', 'True').lower() in ('true', '1', 'yes')

# NEWMODELFLAG: Create new models from scratch vs loading existing ones
# True = Train new models, False = Continue training existing models
NEWMODELFLAG = os.environ.get('NEWMODEL', 'False').lower() in ('true', '1', 'yes')

# DETERMINISTIC: Use deterministic policy for consistent results during evaluation
DETERMINISTIC = os.environ.get('DETERMINISTIC', 'True').lower() in ('true', '1', 'yes')

# GENERATEPOSTERIOR: Generate posterior trading analysis after training
GENERATEPOSTERIOR = os.environ.get('GENERATEPOSTERIOR', 'True').lower() in ('true', '1', 'yes')

# POSTERIORPLOTS: Generate trading visualization plots
POSTERIORPLOTS = os.environ.get('POSTERIORPLOTS', 'True').lower() in ('true', '1', 'yes')
FAKE = True
GENOPTSIG = False

#Base data scope
BENCHMARKHORIZON = 375*5
HORIZONDAYS = 60
INITIAL_ACCOUNT_BALANCE = 100000
TRAIN_MAX = 0.75
NLAGS = 5 #This is the TS consumed by the model 
MAXIMUM_SHORT_VALUE = INITIAL_ACCOUNT_BALANCE
TOPN = 0
MAXITERPOSTERIOR = 1 #Deterministic it's all the same 
NQUANTILES = 5
LAGS = [1,2,3,5,7,13,17,19,23] #Thes are lags taken in at each step
LAGCOLS = [f'lret{lag}' for lag in LAGS]
GENERICS = ['vwap', 'dv', 'c','o','h','l','v', 'co', 'scco', 'vscco', 'dvscco', \
'hl', 'vhl','opc', 'dvwap', 'd2vwap', 'ddv', 'd2dv', 'h5scco', 'h5vscco', 'h5dvscco','codv',\
'macd', 'macd_signal', 'macd_histogram', 'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position',\
'rsi', 'rsi_oversold', 'rsi_overbought', 'stoch_k', 'stoch_d', 'atr', 'williams_r',\
'sma5', 'sma10', 'sma20', 'sma50', 'price_vs_sma5', 'price_vs_sma10', 'price_vs_sma20',\
'volume_sma', 'volume_ratio', 'price_volume', 'momentum', 'rate_of_change', 'volatility',\
'vol_spike', 'bear_signal', 'oversold_extreme', 'bb_squeeze', 'lower_highs', 'lower_lows',\
'support_break', 'hammer_pattern', 'volume_divergence', 'trend_alignment_bear', 'market_fear',\
'bull_signal', 'overbought_extreme', 'bb_expansion', 'higher_highs', 'higher_lows',\
'resistance_break', 'morning_star', 'volume_confirmation', 'trend_alignment_bull', 'market_greed',\
'pivot_high_3', 'pivot_low_3', 'pivot_high_5', 'pivot_low_5', 'pivot_high_7', 'pivot_low_7',\
'pivot_strength', 'local_max', 'local_min', 'swing_high', 'swing_low']
HISTORICAL = ['ndv', 'nmomentum']
QCOLS = ['q' + x for x in GENERICS + LAGCOLS] 
ALLSIGS = GENERICS + QCOLS + LAGCOLS + HISTORICAL

#PPO details
MAX_STEPS = 100  # Shorter episodes since we're training on entire dataset with daily liquidation
BASEMODELITERATIONS = 3000000  # More iterations for full dataset training
REDUCEDMCMCITERATIONS = BASEMODELITERATIONS//4
MEANREWARDTHRESHOLD = 0.2 #Corresponds to a more realistic 2% return for minute-based trading
BUYTHRESHOLD = 0.3
SELLTHRESHOLD = -0.3
COST_PER_TRADE = 0
MAXITERREPEAT = 1  # Fewer iterations since we train on full dataset
CLIP_RANGE = 0.2  # Standard clip range for policy updates
CLIP_RANGE_VF = 0.2  # Conservative value function clipping
VF_COEF = 0.25  # Standard value function coefficient for full dataset learning
VALUE_LOSS_BOUND = 1.0  # Bound value loss to [-1, 1] to prevent training instability
GAMMA = 0.99  # Standard discount factor for intraday trading with daily liquidation
RETRAIN = False
NORAD = True
ENTROPY_BOUND = 1.0  # Bound entropy loss to [-1, 1] to prevent training instability
MAX_GRAD_NORM = 0.25  # Gradient clipping for stability #SID
VERBOSITY = 0
TRAINREPS = 1
LOGFREQ = min(BASEMODELITERATIONS//2,500)  # Less frequent logging for long episodes
STATS_WINDOW_SIZE = 100  # Larger window for full dataset training stability

#To be Optimized - Realistic Trading System Settings
GLOBALLEARNINGRATE = 5e-5  # Standard learning rate
N_EPOCHS = 1  # OPTIMIZED epochs
ENT_COEF = 0.005  # Standard entropy coefficient
N_STEPS = 3072  # OPTIMIZED rollout buffer (3072 is optimal)
TARGET_KL = 0.02  # Standard KL divergence
GAE_LAMBDA = 0.8  # Standard GAE lambda
BATCH_SIZE = 512   # OPTIMIZED batch size
USE_SDE = True  # Disable SDE #SID
SDE_SAMPLE_FREQ = 4  # Sample new noise every 4 steps (only used if USE_SDE=True)
TOTAL_TIMESTEPS = N_STEPS * N_ENVS  # Total steps across all environments

# Learning Rate Scheduling Parameters
USE_LR_SCHEDULE = True  # Enable learning rate scheduling
INITIAL_LR = 1e-3  # Starting learning rate
FINAL_LR = 1e-5    # Final learning rate
LR_SCHEDULE_TYPE = "exponential"  # Options: "linear", "exponential", "cosine"

# Entropy Coefficient Scheduling Parameters
USE_ENT_SCHEDULE = True  # Enable entropy coefficient scheduling
INITIAL_ENT_COEF = 0.05  # Starting entropy coefficient
FINAL_ENT_COEF = 0.005   # Final entropy coefficient

# Target KL Scheduling Parameters
USE_TARGET_KL_SCHEDULE = True  # Enable target KL scheduling
INITIAL_TARGET_KL = 0.1  # Starting target KL
FINAL_TARGET_KL = 0.01    # Final target KL

# Simple GPU settings
USE_MIXED_PRECISION = True  # Disable for stability in trading systems
GPU_MEMORY_GROWTH = True  # Allow gradual GPU memory allocation
OPTIMIZE_FOR_GPU = False  # Focus on CPU optimization for trading

signalhorizons = {x:y for x,y in zip(ALLSIGS,[1]*len(ALLSIGS))}
for col in QCOLS:
  signalhorizons[col] = 5 #You need at least 5 rows to compute the generic quantile (5 bars including)
  if col.startswith('h5'):
    signalhorizons[col] = 9 #5 bars + last bar has a tail of 4 - so it becomes 5 + 4 = 9
for lag in LAGS:
  signalhorizons['lret'+str(lag)] = lag + 1
  signalhorizons['qlret'+str(lag)] = lag + 5 #
MINHORIZON = max(signalhorizons.values()) + NLAGS + 1

# Kite API credentials - centralized configuration
API_KEY = "wh7m5jcdtj4g57oh"
API_SECRET = "2owm89v2qjd9mx4sngodejq9hdelfwxj"
USER_ID = "PN1089"
PASSWORD = "Pillowbat123$"
TOTP_KEY = "LC4GFGV5GZTYGFPXQRQ6J4YO2IU6ZL56"



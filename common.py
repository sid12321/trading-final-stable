
import shutil
import gc

from lib import *
from bounded_entropy_ppo import BoundedEntropyPPO
from optimized_signal_generator import generate_optimized_signals_for_dataframe
from parameters import (
    VALUE_LOSS_BOUND, OPTIMIZATION_METHOD, USE_PARALLEL_OPTIMIZATION, 
    USE_GPU_ACCELERATION, SIGNAL_OPTIMIZATION_WORKERS
)

def cleanup_gpu_memory(aggressive=False):
    """Clean up GPU memory to prevent memory leaks"""
    if torch.cuda.is_available():
        if aggressive:
            # More aggressive cleanup for better GPU utilization
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
        else:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        # MPS memory cleanup
        if aggressive:
            # Force garbage collection for Metal
            torch.mps.empty_cache()
            torch.mps.synchronize() if hasattr(torch.mps, 'synchronize') else None
        else:
            torch.mps.empty_cache()
    gc.collect()

def optimize_gpu_settings():
    """Optimize GPU settings for maximum utilization without memory fragmentation"""
    if torch.cuda.is_available():
        # Enable optimizations for RTX GPUs
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
        
        # Memory optimizations - prevent fragmentation
        torch.backends.cuda.enable_flash_sdp(True)
        
        # Set conservative memory allocation to prevent fragmentation
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        
        # Use 80% of GPU memory to leave room for multiple processes
        max_memory_allocated = int(total_memory * 0.80)
        
        # Set memory fraction to prevent out-of-memory errors
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(0.8)
        
        # Enable memory pool for better allocation
        if hasattr(torch.cuda, 'memory'):
            torch.cuda.empty_cache()
        
        print(f"GPU optimization enabled for {torch.cuda.get_device_name(device)}")
        print(f"Total GPU memory: {total_memory / 1e9:.1f}GB")
        print(f"Target memory usage: {max_memory_allocated / 1e9:.1f}GB (80% to prevent fragmentation)")
        
        return True
    elif torch.backends.mps.is_available():
        # MPS (Metal) optimizations for Apple Silicon
        import subprocess
        
        # Get system memory info
        try:
            mem_bytes = int(subprocess.run(['sysctl', '-n', 'hw.memsize'], 
                                          capture_output=True, text=True).stdout.strip())
            total_memory = mem_bytes
            
            # Get chip info
            chip_info = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                     capture_output=True, text=True).stdout.strip()
        except:
            total_memory = 0
            chip_info = "Apple Silicon"
        
        # Clear MPS cache
        torch.mps.empty_cache()
        
        print(f"Metal optimization enabled for {chip_info}")
        print(f"Total unified memory: {total_memory / 1e9:.1f}GB")
        print(f"Using Apple Metal Performance Shaders (MPS)")
        
        return True
    return False

def get_gpu_stats():
    """Get GPU/Metal utilization stats, cross-platform"""
    import platform
    import subprocess
    
    if torch.cuda.is_available():
        # NVIDIA GPU stats
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw', 
                                   '--format=csv,noheader,nounits'], 
                                   capture_output=True, text=True)
            if result.returncode == 0:
                line = result.stdout.strip().split('\n')[0]
                parts = line.split(', ')
                return {
                    'device_type': 'cuda',
                    'utilization': int(parts[0]),
                    'memory_used': int(parts[1]),
                    'memory_total': int(parts[2]),
                    'temperature': int(parts[3]),
                    'power': float(parts[4])
                }
        except Exception:
            pass
    elif torch.backends.mps.is_available() and platform.system() == 'Darwin':
        # Mac Metal stats
        try:
            # Get memory info
            vm_stat = subprocess.run(['vm_stat'], capture_output=True, text=True)
            
            # Get system activity (simplified)
            top_result = subprocess.run(['top', '-l', '1', '-n', '0'], 
                                      capture_output=True, text=True)
            
            # Parse basic system stats (simplified for Mac)
            return {
                'device_type': 'mps',
                'utilization': None,  # Metal doesn't expose this easily
                'memory_used': None,  # Would need more complex parsing
                'memory_total': None,
                'temperature': None,  # Not easily accessible
                'power': None,
                'status': 'Metal GPU Active'
            }
        except Exception:
            pass
    
    return {
        'device_type': 'cpu',
        'utilization': None,
        'memory_used': None,
        'memory_total': None,
        'temperature': None,
        'power': None,
        'status': 'CPU Only'
    }

def clean_tensorboard_logs():
    """Clean TensorBoard logs directory before starting new training"""
    import shutil
    import os
    from parameters import tensorboard_log_path
    
    print("🧹 Cleaning TensorBoard logs...")
    
    if os.path.exists(tensorboard_log_path):
        try:
            # Remove all contents but keep the directory
            for filename in os.listdir(tensorboard_log_path):
                filepath = os.path.join(tensorboard_log_path, filename)
                if os.path.isfile(filepath) or os.path.islink(filepath):
                    os.unlink(filepath)
                elif os.path.isdir(filepath):
                    shutil.rmtree(filepath)
            print("✅ TensorBoard logs cleaned")
        except Exception as e:
            print(f"⚠️  Warning: Could not clean TensorBoard logs: {e}")
    else:
        # Create directory if it doesn't exist
        os.makedirs(tensorboard_log_path, exist_ok=True)
        print("📁 Created TensorBoard logs directory")
    
    return tensorboard_log_path

#########Classes

class OnnxableSB3Policy(torch.nn.Module):
    def __init__(self, policy: BasePolicy):
        super().__init__()
        self.policy = policy

    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # NOTE: Preprocessing is included, but postprocessing
        # (clipping/inscaling actions) is not,
        # If needed, you also need to transpose the images so that they are channel first
        # use deterministic=False if you want to export the stochastic policy
        # policy() returns `actions, values, log_prob` for PPO
        return self.policy(observation, deterministic=DETERMINISTIC)

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout


#########Functions

def getprevcur(prevtime,curtime,datatime):
    if datatime is not None:
      prevtime = datatime+timedelta(minutes=1)
    curtime = datetime.now()
    if curtime<prevtime:
      curtime = prevtime
    hour1 = prevtime.hour
    min1 = prevtime.minute
    hour2 = curtime.hour
    min2 = curtime.minute
    t1 = str(hour1) + ":" + "{:02d}".format(min1)
    t2 = str(hour2) + ":" + "{:02d}".format(min2)
    return t1,t2


def getprevcurfake(prevtime,curtime,datatime):
    if datatime is not None:
      prevtime = datatime+timedelta(minutes=1)
    if (curtime > (prevtime + timedelta(minutes=1))):
      print("")
    else:
      curtime = prevtime 
    hour1 = prevtime.hour
    min1 = prevtime.minute
    hour2 = curtime.hour
    min2 = curtime.minute
    t1 = str(hour1) + ":" + "{:02d}".format(min1)
    t2 = str(hour2) + ":" + "{:02d}".format(min2)
    return t1,t2

def validation(portfolio,cashposition,trade_size,intendedprice,SYM):
  if trade_size>0:
    if cashposition[SYM]<=0:
      return False
    if ((cashposition[SYM] - trade_size*intendedprice) > 0 ):
      return True
    else:
      return False
  elif trade_size<0:  
    if(-((trade_size+portfolio[SYM])*intendedprice)<INITIAL_ACCOUNT_BALANCE):
      return True
    else:
      return False
  else:
    return False

def correcttillvalid(portfolio,cashposition,trade_size,intendedprice,SYM):
  if trade_size>0:
    if cashposition[SYM]<=0:
      return 0
    trade_size_max = np.floor(cashposition[SYM]/intendedprice)
    if trade_size<trade_size_max:
      return trade_size
    else:
      return trade_size_max
  elif trade_size<0:
    if (-((trade_size+portfolio[SYM])*intendedprice)>INITIAL_ACCOUNT_BALANCE):
      return 0
    trade_size_min = -INITIAL_ACCOUNT_BALANCE/intendedprice - portfolio[SYM]
    if trade_size>trade_size_min:
      return trade_size
    else:
      return trade_size_min
  
def internet(host="8.8.8.8", port=53, timeout=3):
  """
  Host: 8.8.8.8 (google-public-dns-a.google.com)
  OpenPort: 53/tcp
  Service: domain (DNS/TCP)
  """
  try:
    socket.setdefaulttimeout(timeout)
    socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
    return True
  except socket.error as ex:
    print(ex)
    return False

def getdataforme(x):
  headers = {
      'X-Kite-Version': '3',
      'Authorization': 'token ' + x[0],
  }
  
  response = requests.get(
      x[1],
      headers=headers,
  )

  if response.status_code==200:
    responsejson = response.json()
    return responsejson['data']['candles'],response.status_code
  else:
    print("No data")
    return None, 403

def pctrank(x):
  x = np.array(x)
  i = x.argsort().argmax() + 1
  n = len(x)
  return i/n

def createallvars(SYM,datalistdf,rdflistpdf):
  ff = datalistdf.tail(MINHORIZON).reset_index(drop=True)
  
  ff['opc'] = (ff['o'] - ff['c'].shift(1))/ff['c'].shift(1)
  ff['dvwap'] = (ff['vwap'] - ff['vwap'].shift(1))/ff['vwap'].shift(1)
  ff['d2vwap'] = (ff['dvwap'] - ff['dvwap'].shift(1))/(ff['dvwap'].shift(1) + 1e-10)
  ff['ddv'] = (ff['dv'] - ff['dv'].shift(1))/(ff['dv'].shift(1) + 1e-10)
  ff['d2dv'] = (ff['ddv'] - ff['ddv'].shift(1))/(ff['ddv'].shift(1) + 1e-10)
  ff['h5scco'] = ff['scco'].rolling(5).mean()
  ff['h5vscco'] = ff['vscco'].rolling(5).mean()
  ff['h5dvscco'] = ff['dvscco'].rolling(5).mean()
  ff['codv'] = (ff['c'] - ff['o'])/(ff['dv'] + 1e-10)
  
  for delays in LAGS:
    ff['lret'+str(delays)] = (ff['vwap'] - ff['vwap'].shift(delays))/ff['vwap'].shift(delays)

  for col in QCOLS:
    col = col[1:]
    ff["q"+col] = ts_rank(ff[col], 5)
  
  ff['ndv'] = ff['dv'].apply(lambda x: historicpct(x,"dv",SYM,rdflistpdf,BENCHMARKHORIZON))  #SID: Convert this to an expanding apply
  ff['nmomentum'] = ff['codv'].apply(lambda x: historicpct(x,"codv",SYM,rdflistpdf,BENCHMARKHORIZON)) #SID: Convert this to an expanding apply
  
  if len(intersect(ALLSIGS,ff.columns))!=len(ALLSIGS):
    stop("All signals not present. Stopping execution")
  
  ff = ff.tail(NLAGS+1)
  if check_inf_nan(ff[lol[SYM]]): 
    stop("Nans or infs detected.") 
  
  return ff

def check_inf_nan(df):
    numeric_df = df.select_dtypes(include=np.number)
    has_inf_nan = (~numeric_df.isin([np.inf, -np.inf, np.nan])).all().all()
    return not has_inf_nan

def historicpct(x,var,SYM,rdflistpdf,HORIZON):
  xapp = rdflistpdf[var][-(HORIZON-1):].tolist() + [x]  #Correction for including the current value 
  res = pctrank(xapp)
  return res

def resetpositions():
  portfolio = {}
  cashposition = {}
  positivetrades = {}
  negativetrades = {}
  for SYM in TESTSYMBOLS:
    cashposition[SYM] = INITIAL_ACCOUNT_BALANCE
    portfolio[SYM] = 0
    positivetrades[SYM] = 0
    negativetrades[SYM] = 0
  datalist = {}
  startingnetvalue = sum(cashposition.values())
  return portfolio,cashposition,datalist,startingnetvalue,positivetrades,negativetrades

#tracks current stock value and stock position - cash not tracked here - first value is wrong - put that in portfolio tracking - only 2 is sufficient 
def updateportfolio(SYM,response,portfolio,cashposition):
  if SYM in portfolio.keys():
    portfolio[SYM] = portfolio[SYM] + float(response['trade_size'])
  else:
    portfolio[SYM] = float(response['trade_size'])
  cashposition[SYM] = cashposition[SYM] - float(response['trade_size'])*float(response['price'])
  return portfolio,cashposition


def getpositions(APIKEY,access_token_kite_p):
  commandp = 'curl "https://api.kite.trade/portfolio/positions" -H "X-Kite-Version: 3" -H "Authorization: token '+APIKEY+':'+access_token_kite_p+'"'
  statusp, outputp = subprocess.getstatusoutput(commandp) 
  match = re.search(r"\{",outputp)
  if match:
    outputp = outputp[match.start():] 
    outputp = json.loads(outputp)
  else:
    print("No matches found")
  positions = outputp['data']['net']
  positionframe = pd.DataFrame({'SYMBOL':[item['tradingsymbol'] for item in positions],'POSITION':[item['buy_quantity']-item['sell_quantity'] for item in positions],'ITOKEN':[item['instrument_token'] for item in positions]})
  return positionframe

def killpositions(x,portfolio,cashposition):
  allpositions = getpositions(APIKEY,access_token_kite_p) #should give position and current position as dataframe 
  for SYM in allpositions['SYMBOL']:
    try:
      trade_size = allpositions[allpositions['SYMBOL']==SYM]["POSITION"].iloc[0]
      response = transmitaction(SYM,APIKEY,access_token_kite_p,-trade_size,None) 
      portfolio,cashposition = updateportfolio(SYM,response,portfolio,cashposition) #Probably need to hold off on this to be once every few cycles 
    except:
      print(SYM + " position is 0")


      
def initialparse(x):
  datadf = pd.DataFrame(x)
  datadf.columns = ["t","o","h","l","c","v"]
  datadf['t'] = pd.to_datetime(datadf['t'], format="%Y-%m-%dT%H:%M:%S%z")
  datadf['date'] = datadf['t'].apply(lambda x: x.date())
  dates = datadf.pop('date')
  datadf.insert(0, 'date', dates)
  datadf['vwap'] = (2*datadf.l + 2*datadf.h + 3*datadf.c + 3*datadf.o)/10
  datadf = datadf.assign(vwap2 = datadf['vwap'])
  datadf['co'] = (datadf.c - datadf.o)/datadf.o
  datadf['dv'] = datadf.vwap * datadf.v
  datadf['scco'] = (datadf.c - datadf.o)/(datadf.h - datadf.l + 1e-10)
  datadf['vscco'] = datadf.scco * datadf.v
  datadf['dvscco'] = datadf.vwap*datadf.vscco
  datadf['hl'] = (datadf.h - datadf.l)/datadf.l
  datadf['vhl'] = datadf.hl * datadf.v
  return datadf

def updatedf(df1,df2): #test method
  try:
    if((type(df1)==type(df2))&(df1.shape[1]==df2.shape[1])):
      return(pd.concat([df1,df2],axis=0,ignore_index=True))
    else:
      return df2
  except:
    print("df1")
    print(df1)
    print("df2")
    print(df2)
    return df2

def unload_module(module_name):
    if module_name in sys.modules:
        del sys.modules[module_name]
        print(f"Module '{module_name}' unloaded.")
    else:
        print(f"Module '{module_name}' not found.")


def checkdepth(SYM,signalprice,trade_size,APIKEY,access_token_kite_p):
  if signalprice is None:
    return True
  else:
    commandd = 'curl "https://api.kite.trade/quote?i=NSE:'+SYM+'" -H "X-Kite-Version: 3" -H "Authorization: token '+APIKEY+':'+access_token_kite_p+'"'
    statusd, outputd = subprocess.getstatusoutput(commandd) 
    match = re.search(r"\{",outputd)
    if match:
      outputd = outputd[match.start():] 
      outputd = json.loads(outputd)
    else:
      return False
    buysell = outputd['data']['NSE:'+SYM]['depth']
    if trade_size>0:
      buy = buysell['buy']  
      if ((buy[0]['quantity']>trade_size)&(buy[0]['price']<(signalprice+1))):
        return True
      else:
        return False
    elif trade_size<0:
      sell = buysell['sell']
      if ((sell[0]['quantity']>trade_size)&(sell[0]['price']>(signalprice-1))):
        return True
      else:
        return False
    else:
      return True

def readpolicyfromzip(filename):
  # Open the zip file
  with zipfile.ZipFile(filename, 'r') as myzip:
      # Get the list of files inside the zip archive
      file_list = myzip.namelist()
  
      # Assuming the file you want to load is 'your_file.pth'
      if 'policy.pth' in file_list:
          # Open the specific file inside the zip archive in binary read mode
          with myzip.open('policy.pth', 'r') as myfile:
              # Read the content of the file
              buffer = io.BytesIO(myfile.read())
              # Load the file using torch.load
              model = torch.load(buffer,map_location=torch.device('cpu'), weights_only=True)
          # 'data' now contains the loaded object from the file
          print("Model loaded successfully!")
      else:
          print("Model not found in the zip archive.")
  return model

def create_signal(values):
    """Create trading signals based on quantiles"""
    values = np.array(values)
    lq = np.nanquantile(values, 0.3)
    uq = np.nanquantile(values, 0.7)
    signal = np.zeros_like(values)
    signal[values <= lq] = 1
    signal[values >= uq] = -1
    signal[np.isnan(values)] = 0
    return signal

# def create_command(syms, api_key, access_token):
#     """Create command for Kite API to fetch quotes"""
#     processed_syms = []
#     for sym in syms:
#         sym = sym.replace("&", "%26")
#         sym = sym.replace("NSE:NIFTY ", "NSE:NIFTY%20")
#         processed_syms.append(sym)
#     
#     command1 = 'curl "https://api.kite.trade/quote/ltp?i='
#     command2 = "&i=".join(processed_syms)
#     command3 = '" -H "X-Kite-Version: 3" -H "Authorization:token '
#     command4 = f"{api_key}:{access_token}"
#     
#     return command1 + command2 + command3 + command4

# def process_time(from_date, to_date, hm1=None, hm2=None):
#     """Process time for historical data API request"""
#     if hm1 is None or hm2 is None:
#         fromto = f"from={from_date}+09:15:00&to={to_date}+15:30:00"
#     else:
#         fromto = f"from={from_date}+{hm1}:00&to={to_date}+{hm2}:00"
#     
#     return fromto

def get_historical_curl(sym, api_key, access_token, from_time, to_time, hm1=None, hm2=None):
    """Create curl command for historical data"""
    fromto = process_time(from_time, to_time, hm1, hm2)
    
    # Find instrument token
    nse_filtered = nse[nse['tradingsymbol'] == sym]
    if nse_filtered.empty:
        print("No such symbol")
        return None
    
    inst_token = nse_filtered['instrument_token'].iloc[0]
    
    command1 = f'curl -s "https://api.kite.trade/instruments/historical/{inst_token}/minute?'
    command2 = f'{fromto}"'
    command3 = ' -H "X-Kite-Version: 3" -H "Authorization:token '
    command4 = f"{api_key}:{access_token}"
    
    return command1 + command2 + command3 + command4

def get_historical_curl2(sym, api_key, access_token, from_time, to_time, hm1=None, hm2=None,nse=None):
    """Alternative method for historical data URL creation"""
    fromto = process_time(from_time, to_time, hm1, hm2)
    
    # Find instrument token
    nse_filtered = nse[nse['tradingsymbol'] == sym]
    if nse_filtered.empty:
        print("No such symbol")
        return None
    
    inst_token = nse_filtered['instrument_token'].iloc[0]
    
    command1 = f'https://api.kite.trade/instruments/historical/{inst_token}/minute?'
    command2 = fromto
    command4 = f"{api_key}:{access_token}"
    
    return [command4, command1 + command2]

def simulate_trades_on_day(vwap, action):
    """Simulate trades based on signals"""
    position = 0
    pnl = 0
    
    for i in range(len(action) - 1):
        if action[i] > 0:
            position += 1
            pnl -= vwap[i + 1]
        elif action[i] < 0:
            position -= 1
            pnl += vwap[i + 1]
    
    pnl += position * vwap[-1]
    position = 0
    
    return [pnl, position]

def delay(series, n):
    """Implement R's delay function"""
    return series.shift(n)

def rollapplyr(series, window, func=np.sum, partial=True):
    """Implement R's rollapplyr function"""
    if partial:
        return series.rolling(window, min_periods=1).apply(func)
    else:
        return series.rolling(window).apply(func)

def create_command(syms, api_key, access_token):
    """Create command for Kite API to fetch quotes"""
    processed_syms = []
    for sym in syms:
        sym = sym.replace("&", "%26")
        sym = sym.replace("NSE:NIFTY ", "NSE:NIFTY%20")
        processed_syms.append(sym)
    
    command1 = 'curl "https://api.kite.trade/quote/ltp?i='
    command2 = "&i=".join(processed_syms)
    command3 = '" -H "X-Kite-Version: 3" -H "Authorization:token '
    command4 = f"{api_key}:{access_token}"
    
    return command1 + command2 + command3 + command4

def process_time(from_date, to_date, hm1=None, hm2=None):
    """Process time for historical data API request"""
    if hm1 is None or hm2 is None:
        fromto = f"from={from_date}+09:15:00&to={to_date}+15:30:00"
    else:
        fromto = f"from={from_date}+{hm1}:00&to={to_date}+{hm2}:00"
    
    return fromto

def get_historical_curl(sym, api_key, access_token, from_time, to_time, hm1=None, hm2=None):
    """Create curl command for historical data"""
    fromto = process_time(from_time, to_time, hm1, hm2)
    
    # Find instrument token
    nse_filtered = nse[nse['tradingsymbol'] == sym]
    if nse_filtered.empty:
        print("No such symbol")
        return None
    
    inst_token = nse_filtered['instrument_token'].iloc[0]
    
    command1 = f'curl -s "https://api.kite.trade/instruments/historical/{inst_token}/minute?'
    command2 = f'{fromto}"'
    command3 = ' -H "X-Kite-Version: 3" -H "Authorization:token '
    command4 = f"{api_key}:{access_token}"
    
    return command1 + command2 + command3 + command4

# def simulate_trades_on_day(vwap, action):
#     """Simulate trades based on signals"""
#     position = 0
#     pnl = 0
#     
#     for i in range(len(action) - 1):
#         if action[i] > 0:
#             position += 1
#             pnl -= vwap[i + 1]
#         elif action[i] < 0 and position > 0:
#             position -= 1
#             pnl += vwap[i + 1]
#     
#     # Zero out position at end of day - assume you sell at VWAP at last time step
#     if position > 0:
#         pnl += position * vwap[-1]
#         position = 0
#     
#     if position < 0:
#         pnl += position * vwap[-1]
#         position = 0
#     
#     return [pnl, position]

def delay(series, n):
    """Implement R's delay function"""
    return series.shift(n)

def ts_rank(series, window):
    """Implement time series rank function"""
    return series.rolling(window).apply(lambda x: stats.rankdata(x)[-1] / len(x), raw=True)

def rollapplyr(series, window, func=np.sum, partial=True):
    """Implement R's rollapplyr function"""
    if partial:
        return series.rolling(window, min_periods=1).apply(func)
    else:
        return series.rolling(window).apply(func)
      
      

def transmitaction(SYM,APIKEY,access_token_kite_p,trade_size,signalprice):
  if(~checkdepth(SYM,signalprice,trade_size,APIKEY,access_token_kite_p)): 
    finaloutput = {'trade_size':0,'price':signalprice}
    return finaloutout
  trade_size = int(trade_size)
  if trade_size>0:
    ACTION = "BUY"
  else:
    ACTION = "SELL"
  trade_size = abs(trade_size)
  command = 'curl https://api.kite.trade/orders/regular \
    -H "X-Kite-Version: 3" \
    -H "Authorization: token ' + APIKEY+":"+access_token_kite_p+'" \
    -d "tradingsymbol=' + SYM + '" \
    -d "exchange=NSE" \
    -d "transaction_type='+ ACTION +'" \
    -d "order_type=MARKET" \
    -d "quantity='+str(trade_size)+'" \
    -d "product=MIS" \
    -d "validity=DAY"'
    
  status, output = subprocess.getstatusoutput(command)
  match = re.search(r"\{",output)
  if match:
    output = output[match.start():] 
    output = json.loads(output)
  else:
    finaloutput = {'trade_size':0,'price':signalprice}
    return finaloutout
  
  orderid = '0'
  if output['status'] == 'success':
    orderid = output['order_id']
    finaloutput = {}
    finaloutput['trade_size'] = trade_size
    commandp = 'curl "https://api.kite.trade/orders/'+orderid+'" -H "Authorization: token '+APIKEY+':'+access_token_kite_p+'"'
    statusp, outputp = subprocess.getstatusoutput(commandp) 
    match = re.search(r"\{",output)
    if match:
      output = output[match.start():] 
      output = json.loads(output)
      finaloutput['price'] = output['average_price']
      finaloutput['trade_size'] = output['filled_quantity']
    else:
      finaloutput = {'trade_size':0,'price':signalprice}
    return finaloutput
  else:
    finaloutput = {'trade_size':0,'price':signalprice}
    return finaloutput

def transmitactionfake(SYM,APIKEY,access_token_kite_p,trade_size,signalprice):
  response = '{"trade_size":"'+str(trade_size)+'","price":"'+str(signalprice)+'"}'
  return json.loads(response) 


def normalizedf(df,SYM,finalsignalsp,qtsym):
  dft = df.copy()
  dft[finalsignalsp] = pd.DataFrame(qtsym.transform(dft[finalsignalsp].to_numpy()),columns=finalsignalsp)
  return dft

def modelscore(df,SYM,prefix,rdflistpdf,lolsym,model,qtsym,STARTING_ACCOUNT_BALANCE,STARTING_NET_WORTH, STARTING_SHARES_HELD): 
    finalsignalsp = lolsym 
    NUMVARS = len(finalsignalsp)
    df = normalizedf(df,SYM,finalsignalsp,qtsym) #Normalization only here
    env_class = StockTradingEnvOptimized
    env = DummyVecEnv([lambda: env_class(df,NLAGS,NUMVARS,MAXIMUM_SHORT_VALUE,STARTING_ACCOUNT_BALANCE,MAX_STEPS,finalsignalsp,STARTING_NET_WORTH, STARTING_SHARES_HELD)])
    obs = env.reset()
    action,additional1,additional2 = model.run(None, {"input": obs.astype(np.float32)})
    with Capturing() as output:
        obs, rewards, done, info = env.step([action,True])
    return action[0][0],action[0][1],int(float(re.search('Position: (.+?),', output[0]).group(1)))


#Modifying this to be a static sleep time given candles can arrive randomly without explicitly assembling tick data 
def steppingfunction(prevtime,curtime):
  prevtime = curtime
  curtime = datetime.now()
  curtime = datetime(curtime.year, curtime.month, curtime.day, curtime.hour, curtime.minute, curtime.second, 0)
  if(curtime.minute==prevtime.minute):
    #time.sleep(60-curtime.second)
    time.sleep(10)
    curtime = datetime.now()
    curtime = datetime(curtime.year, curtime.month, curtime.day, curtime.hour, curtime.minute, curtime.second, 0)
  return prevtime,curtime

def steppingfunctionfake(prevtime,curtime):
  prevtime = curtime
  curtime = prevtime + timedelta(minutes=1)
  return prevtime,curtime


def getstartendvalid(curtime,MINHORIZON):
  marketstart = datetime(curtime.year, curtime.month, curtime.day, 9, 15, 0, 0)
  marketend = datetime(curtime.year, curtime.month, curtime.day, 15, 30, 0, 0)
  
  if (curtime<=(marketstart+timedelta(minutes=MINHORIZON))):
    diff = (marketstart-curtime).total_seconds() 
    print("Market not yet open. Sleep for "+str(diff))
    time.sleep(diff)
  
  if curtime>marketend:
    sys.exit("Market is done for the day")

  return marketstart,marketend  

def setupformarket():
  curtime = datetime.now()
  curtime = datetime(curtime.year, curtime.month, curtime.day, curtime.hour, curtime.minute, curtime.second, 0)
  prevtime = datetime(curtime.year, curtime.month, curtime.day, 9, 15, 0, 0)
  FROMDATE = curtime.strftime("%Y-%m-%d")
  LATESTDATE = FROMDATE
  return FROMDATE,LATESTDATE,prevtime,curtime

def setupformarketfake(YR=2025,MO=4,DA=21):
  curtime = datetime(YR, MO, DA, 9, 45, 0, 0)
  prevtime = datetime(YR, MO, DA, 9, 15, 0, 0)
  FROMDATE = curtime.strftime("%Y-%m-%d")
  LATESTDATE = FROMDATE
  return FROMDATE,LATESTDATE,prevtime,curtime


def loadallmodels(SYMLISTGIVEN=TESTSYMBOLS):
  allmodels = {}
  for SYM in SYMLISTGIVEN:
    for prefix in ["final"]:
      if(prefix=="final"):
        modelfilename = SYM + 'localmodel'
        modelfilenameonnx = SYM + 'localmodel' + '.onnx'
        print(modelfilename)
        try:
          model = PPO.load(basepath + "/models/" + modelfilename, device="cpu")
          onnx_policy = OnnxableSB3Policy(model.policy)
          observation_size = model.observation_space.shape
          dummy_input = torch.randn(1, *observation_size)
          torch.onnx.export(
              onnx_policy,
              dummy_input,
              basepath + "/models/" + modelfilenameonnx,
              opset_version=17,
              input_names=["input"],
          )          
          allmodels[SYM+prefix] = ort.InferenceSession(basepath + "/models/" + modelfilenameonnx)
        except:
          print("Model could not be loaded for SYM: " + SYM + prefix)
          allmodels[SYM+prefix] = None
      else:
        modelfilename = 'globalmodel' + '.zip'
        modelfilenameonnx = 'globalmodel' + '.onnx'
        print(modelfilename)
        try:
          model = PPO.load(basepath + "/models/" + modelfilename, device="cpu")
          onnx_policy = OnnxableSB3Policy(model.policy)
          observation_size = model.observation_space.shape
          dummy_input = torch.randn(1, *observation_size)
          torch.onnx.export(
              onnx_policy,
              dummy_input,
              basepath + "/models/" + modelfilenameonnx,
              opset_version=17,
              input_names=["input"],
          )          
          print("Loaded global model")
          allmodels[SYM+prefix] = ort.InferenceSession(basepath + "/models/" + modelfilenameonnx)
        except:
          print("Model could not be loaded for SYM: " + SYM + prefix)
          allmodels[SYM+prefix] = None
  return allmodels 


def preprocess(kite): 
  LATEST_DATE = date.today() - timedelta(days=1)
  FROM_DATE = LATEST_DATE - timedelta(days=HORIZONDAYS)
  
  nse = pd.DataFrame(kite.instruments("NSE"))
  nfo = pd.DataFrame(kite.instruments("NFO"))
  
  # Get instruments
  # Initialize results list
  reslist = {}
  
  # Fetch historical data for each symbol
  for SYM in TESTSYMBOLS:
      print("Fetching historical data for " + SYM)
      
      # Get historical data directly from kite API instead of using curl
      from_date_str = FROM_DATE.strftime("%Y-%m-%d")
      to_date_str = LATEST_DATE.strftime("%Y-%m-%d")
      
      try:
          # Get instrument token
          inst_token = nse[nse['tradingsymbol'] == SYM]['instrument_token'].iloc[0]
          
          # Fetch data using kite API
          data = kite.historical_data(
              instrument_token=inst_token,
              from_date=f"{from_date_str} 09:15:00",
              to_date=f"{to_date_str} 15:30:00",
              interval="minute"
          )
          
          reslist[SYM] = data
      except Exception as e:
          print(f"Error fetching data for {SYM}: {e}")
          reslist[SYM] = None
  # Process each symbol's data
  for SYM in TESTSYMBOLS:
      print("Processing data for " + SYM)
      
      if reslist[SYM] is None:
          print(f"No data for {SYM}, skipping")
          continue
      
      # Convert to pandas DataFrame
      datadf = pd.DataFrame(reslist[SYM])
      
      # Rename columns to match R script
      datadf.rename(columns={
          'date': 't',
          'open': 'o',
          'high': 'h',
          'low': 'l',
          'close': 'c',
          'volume': 'v'
      }, inplace=True)
      
      # Convert timestamp to datetime
      datadf['t'] = pd.to_datetime(datadf['t'])
      
      # Add date column
      datadf['date'] = datadf['t'].dt.date
      
      # Reorder columns to have date first
      cols = datadf.columns.tolist()
      cols.remove('date')
      datadf = datadf[['date'] + cols]
      
      # Calculate VWAP and other metrics
      datadf['vwap'] = (2*datadf['l'] + 2*datadf['h'] + 3*datadf['c'] + 3*datadf['o'])/10
      datadf = datadf.assign(vwap2 = datadf['vwap'])
      datadf['co'] = (datadf['c'] - datadf['o']) / datadf['o']  
      datadf['dv'] = datadf['vwap'] * datadf['v']
      datadf['scco'] = (datadf['c'] - datadf['o']) / (datadf['h'] - datadf['l'] + 1e-10)
      datadf['vscco'] = datadf['v'] * datadf['scco']
      datadf['dvscco'] = datadf['vwap'] * datadf['vscco']
      datadf['hl'] = (datadf['h'] - datadf['l']) / datadf['l']  
      datadf['vhl'] = datadf['hl'] * datadf['v']
      datadf['codv'] = (datadf['c'] - datadf['o'])/(datadf['dv'] + 1e-10)
      
      # Enhanced Technical Indicators
      # MACD (12, 26, 9)
      datadf['ema12'] = datadf['c'].ewm(span=12).mean()
      datadf['ema26'] = datadf['c'].ewm(span=26).mean()
      datadf['macd'] = datadf['ema12'] - datadf['ema26']
      datadf['macd_signal'] = datadf['macd'].ewm(span=9).mean()
      datadf['macd_histogram'] = datadf['macd'] - datadf['macd_signal']
      
      # Bollinger Bands (20 period, 2 std)
      datadf['bb_middle'] = datadf['c'].rolling(window=20).mean()
      datadf['bb_std'] = datadf['c'].rolling(window=20).std()
      datadf['bb_upper'] = datadf['bb_middle'] + (2 * datadf['bb_std'])
      datadf['bb_lower'] = datadf['bb_middle'] - (2 * datadf['bb_std'])
      datadf['bb_width'] = (datadf['bb_upper'] - datadf['bb_lower']) / datadf['bb_middle']
      datadf['bb_position'] = (datadf['c'] - datadf['bb_lower']) / (datadf['bb_upper'] - datadf['bb_lower'])
      
      # RSI (14 period)
      delta = datadf['c'].diff()
      gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
      loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
      rs = gain / loss
      datadf['rsi'] = 100 - (100 / (1 + rs))
      datadf['rsi_oversold'] = (datadf['rsi'] < 30).astype(int)
      datadf['rsi_overbought'] = (datadf['rsi'] > 70).astype(int)
      
      # Stochastic Oscillator (14, 3, 3)
      datadf['stoch_k'] = ((datadf['c'] - datadf['l'].rolling(14).min()) / 
                          (datadf['h'].rolling(14).max() - datadf['l'].rolling(14).min())) * 100
      datadf['stoch_d'] = datadf['stoch_k'].rolling(3).mean()
      
      # Average True Range (14 period)
      datadf['tr1'] = datadf['h'] - datadf['l']
      datadf['tr2'] = abs(datadf['h'] - datadf['c'].shift(1))
      datadf['tr3'] = abs(datadf['l'] - datadf['c'].shift(1))
      datadf['tr'] = datadf[['tr1', 'tr2', 'tr3']].max(axis=1)
      datadf['atr'] = datadf['tr'].rolling(window=14).mean()
      
      # Williams %R (14 period)
      datadf['williams_r'] = ((datadf['h'].rolling(14).max() - datadf['c']) / 
                             (datadf['h'].rolling(14).max() - datadf['l'].rolling(14).min())) * -100
      
      # Moving Average Convergence/Divergence with different periods
      datadf['sma5'] = datadf['c'].rolling(window=5).mean()
      datadf['sma10'] = datadf['c'].rolling(window=10).mean()
      datadf['sma20'] = datadf['c'].rolling(window=20).mean()
      datadf['sma50'] = datadf['c'].rolling(window=50).mean()
      
      # Price position relative to moving averages
      datadf['price_vs_sma5'] = (datadf['c'] - datadf['sma5']) / datadf['sma5']
      datadf['price_vs_sma10'] = (datadf['c'] - datadf['sma10']) / datadf['sma10']
      datadf['price_vs_sma20'] = (datadf['c'] - datadf['sma20']) / datadf['sma20']
      
      # Volume indicators
      datadf['volume_sma'] = datadf['v'].rolling(window=20).mean()
      datadf['volume_ratio'] = datadf['v'] / datadf['volume_sma']
      datadf['price_volume'] = datadf['co'] * datadf['volume_ratio']
      
      # Momentum indicators
      datadf['momentum'] = datadf['c'] / datadf['c'].shift(10) - 1
      datadf['rate_of_change'] = (datadf['c'] - datadf['c'].shift(12)) / datadf['c'].shift(12) * 100
      
      # Volatility indicators
      datadf['volatility'] = datadf['c'].rolling(window=20).std() / datadf['c'].rolling(window=20).mean()
      
      # Enhanced Bear Market and Bottom Detection Signals
      # VIX-like volatility spike indicator
      datadf['vol_spike'] = datadf['volatility'] / datadf['volatility'].rolling(window=50).mean()
      
      # Market regime indicators
      datadf['bear_signal'] = ((datadf['sma5'] < datadf['sma20']) & 
                               (datadf['sma20'] < datadf['sma50']) &
                               (datadf['rsi'] < 40)).astype(int)
      
      # Bottom detection signals
      datadf['oversold_extreme'] = (datadf['rsi'] < 25).astype(int)
      datadf['bb_squeeze'] = (datadf['bb_width'] < datadf['bb_width'].rolling(50).quantile(0.2)).astype(int)
      
      # Price action signals for downtrends
      datadf['lower_highs'] = ((datadf['h'].shift(1) > datadf['h']) & 
                              (datadf['h'].shift(2) > datadf['h'].shift(1))).astype(int)
      datadf['lower_lows'] = ((datadf['l'].shift(1) > datadf['l']) & 
                             (datadf['l'].shift(2) > datadf['l'].shift(1))).astype(int)
      
      # Market structure break signals
      datadf['support_break'] = ((datadf['c'] < datadf['l'].rolling(20).min()) & 
                                (datadf['c'].shift(1) >= datadf['l'].shift(1).rolling(20).min())).astype(int)
      
      # Reversal signals for bottoms
      datadf['hammer_pattern'] = ((datadf['l'] < datadf['o']) & 
                                 (datadf['c'] > (datadf['o'] + datadf['h']) / 2) &
                                 ((datadf['h'] - datadf['l']) > 3 * abs(datadf['c'] - datadf['o']))).astype(int)
      
      # Volume divergence signals
      datadf['volume_divergence'] = ((datadf['c'] < datadf['c'].shift(5)) & 
                                    (datadf['v'] > datadf['v'].shift(5))).astype(int)
      
      # Multi-timeframe trend alignment
      datadf['trend_alignment_bear'] = ((datadf['price_vs_sma5'] < 0) & 
                                       (datadf['price_vs_sma10'] < 0) &
                                       (datadf['price_vs_sma20'] < 0)).astype(int)
      
      # Market fear indicator (combination of signals)
      datadf['market_fear'] = (datadf['bear_signal'] + datadf['oversold_extreme'] + 
                              datadf['vol_spike'].apply(lambda x: 1 if x > 1.5 else 0) +
                              datadf['support_break'] + datadf['trend_alignment_bear'])
      
      ## BULLISH INDICATORS TO BALANCE BEARISH BIAS ##
      
      # Bull market regime indicator (opposite of bear_signal)
      datadf['bull_signal'] = ((datadf['sma5'] > datadf['sma20']) & 
                               (datadf['sma20'] > datadf['sma50']) &
                               (datadf['rsi'] > 60)).astype(int)
      
      # Top detection signals
      datadf['overbought_extreme'] = (datadf['rsi'] > 75).astype(int)
      datadf['bb_expansion'] = (datadf['bb_width'] > datadf['bb_width'].rolling(50).quantile(0.8)).astype(int)
      
      # Price action signals for uptrends
      datadf['higher_highs'] = ((datadf['h'].shift(1) < datadf['h']) & 
                               (datadf['h'].shift(2) < datadf['h'].shift(1))).astype(int)
      datadf['higher_lows'] = ((datadf['l'].shift(1) < datadf['l']) & 
                              (datadf['l'].shift(2) < datadf['l'].shift(1))).astype(int)
      
      # Market structure break signals for upside
      datadf['resistance_break'] = ((datadf['c'] > datadf['h'].rolling(20).max()) & 
                                   (datadf['c'].shift(1) <= datadf['h'].shift(1).rolling(20).max())).astype(int)
      
      # Bullish reversal signals
      datadf['morning_star'] = ((datadf['h'] > datadf['o']) & 
                               (datadf['c'] < (datadf['o'] + datadf['l']) / 2) &
                               ((datadf['h'] - datadf['l']) > 3 * abs(datadf['c'] - datadf['o']))).astype(int)
      
      # Volume confirmation for uptrends
      datadf['volume_confirmation'] = ((datadf['c'] > datadf['c'].shift(5)) & 
                                      (datadf['v'] > datadf['v'].shift(5))).astype(int)
      
      # Multi-timeframe bull trend alignment
      datadf['trend_alignment_bull'] = ((datadf['price_vs_sma5'] > 0) & 
                                       (datadf['price_vs_sma10'] > 0) &
                                       (datadf['price_vs_sma20'] > 0)).astype(int)
      
      # Market greed indicator (combination of bullish signals)
      datadf['market_greed'] = (datadf['bull_signal'] + datadf['overbought_extreme'] + 
                               datadf['vol_spike'].apply(lambda x: 1 if x < 0.7 else 0) +
                               datadf['resistance_break'] + datadf['trend_alignment_bull'])
      
      ## PIVOT DETECTION SIGNALS FOR TOP/BOTTOM IDENTIFICATION ##
      
      # Pivot High Detection (local maxima) - different lookback periods
      def detect_pivot_high(series, lookback):
          """Detect pivot highs with given lookback period"""
          pivot_high = np.zeros(len(series))
          for i in range(lookback, len(series) - lookback):
              window = series[i - lookback:i + lookback + 1]
              if series[i] == window.max() and series[i] > series[i-1] and series[i] > series[i+1]:
                  pivot_high[i] = 1
          return pivot_high
      
      # Pivot Low Detection (local minima) - different lookback periods  
      def detect_pivot_low(series, lookback):
          """Detect pivot lows with given lookback period"""
          pivot_low = np.zeros(len(series))
          for i in range(lookback, len(series) - lookback):
              window = series[i - lookback:i + lookback + 1]
              if series[i] == window.min() and series[i] < series[i-1] and series[i] < series[i+1]:
                  pivot_low[i] = 1
          return pivot_low
      
      # Apply pivot detection with different lookback periods (3, 5, 7 bars)
      datadf['pivot_high_3'] = detect_pivot_high(datadf['h'], 3)
      datadf['pivot_low_3'] = detect_pivot_low(datadf['l'], 3)
      datadf['pivot_high_5'] = detect_pivot_high(datadf['h'], 5) 
      datadf['pivot_low_5'] = detect_pivot_low(datadf['l'], 5)
      datadf['pivot_high_7'] = detect_pivot_high(datadf['h'], 7)
      datadf['pivot_low_7'] = detect_pivot_low(datadf['l'], 7)
      
      # Pivot Strength - measures how significant a pivot is
      def calculate_pivot_strength(high_series, low_series, lookback=5):
          """Calculate pivot strength based on price range and volume"""
          strength = np.zeros(len(high_series))
          for i in range(lookback, len(high_series) - lookback):
              # Check for pivot high
              if high_series[i] == high_series[i-lookback:i+lookback+1].max():
                  price_range = high_series[i] - low_series[i-lookback:i+lookback+1].min()
                  avg_range = np.mean(high_series[i-lookback:i+lookback+1] - low_series[i-lookback:i+lookback+1])
                  if avg_range > 0:
                      strength[i] = min(price_range / avg_range, 5.0)  # Cap at 5x
              # Check for pivot low  
              elif low_series[i] == low_series[i-lookback:i+lookback+1].min():
                  price_range = high_series[i-lookback:i+lookback+1].max() - low_series[i]
                  avg_range = np.mean(high_series[i-lookback:i+lookback+1] - low_series[i-lookback:i+lookback+1])
                  if avg_range > 0:
                      strength[i] = min(price_range / avg_range, 5.0)  # Cap at 5x
          return strength
      
      datadf['pivot_strength'] = calculate_pivot_strength(datadf['h'], datadf['l'])
      
      # Local Maximum/Minimum detection (simpler version using rolling windows)
      datadf['local_max'] = ((datadf['h'] == datadf['h'].rolling(7, center=True).max()) & 
                             (datadf['h'] > datadf['h'].shift(1)) & 
                             (datadf['h'] > datadf['h'].shift(-1))).astype(int)
      
      datadf['local_min'] = ((datadf['l'] == datadf['l'].rolling(7, center=True).min()) & 
                             (datadf['l'] < datadf['l'].shift(1)) & 
                             (datadf['l'] < datadf['l'].shift(-1))).astype(int)
      
      # Swing High/Low Detection (more robust version considering volume and momentum)
      def detect_swing_points(df, lookback=5):
          """Detect swing highs and lows with volume and momentum confirmation"""
          swing_high = np.zeros(len(df))
          swing_low = np.zeros(len(df))
          
          for i in range(lookback, len(df) - lookback):
              # Swing High: highest high in window + volume confirmation
              if df['h'].iloc[i] == df['h'].iloc[i-lookback:i+lookback+1].max():
                  # Volume confirmation (above average)
                  avg_volume = df['v'].iloc[i-lookback:i+lookback+1].mean()
                  if df['v'].iloc[i] >= avg_volume * 0.8:  # At least 80% of avg volume
                      swing_high[i] = 1
              
              # Swing Low: lowest low in window + volume confirmation  
              if df['l'].iloc[i] == df['l'].iloc[i-lookback:i+lookback+1].min():
                  avg_volume = df['v'].iloc[i-lookback:i+lookback+1].mean()
                  if df['v'].iloc[i] >= avg_volume * 0.8:  # At least 80% of avg volume
                      swing_low[i] = 1
                      
          return swing_high, swing_low
      
      swing_high, swing_low = detect_swing_points(datadf)
      datadf['swing_high'] = swing_high
      datadf['swing_low'] = swing_low
      
      # Drop temporary columns
      datadf.drop(['tr1', 'tr2', 'tr3', 'tr', 'ema12', 'ema26', 'bb_std'], axis=1, inplace=True)
      
      datadf = datadf.sort_values('t')
      
      # Calculate by date groups
      grouped = datadf.groupby('date')
      
      for name, group in grouped:
        group = group.sort_values('t')
        datadf.loc[group.index, 'opc'] = (group['o'] - delay(group['c'], 1)) / delay(group['c'], 1)
        datadf.loc[group.index, 'dvwap'] = (group['vwap'] - delay(group['vwap'], 1)) / delay(group['vwap'], 1)
        datadf.loc[group.index, 'd2vwap'] = (datadf.loc[group.index, 'dvwap'] - delay(datadf.loc[group.index, 'dvwap'], 1)) / (delay(datadf.loc[group.index, 'dvwap'], 1) + 1e-10)
        datadf.loc[group.index, 'ddv'] = (group['dv'] - delay(group['dv'], 1)) / (delay(group['dv'], 1) + 1e-10)
        datadf.loc[group.index, 'd2dv'] = (datadf.loc[group.index, 'ddv'] - delay(datadf.loc[group.index, 'ddv'], 1)) / (delay(datadf.loc[group.index, 'ddv'], 1) + 1e-10)
      
      # Calculate rolling means by date
      for name, group in grouped:
        group = group.sort_values('t')
        # Calculate 5-period rolling means aligned left
        datadf.loc[group.index, 'h5scco'] = group['scco'].rolling(5, min_periods=5).mean()
        datadf.loc[group.index, 'h5vscco'] = group['vscco'].rolling(5, min_periods=5).mean()
        datadf.loc[group.index, 'h5dvscco'] = group['dvscco'].rolling(5, min_periods=5).mean()
      
      # Calculate lagged returns
      for name, group in grouped:
        group = group.sort_values('t')
        for lag in LAGS:
          datadf.loc[group.index, f'lret{lag}'] = (group['vwap'] - delay(group['vwap'], lag)) / delay(group['vwap'], lag)
      
      # Calculate quantile rank for each column
      for col in QCOLS:
        col = col[1:]
        for name, group in grouped:
          group = group.sort_values('t')
          datadf.loc[group.index, f'q{col}'] = ts_rank(group[col], 5)
              
      # Calculate cumulative values by date
      for name, group in grouped:
        group = group.sort_values('t')
        datadf.loc[group.index, 'cdv'] = group['dv'].cumsum()
        datadf.loc[group.index, 'cv'] = group['v'].cumsum()
      
      # Sort by time in ascending order (ensure)
      datadf = datadf.sort_values('t')
      
      # Calculate time series ranks
      datadf['ndv'] = ts_rank(datadf['dv'], BENCHMARKHORIZON) 
      datadf['nmomentum'] = ts_rank(datadf['codv'], BENCHMARKHORIZON) 
      
      #retcolumns = [col for col in datadf.columns if col.startswith('ret')]
      signalcolumns = GENERICS + QCOLS + LAGCOLS + HISTORICAL
      
      # Remove rows with NAs in signal columns
      mldf = datadf.dropna(subset=signalcolumns) 
      
      # Keep dates with enough data points
      date_counts = mldf.groupby('date').size()
      keepdates = date_counts[date_counts >= date_counts.mean()].index 
      mldf = mldf[mldf['date'].isin(keepdates)]
      
      # Add additional columns
      mldf['currentt'] = mldf['t'] 
      mldf['currento'] = mldf['o'] 
      #SID: At some point eliminate the above 
      
      pnlframe = pd.DataFrame()

      # Create signals and simulate trades
      for signalmultiplier in [1, -1]:
        for var in signalcolumns:
          #print(var)
          action_col = f'action_{var}'
          # Create action signal
          mldf[action_col] = create_signal(signalmultiplier * mldf[var])
          # Simulate trades for each date
          results = []
          for date_val, group in mldf.groupby('date'):
            group = group.sort_values('t')
            pnl, position = simulate_trades_on_day(group['vwap'].tolist(), group[action_col].tolist())
            results.append({'date': date_val, 'pnl': pnl, 'position': position}) 
          # Create a DataFrame with results
          results_df = pd.DataFrame(results)
          # Merge results with mldf
          for idx, row in results_df.iterrows():
            mldf.loc[mldf['date'] == row['date'], f'pnl_{var}'] = row['pnl']
            mldf.loc[mldf['date'] == row['date'], f'position_{var}'] = row['position']
          # Create append frame
          appendframe = mldf[['date', f'pnl_{var}', f'position_{var}']].drop_duplicates(subset=['date']).copy()
          appendframe['var'] = var
          appendframe['signalmultiplier'] = signalmultiplier
          appendframe.rename(columns={f'pnl_{var}': 'pnl', f'position_{var}': 'position'}, inplace=True)
          # Append to pnlframe
          pnlframe = pd.concat([pnlframe, appendframe])

      # Generate optimized signals using MCMC/Simulated Annealing with GPU parallelism
      if GENOPTSIG:
        print(f"Generating optimized signals for {SYM}...")
        try:
            optimized_signals, enhanced_pnlframe = generate_optimized_signals_for_dataframe(
                mldf, signalcolumns, 
                method=OPTIMIZATION_METHOD,
                use_parallel=USE_PARALLEL_OPTIMIZATION,
                use_gpu=USE_GPU_ACCELERATION,
                max_workers=SIGNAL_OPTIMIZATION_WORKERS
            )
            
            # Add optimized signals to pnlframe
            if not enhanced_pnlframe.empty:
              enhanced_pnlframe['var'] = 'opt_' + enhanced_pnlframe['var'].astype(str)
              pnlframe = pd.concat([pnlframe, enhanced_pnlframe])
              print(f"Added {len(enhanced_pnlframe)} optimized signal entries to pnlframe")
            
            # Save optimized parameters for reference
            if optimized_signals:
                import json
                opt_params_file = os.path.join(basepath+'/traindata/', f"optimized_params_{SYM}.json")
                with open(opt_params_file, 'w') as f:
                    # Convert numpy types to native Python types for JSON serialization
                    serializable_signals = {}
                    for key, params in optimized_signals.items():
                        serializable_signals[key] = {
                            k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                            for k, v in params.items()
                        }
                    json.dump(serializable_signals, f, indent=2)
                print(f"Saved optimized parameters to {opt_params_file}")
                
        except Exception as e:
            print(f"Error generating optimized signals for {SYM}: {e}")
            print("Continuing with standard signals only...")

      # Analyze performance of signals (now includes both original and optimized)
      signal_performance = pnlframe.groupby(['var', 'signalmultiplier'])['pnl'].mean().reset_index()
      signal_performance = signal_performance.sort_values('pnl', ascending=False)

      # Select good signals
      goodsignals = signal_performance.copy()
      #goodsignals = goodsignals.iloc[:int(goodsignals.shape[0]/2)] #Random cut down 
      goodsignals['var'] = [x + '_mult_' + str(y) if 'opt_' in x else x for x,y in zip(goodsignals['var'],goodsignals['signalmultiplier'])]
      goodsignals = goodsignals.drop_duplicates(['var'])
      
      mldf.columns = [re.sub("opt_action_","opt_",x) for x in mldf.columns]
      # CORRELATION FILTERING COMMENTED OUT - POTENTIALLY CAUSING PROBLEMS
      # numeric_df = mldf[goodsignals['var'].tolist()]
      # ts = time.time()
      # cormat = numeric_df.corr()
      # te = time.time()
      # print(te-ts)
      # 
      # #SID: Implement collider filtering here 
      # 
      # # Select final signals based on correlation
      # CORMATTHRESHOLD1 = np.nanquantile(cormat,0.8) #This value needs to be tuned - also this may not be a good way to triage signals - individual performance may not mean much 
      # CORMATTHRESHOLD2 = np.nanquantile(cormat,0.95) #This value needs to be tuned - also this may not be a good way to triage signals - individual performance may not mean much 
      # cormat = cormat.fillna(0.99)
      #
      # finalsignals = [goodsignals['var'].iloc[0]]
      # for ss in goodsignals['var'].tolist():
      #   if ss in finalsignals:
      #       continue
      #   # Check if correlation with existing signals is too high
      #   max_corr = max([abs(cormat.loc[ss, fs]) if ss in cormat.index and fs in cormat.columns else 0.99
      #                    for fs in finalsignals])
      #   if max_corr >= CORMATTHRESHOLD1:
      #       continue
      #   finalsignals.append(ss)
      #
      # finalsignals2 = [goodsignals['var'].iloc[0]]
      # for ss in goodsignals['var'].tolist():
      #   if ss in finalsignals2:
      #       continue
      #   # Check if correlation with existing signals is too high
      #   max_corr = max([abs(cormat.loc[ss, fs]) if ss in cormat.index and fs in cormat.columns else 0.99
      #                    for fs in finalsignals2])
      #   if max_corr >= CORMATTHRESHOLD2:
      #       continue
      #   finalsignals2.append(ss)
      #
      # print(len(finalsignals))
      # print(len(finalsignals2))
      # print(len(signalcolumns))
      # 
      # print(finalsignals)
      # print(finalsignals2)
      # print(signalcolumns)
      
      # USE ALL SIGNALS WITHOUT CORRELATION FILTERING
      finalsignals = goodsignals['var'].tolist()
      finalsignals2 = goodsignals['var'].tolist()
      
      print(f"Using all {len(finalsignals)} signals without correlation filtering")
      print(f"Signals: {len(signalcolumns)} signal columns available")

      finalmldf = mldf[np.unique(['currentt', 'currento', 't', 'vwap2'] + finalsignals).tolist()].copy() #Final signals 
      finalmldf2 = mldf[np.unique(['currentt', 'currento', 't', 'vwap2'] + finalsignals2).tolist()].copy() #Final signals 
      mldf = mldf[np.unique(['currentt', 'currento', 't', 'vwap2'] + signalcolumns).tolist()].copy() #Original signals 
            
      # Save results
      finalmldf.to_csv(os.path.join(basepath+'/traindata/', f"finalmldf{SYM}.csv"), index=False)
      finalmldf2.to_csv(os.path.join(basepath+'/traindata/', f"finalmldf2{SYM}.csv"), index=False)
      mldf.to_csv(os.path.join(basepath+'/traindata/', f"mldf{SYM}.csv"), index=False)
      
      print(f"Processed {SYM} successfully")
  print("Preprocessing complete!")
  return nse,nfo


def registersignals(SYM,symsignals):
  global symbolsignals
  symbolsignals[SYM] = symsignals  

def forwardreturn(col):
  #Assumes an ascending ordered column 
  return (col-col.shift(1))/col

def computedelta(i,allreturns):
  ret = allreturns[i]
  benchmark = [allreturns[j] for j in range(len(allreturns)) if j!=i]
  benchmark = pd.Series([item for sublist in benchmark for item in sublist])
  ret = ret[~np.isnan(ret)]
  benchmark = benchmark[~np.isnan(benchmark)]
  return float(wasserstein_distance(ret,benchmark))

def listproduct(*lists):
    return list(itertools.product(*lists))


class CustomTensorboardCallback(BaseCallback):
    """
    Custom callback for TensorBoard logging with additional financial metrics
    This implementation uses stable-baselines3's logger which supports TensorBoard
    """
    def __init__(self, verbose=VERBOSITY, log_freq=LOGFREQ, tensorboard_log=None):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.tensorboard_log_dir = tensorboard_log or basepath+"/tmp/tb_logs/"
        # Ensure the log directory exists
        os.makedirs(self.tensorboard_log_dir, exist_ok=True)
        
        # Initialize financial metrics
        self.returns = []
        self.sharpe_ratio = 0
        self.max_drawdown = 0
        
    def _on_step(self):
        # Log information at specified frequency
        if self.log_freq > 0 and self.num_timesteps % self.log_freq == 0:
            # Get metrics from logger
            log_dict = self.logger.name_to_value
            
            # Note: PPO training metrics (value_loss, explained_variance, policy_loss, etc.)
            # are logged directly by BoundedEntropyPPO.train() method under the train/ namespace.
            # We don't duplicate them here to avoid confusion with zero values.
            
            # Extract rewards from environment if possible
            try:
                if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                    # Get episode rewards from the model's episode info buffer
                    recent_rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
                    if recent_rewards:
                        mean_reward = np.mean(recent_rewards)
                        self.logger.record("custom/mean_reward", mean_reward)
                        
                        # Calculate Sharpe ratio if enough data
                        if len(recent_rewards) > 1:
                            std_reward = np.std(recent_rewards) + 1e-8  # Avoid division by zero
                            self.sharpe_ratio = mean_reward / std_reward
                            self.logger.record("custom/sharpe_ratio", self.sharpe_ratio)
                
                # Try to extract rewards from VecEnv
                elif hasattr(self.model, 'env'):
                    # This approach works with DummyVecEnv
                    try:
                        if hasattr(self.model.env, 'buf_rews'):
                            recent_rewards = self.model.env.buf_rews
                            if len(recent_rewards) > 0:
                                mean_reward = np.mean([r for r in recent_rewards if r is not None])
                                self.logger.record("custom/mean_reward", mean_reward)
                    except (AttributeError, IndexError):
                        pass
            except Exception as e:
                if self.verbose > 0:
                    print(f"Error calculating reward metrics: {e}")
            
            # Log to console if verbose
            if self.verbose > 0:
                print(f"Step {self.num_timesteps}:")
                print(f"  Value Loss: {value_loss}")
                print(f"  Explained Variance: {explained_variance}")
                if hasattr(self, 'sharpe_ratio'):
                    print(f"  Sharpe Ratio: {self.sharpe_ratio}")
            
            # Ensure metrics are written to TensorBoard
            self.logger.dump(self.num_timesteps)
                
        return True

class CustomLoggerCallback(BaseCallback):
    def __init__(self, verbose=VERBOSITY, log_freq=1000):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.tmp_path = basepath+"/tmp/sb3_log/"
        # Ensure the log directory exists
        os.makedirs(self.tmp_path, exist_ok=True)
        # Clear previous log file
        with open(os.path.join(self.tmp_path, METRICS_FILE), "w") as f:
            f.write("timesteps,mean_reward\n")
        
        # Store last observed mean reward
        self.last_mean_reward = 0.0
    
    def _on_step(self):
        # Log information at specified frequency
        if self.log_freq > 0 and self.num_timesteps % self.log_freq == 0:
            # Note: PPO training metrics (value_loss, explained_variance, etc.) are logged
            # directly by BoundedEntropyPPO.train() method under the train/ namespace.
            
            # Enhanced reward extraction with fallback to cumulative rewards
            mean_reward = None
            
            # Get metrics from logger if available
            log_dict = self.logger.name_to_value if hasattr(self.logger, 'name_to_value') else {}
            
            # Primary source: check standard rollout metrics first
            if 'rollout/ep_rew_mean' in log_dict:
                mean_reward = log_dict['rollout/ep_rew_mean']
            
            # Secondary: episode info buffer (most reliable for completed episodes)
            elif hasattr(self.model, 'ep_info_buffer') and self.model.ep_info_buffer and len(self.model.ep_info_buffer) > 0:
                try:
                    recent_rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer if "r" in ep_info]
                    if recent_rewards:
                        mean_reward = np.mean(recent_rewards)
                except (KeyError, TypeError):
                    pass
            
            # Fallback 1: Check VecNormalize for cumulative returns
            elif hasattr(self.model, 'env') and hasattr(self.model.env, 'ret'):
                try:
                    if hasattr(self.model.env, 'ret') and self.model.env.ret is not None:
                        ret_array = np.array(self.model.env.ret)
                        if ret_array.size > 0:
                            mean_reward = np.mean(ret_array[ret_array != 0])  # Exclude zero returns
                except:
                    pass
            
            # Fallback 2: Use rollout buffer rewards if available
            elif hasattr(self.model, 'rollout_buffer') and hasattr(self.model.rollout_buffer, 'rewards'):
                try:
                    recent_rewards = self.model.rollout_buffer.rewards.flatten()
                    if len(recent_rewards) > 0:
                        mean_reward = np.mean(recent_rewards)
                except:
                    pass
            
            # Use previous value if we can't find anything
            if mean_reward is None:
                mean_reward = self.last_mean_reward
            
            # Update the last mean reward for future use
            if mean_reward is not None and mean_reward != 0.0:
                self.last_mean_reward = mean_reward
            
            # Format mean reward for logging
            if isinstance(mean_reward, (int, float)):
                mean_reward_str = f"{mean_reward:.6f}"
            else:
                mean_reward_str = str(mean_reward)
            
            # Debug: Print available log_dict keys to understand what's available
            if self.verbose > 1 and self.num_timesteps % (self.log_freq * 5) == 0:  # Every 5th log cycle
                available_keys = list(log_dict.keys())
                print(f"Debug - Available log_dict keys: {[k for k in available_keys if 'reward' in k.lower() or 'ep_' in k.lower()]}")
            
            # Log to console (only log mean reward since other metrics are logged by PPO)
            self.logger.info(f"Step {self.num_timesteps}: Mean Reward: {mean_reward_str}")
            
            # Debug: Check if we're getting actual episode rewards
            if self.verbose > 1:
                if hasattr(self.model, 'ep_info_buffer'):
                    ep_buffer_len = len(self.model.ep_info_buffer) if self.model.ep_info_buffer else 0
                    print(f"Debug - Episode buffer length: {ep_buffer_len}")
                    if ep_buffer_len > 0:
                        recent_ep_info = self.model.ep_info_buffer[-min(3, ep_buffer_len):]  # Last 3 episodes
                        print(f"Debug - Recent episode info: {recent_ep_info}")
                else:
                    print("Debug - No ep_info_buffer attribute found")
                
                # Check environment info
                if hasattr(self.model, 'env'):
                    print(f"Debug - Environment type: {type(self.model.env)}")
                    if hasattr(self.model.env, 'envs'):
                        print(f"Debug - Number of environments: {len(self.model.env.envs)}")
                        if len(self.model.env.envs) > 0:
                            env = self.model.env.envs[0]
                            if hasattr(env, 'current_step'):
                                print(f"Debug - Environment current step: {env.current_step}")
                            if hasattr(env, 'MAX_STEPS'):
                                print(f"Debug - Environment MAX_STEPS: {env.MAX_STEPS}")
                            if hasattr(env, 'df'):
                                print(f"Debug - Environment data length: {len(env.df)}")
                    
                    # Check if VecNormalize has episode returns
                    if hasattr(self.model.env, 'ret'):
                        print(f"Debug - VecNormalize returns shape: {self.model.env.ret.shape if hasattr(self.model.env.ret, 'shape') else 'no shape'}")
                        print(f"Debug - VecNormalize returns: {self.model.env.ret}")
                    
                    # Check rollout buffer for rewards
                    if hasattr(self.model, 'rollout_buffer') and hasattr(self.model.rollout_buffer, 'rewards'):
                        rewards_shape = self.model.rollout_buffer.rewards.shape
                        recent_rewards = self.model.rollout_buffer.rewards.flatten()
                        print(f"Debug - Rollout buffer rewards shape: {rewards_shape}")
                        print(f"Debug - Recent rollout rewards (last 5): {recent_rewards[-5:] if len(recent_rewards) >= 5 else recent_rewards}")
                        print(f"Debug - Mean rollout reward: {np.mean(recent_rewards) if len(recent_rewards) > 0 else 'N/A'}")
                
                print(f"Debug - Current timestep: {self.num_timesteps}")
            
            # Log to TensorBoard via the logger
            if hasattr(self.logger, 'record'):
                # Only log the mean reward which we calculate ourselves
                self.logger.record("custom/mean_reward", mean_reward)
                # Note: PPO training metrics are logged directly by BoundedEntropyPPO.train()
                self.logger.dump(step=self.num_timesteps)
            
            # Write to file (only log mean reward since other metrics come from PPO)
            with open(os.path.join(self.tmp_path, METRICS_FILE), "a") as f:
                f.write(f"{self.num_timesteps},{mean_reward_str}\n")
        
        return True

def modeltrain(rdflistp,NEWMODEL=NEWMODELFLAG,SYMLISTGIVEN=TESTSYMBOLS,DELETEMODELS=True,SAVE_BEST_MODEL=True,lol=None,progress_tracker=None):
  import shutil  # Import shutil at the beginning of the function
  # Ensure log directories exist
  if not os.path.exists(basepath+'/tmp'):
    os.makedirs(basepath+'/tmp')
  if not os.path.exists(basepath+'/tmp/sb3_log'):
    os.makedirs(basepath+'/tmp/sb3_log')
  if not os.path.exists(basepath+'/tmp/checkpoints'):
    os.makedirs(basepath+'/tmp/checkpoints')
  if not os.path.exists(basepath+'/tmp/tensorboard_logs'):
    os.makedirs(basepath+'/tmp/tensorboard_logs')
  if not os.path.exists(basepath+'/models'):
    os.makedirs(basepath+'/models')
      
  if DELETEMODELS:
    # Delete models but preserve any needed files like quantile transformers
    models_to_delete = glob.glob(basepath+'/models/*model*.zip') + glob.glob(basepath+'/models/*model*.onnx')
    for f in models_to_delete:
      os.remove(f)  
    files = glob.glob(basepath+'/tmp/checkpoints/*')
    for f in files:
      os.remove(f)  
    files = glob.glob(basepath+'/tmp/sb3_log/*')
    for f in files:
      os.remove(f)  
    files = glob.glob(basepath+'/tmp/tensorboard_logs/*')
    for f in files:
      shutil.rmtree(f)
    
  for SYM in SYMLISTGIVEN: 
    # Start tracking progress for this symbol
    if progress_tracker:
      progress_tracker.start_symbol(SYM)
    
    for prefix in ["final"]: 
      print('SYM:' + SYM + ' prefix:' + prefix)
      df = rdflistp[SYM+prefix]
      print(df)
      df['currentt'] = pd.to_datetime(df['currentt'])
      df['currentdate'] = df['currentt'].dt.date
      alldates = df['currentdate'].unique()
      traindates = alldates[:int(np.ceil(len(alldates)*TRAIN_MAX))]
      TRAINROWS = df[df.currentdate.isin(traindates)].shape[0]
      
      if prefix=="final":
        finalsignalsp = lol[SYM] #df.columns[~df.columns.isin(['currentt','currento','currentdate','vwap2'])].tolist()
      else:
        finalsignalsp = globalsignals
      
      if(TOPN>0):
        dropcols = finalsignalsp[TOPN:]
        df = df.drop(dropcols,axis=1)
        finalsignalsp = finalsignalsp[:TOPN]
      
      df_train = df.iloc[:TRAINROWS].reset_index(drop=True)
      df_test = df.iloc[(TRAINROWS+1):].reset_index(drop=True)
      
      # Enhanced quantiling strategy for market regime detection
      # Separate quantile transformers for different signal types (bear, bull, and pivot)
      regime_signals = ['bear_signal', 'oversold_extreme', 'bb_squeeze', 'support_break', 
                       'hammer_pattern', 'volume_divergence', 'trend_alignment_bear', 'market_fear',
                       'bull_signal', 'overbought_extreme', 'bb_expansion', 'resistance_break',
                       'morning_star', 'volume_confirmation', 'trend_alignment_bull', 'market_greed',
                       'pivot_high_3', 'pivot_low_3', 'pivot_high_5', 'pivot_low_5', 'pivot_high_7', 'pivot_low_7',
                       'local_max', 'local_min', 'swing_high', 'swing_low']
      regular_signals = [sig for sig in finalsignalsp if sig not in regime_signals]
      
      # Use different quantile strategies
      qt_regular = QuantileTransformer(n_quantiles=min(NQUANTILES, len(df_train)//10), 
                                      output_distribution='uniform', random_state=0)
      qt_regime = QuantileTransformer(n_quantiles=min(3, len(df_train)//20),  # Fewer quantiles for binary signals
                                     output_distribution='uniform', random_state=0)
      
      df_train_transformed = df_train.copy()
      df_test_transformed = df_test.copy()
      
      # Transform regular signals with parallel processing
      if regular_signals:
          print(f"Transforming {len(regular_signals)} regular signals using {N_CORES} cores")
          qt_regular.fit(df_train[regular_signals].fillna(0))
          
          # Use parallel processing for transformation if available
          try:
              from joblib import Parallel, delayed
              
              def transform_chunk(data_chunk):
                  return qt_regular.transform(data_chunk.fillna(0))
              
              # Parallel transform for train data
              train_chunks = np.array_split(df_train_transformed[regular_signals], N_CORES)
              train_results = Parallel(n_jobs=N_CORES)(
                  delayed(transform_chunk)(chunk) for chunk in train_chunks
              )
              df_train_transformed[regular_signals] = pd.DataFrame(
                  np.vstack(train_results), columns=regular_signals, index=df_train_transformed.index)
              
              # Parallel transform for test data  
              test_chunks = np.array_split(df_test_transformed[regular_signals], N_CORES)
              test_results = Parallel(n_jobs=N_CORES)(
                  delayed(transform_chunk)(chunk) for chunk in test_chunks
              )
              df_test_transformed[regular_signals] = pd.DataFrame(
                  np.vstack(test_results), columns=regular_signals, index=df_test_transformed.index)
                  
          except Exception as e:
              print(f"Parallel processing failed, using serial: {e}")
              # Fallback to serial processing
              df_train_transformed[regular_signals] = pd.DataFrame(
                  qt_regular.transform(df_train_transformed[regular_signals].fillna(0)),
                  columns=regular_signals, index=df_train_transformed.index)
              df_test_transformed[regular_signals] = pd.DataFrame(
                  qt_regular.transform(df_test_transformed[regular_signals].fillna(0)),
                  columns=regular_signals, index=df_test_transformed.index)
          
          joblib.dump(qt_regular, basepath+'/models/'+SYM+'qt_regular.joblib')
      
      # Transform market regime signals with special handling
      available_regime_signals = [sig for sig in regime_signals if sig in df_train.columns]
      if available_regime_signals:
          qt_regime.fit(df_train[available_regime_signals].fillna(0))
          df_train_transformed[available_regime_signals] = pd.DataFrame(
              qt_regime.transform(df_train_transformed[available_regime_signals].fillna(0)),
              columns=available_regime_signals, index=df_train_transformed.index)
          df_test_transformed[available_regime_signals] = pd.DataFrame(
              qt_regime.transform(df_test_transformed[available_regime_signals].fillna(0)),
              columns=available_regime_signals, index=df_test_transformed.index)
          joblib.dump(qt_regime, basepath+'/models/'+SYM+'qt_regime.joblib')
          
          print(f"Enhanced quantiling: {len(regular_signals)} regular signals, {len(available_regime_signals)} regime signals")
      
      # Fallback for backward compatibility
      qt = qt_regular  # Use regular transformer as default
      joblib.dump(qt, basepath+'/models/'+SYM+'qt.joblib')
      NUMVARS = len(finalsignalsp)
      
      try:
        del sys.modules['env.StockTradingEnvOptimized']
      except: 
        print("No such environment")
      
      from StockTradingEnvOptimized import StockTradingEnvOptimized
      USE_OPTIMIZED_ENV = True
      
      # Calculate overall training weight based on dataset characteristics
      df_train_transformed_list = [d.reset_index(drop=True) for _, d in df_train_transformed.groupby('currentdate')] 
      
      allreturns = [forwardreturn(x.vwap2) for x in df_train_transformed_list]
      meanallreturns = [float(np.mean(x)) for x in allreturns]
      
      # Enhanced weighting strategy to emphasize down days and bear market conditions
      dfweightdistribution = [computedelta(i,allreturns) for i in range(len(allreturns))] #This is just how different it is 
      dfweightrecency = [math.pow((i+1),1/3) for i in range(len(allreturns))]  #recency matters
      
      # Additional weighting factors for market conditions - BALANCED APPROACH
      market_condition_weights = []
      volatility_weights = []
      return_day_weights = []
      
      for i, dfs in enumerate(df_train_transformed_list):
          # Balanced market condition weight (reduced bear bias)
          bear_weight = 1.0
          bull_weight = 1.0
          
          if 'bear_signal' in dfs.columns:
              bear_weight = 1.0 + (dfs['bear_signal'].mean() * 0.25)  # Reduced from 0.5 to 0.25
          if 'bull_signal' in dfs.columns:
              bull_weight = 1.0 + (dfs['bull_signal'].mean() * 0.25)  # Equal bull signal weight
              
          # Use geometric mean to balance bear/bull influences
          market_weight = np.sqrt(bear_weight * bull_weight)
          market_condition_weights.append(market_weight)
          
          # Volatility weight (higher weight for high volatility days)
          if 'volatility' in dfs.columns:
              vol_weight = 1.0 + min(dfs['volatility'].mean() * 2, 0.4)  # Up to 40% more weight
          else:
              vol_weight = 1.0
          volatility_weights.append(vol_weight)
          
          # Balanced return day weight (reduced down day bias)
          daily_return = meanallreturns[i]
          if daily_return < 0:
              return_weight = 1.3 + abs(daily_return) * 2  # Reduced from 1.5 + 3x to 1.3 + 2x
          elif daily_return > 0:
              return_weight = 1.2 + daily_return * 1.5  # NEW: Add weight for up days too
          else:
              return_weight = 1.0
          return_day_weights.append(return_weight)
      
      # Combine all weight factors
      dfweights = [a*b*c*d*e for a,b,c,d,e in zip(dfweightdistribution, dfweightrecency, 
                                                   market_condition_weights, volatility_weights, return_day_weights)]
      
      maxdfweights = max(dfweights)
      mindfweights = min(dfweights)
      dfweights = [float(np.exp((x-mindfweights)/(maxdfweights-mindfweights))) for x in dfweights]
      
      print(f"Weight distribution - Min: {min(dfweights):.3f}, Max: {max(dfweights):.3f}, Mean: {np.mean(dfweights):.3f}")
      print(f"Down days emphasis: {sum(1 for r in meanallreturns if r < 0)} out of {len(meanallreturns)} days")
      print(f"Training on entire dataset with {N_ENVS} parallel environments")

      # Train on the entire dataset at once instead of day by day
      dfs = df_train_transformed.reset_index(drop=True)
      print(f"Training on entire dataset with {len(dfs)} rows across {len(df_train_transformed['currentdate'].unique())} days")
      
      # Multi-core environment setup using SubprocVecEnv
      def make_env(dfs_copy, rank=0):
        """Create environment factory function for multiprocessing"""
        def _init():
            # Import here to avoid circular imports in subprocess
            import sys
            import os
            
            # Ensure path is available
            env_path = os.path.join(os.path.dirname(os.path.realpath('__file__')), 'RL-1')
            if env_path not in sys.path:
                sys.path.append(env_path)
            
            # Import parameters locally to ensure they're available in subprocess
            from parameters import INITIAL_ACCOUNT_BALANCE as INIT_BALANCE
            from StockTradingEnvOptimized import StockTradingEnvOptimized
            env_class = StockTradingEnvOptimized
            
            # Use local copies of parameters to avoid pickling issues
            nlags = NLAGS
            numvars = NUMVARS  
            max_short = MAXIMUM_SHORT_VALUE
            init_balance = INIT_BALANCE
            max_steps = MAX_STEPS
            signals = finalsignalsp.copy()  # Copy to avoid reference issues
            
            return env_class(dfs_copy, nlags, numvars, max_short, 
                                  init_balance, max_steps, signals, 
                                  init_balance, INITIAL_SHARES_HELD=0)
        return _init
        
      # Create multiple environments for parallel training
      from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
      
      if N_ENVS > 1:
            
          # Set environment variables for multiprocessing optimization
          os.environ['OMP_NUM_THREADS'] = str(OMP_NUM_THREADS)
          os.environ['MKL_NUM_THREADS'] = str(OMP_NUM_THREADS)
          
          # Create SubprocVecEnv for true parallel processing
          env = SubprocVecEnv([make_env(dfs.copy(), i) for i in range(N_ENVS)], 
                             start_method='spawn')  # Use spawn for multiprocessing compatibility
          print(f"Created {N_ENVS} parallel environments using SubprocVecEnv")
      else:
          # Fallback to DummyVecEnv for single environment
          # Determine which environment class to use
          env_class = StockTradingEnvOptimized
          env = DummyVecEnv([make_env(dfs, 0)])
          print(f"Using single optimized environment with DummyVecEnv")
        
      # Add reward normalization to reduce value loss variance (less aggressive clipping)
      env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=1.0, gamma=GAMMA)
      print("Added VecNormalize wrapper (rewards scaled in environment)")
      #new_logger = configure(tmp_path, ["stdout", "csv"])
      
      # Create checkpoint callback for saving model at regular intervals
      checkpoint_callback = CheckpointCallback(
        save_freq=LOGFREQ,
        save_path=check_path,
        name_prefix="ppo_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
      )

      # Create custom logger callback for metrics recording
      custom_logger = CustomLoggerCallback(verbose=VERBOSITY, log_freq=LOGFREQ)
      
      # Create TensorBoard callback for visualizing training progress
      # Ensure the tensorboard log directory exists
      os.makedirs(tensorboard_log_path, exist_ok=True)
      # Create a unique run name for this training session
      tb_run_name = f"{SYM}_{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
      # Create directory for this specific run
      run_dir = os.path.join(tensorboard_log_path, tb_run_name)
      os.makedirs(run_dir, exist_ok=True)
      tensorboard_callback = CustomTensorboardCallback(
          verbose=VERBOSITY, 
          log_freq=LOGFREQ,  # More frequent logging for TensorBoard
          tensorboard_log=run_dir
      )
      
      # Combine all callbacks (removed EvalCallback since we use find_best_checkpoint instead)
      # Add progress tracking callback
      from progress_tracker import create_progress_callback
      adaptive_timesteps = max(BASEMODELITERATIONS, TOTAL_TIMESTEPS * 5)
      progress_callback = create_progress_callback(adaptive_timesteps, SYM)
      
      callbacks = CallbackList([checkpoint_callback, custom_logger, tensorboard_callback, progress_callback])
      
      if(prefix=="final"):
        modelfilename = SYM + 'localmodel'
        try:
          if NEWMODEL:
            # Prepare policy kwargs with optimized architecture
            policy_kwargs = POLICY_KWARGS.copy()
            policy_kwargs['activation_fn'] = torch.nn.ReLU  # Convert string to actual function
            
            # Remove device from policy_kwargs (handled by SB3 directly)
            if 'device' in policy_kwargs:
                del policy_kwargs['device']
            
            # Base PPO parameters optimized for multi-core training
            ppo_params = {
                'learning_rate': GLOBALLEARNINGRATE,
                'n_steps': N_STEPS,
                'batch_size': BATCH_SIZE,
                'n_epochs': N_EPOCHS,
                'gamma': GAMMA,
                'gae_lambda': GAE_LAMBDA,
                'clip_range': CLIP_RANGE,
                'clip_range_vf': CLIP_RANGE_VF,
                'normalize_advantage': NORAD,
                'ent_coef': ENT_COEF,
                'vf_coef': VF_COEF,
                'max_grad_norm': MAX_GRAD_NORM,
                'target_kl': TARGET_KL,
                'tensorboard_log': run_dir,
                'stats_window_size': STATS_WINDOW_SIZE,
                'policy_kwargs': policy_kwargs,
                'device': DEVICE,  # Use CUDA or CPU
                'seed': 42  # For reproducibility across cores
            }
            
            # Add SDE parameters only if enabled
            if USE_SDE:
                ppo_params['use_sde'] = True
                ppo_params['sde_sample_freq'] = SDE_SAMPLE_FREQ
            
            print(f"Creating PPO model with device: {DEVICE}, network size: {policy_kwargs['net_arch']}")
            # Add entropy bound parameter
            ppo_params['entropy_bound'] = ENTROPY_BOUND
            # Add value loss bound parameter
            ppo_params['value_loss_bound'] = VALUE_LOSS_BOUND
            model = BoundedEntropyPPO("MlpPolicy", env, verbose=VERBOSITY, **ppo_params)
            print("Training local PPO model from scratch")
          else:
            modelpath = basepath + "/models/" + modelfilename
            # Base load parameters
            load_params = {
                'learning_rate': GLOBALLEARNINGRATE,
                'n_steps': N_STEPS,
                'batch_size': BATCH_SIZE,
                'n_epochs': N_EPOCHS,
                'gamma': GAMMA,
                'gae_lambda': GAE_LAMBDA,
                'clip_range': CLIP_RANGE,
                'clip_range_vf': CLIP_RANGE_VF,
                'normalize_advantage': NORAD,
                'ent_coef': ENT_COEF,
                'vf_coef': VF_COEF,
                'max_grad_norm': MAX_GRAD_NORM,
                'target_kl': TARGET_KL,
                'stats_window_size': STATS_WINDOW_SIZE
            }
            
            # Add SDE parameters only if enabled
            if USE_SDE:
                load_params['use_sde'] = True
                load_params['sde_sample_freq'] = SDE_SAMPLE_FREQ
            
            model = PPO.load(modelpath, env, verbose=VERBOSITY, **load_params)
            print("Loaded local PPO model")
        except Exception as e:
          print(f"Error loading model: {e}")
          print("Training local model from scratch")
          # Prepare policy kwargs with optimized architecture
          policy_kwargs = POLICY_KWARGS.copy()
          policy_kwargs['activation_fn'] = torch.nn.ReLU  # Convert string to actual function
          
          # Remove device from policy_kwargs (handled by SB3 directly)
          if 'device' in policy_kwargs:
              del policy_kwargs['device']
          
          # Base PPO parameters
          ppo_params = {
              'learning_rate': GLOBALLEARNINGRATE,
              'n_steps': N_STEPS,
              'batch_size': BATCH_SIZE,
              'n_epochs': N_EPOCHS,
              'gamma': GAMMA,
              'gae_lambda': GAE_LAMBDA,
              'clip_range': CLIP_RANGE,
              'clip_range_vf': CLIP_RANGE_VF,
              'normalize_advantage': NORAD,
              'ent_coef': ENT_COEF,
              'vf_coef': VF_COEF,
              'max_grad_norm': MAX_GRAD_NORM,
              'target_kl': TARGET_KL,
              'tensorboard_log': run_dir,
              'stats_window_size': STATS_WINDOW_SIZE,
              'policy_kwargs': policy_kwargs,
              'device': DEVICE
          }
          
          # Add SDE parameters only if enabled
          if USE_SDE:
              ppo_params['use_sde'] = True
              ppo_params['sde_sample_freq'] = SDE_SAMPLE_FREQ
          
          # Add entropy bound parameter
          ppo_params['entropy_bound'] = ENTROPY_BOUND
          # Add value loss bound parameter
          ppo_params['value_loss_bound'] = VALUE_LOSS_BOUND
          model = BoundedEntropyPPO("MlpPolicy", env, verbose=VERBOSITY, **ppo_params)
          print("Created new PPO model")
            
        # First training run with callbacks
        files = glob.glob(basepath+'/tmp/sb3_log/*')
        for f in files:
          os.remove(f)
          
        # Enhanced training with adaptive timesteps optimized for multi-core
        base_timesteps = BASEMODELITERATIONS
        min_timesteps = TOTAL_TIMESTEPS * 5  # Use multi-core optimized total timesteps
        adaptive_timesteps = max(base_timesteps, min_timesteps)
        
        print(f"Multi-core training: {adaptive_timesteps} total timesteps ({adaptive_timesteps//N_ENVS} per env)")
        print(f"Environments: {N_ENVS}, Device: {DEVICE}")
          
        # Add learning rate scheduling
        try:
            # MPS memory optimization - clear cache before training
            if DEVICE == "mps" and torch.backends.mps.is_available():
                torch.mps.empty_cache()
                torch.mps.synchronize()
            
            model.learn(
                total_timesteps=adaptive_timesteps, 
                callback=callbacks,
                reset_num_timesteps=True,
                tb_log_name=f"{SYM}_{prefix}_training"
            )
            
            # MPS memory optimization - clear cache after training
            if DEVICE == "mps" and torch.backends.mps.is_available():
                torch.mps.empty_cache()
                
        except Exception as e:
            print(f"Training interrupted: {e}")
            print("Continuing with current model state...")
            pass
        
        # After training, find and load the best model from checkpoints
        if SAVE_BEST_MODEL:
            best_model_path = find_best_checkpoint(SYM, prefix, METRICS_FILE=METRICS_FILE)
            if best_model_path:
                print(f"Loading best model checkpoint so far for {SYM}{prefix}")
                # Base load parameters for best checkpoint
                load_params = {
                    'learning_rate': GLOBALLEARNINGRATE,
                    'n_steps': N_STEPS,
                    'batch_size': BATCH_SIZE,
                    'n_epochs': N_EPOCHS,
                    'gamma': GAMMA,
                    'gae_lambda': GAE_LAMBDA,
                    'clip_range': CLIP_RANGE,
                    'clip_range_vf': CLIP_RANGE_VF,
                    'normalize_advantage': NORAD,
                    'ent_coef': ENT_COEF,
                    'vf_coef': VF_COEF,
                    'max_grad_norm': MAX_GRAD_NORM,
                    'target_kl': TARGET_KL,
                    'stats_window_size': STATS_WINDOW_SIZE
                }
                
                # Add SDE parameters only if enabled
                if USE_SDE:
                    load_params['use_sde'] = True
                    load_params['sde_sample_freq'] = SDE_SAMPLE_FREQ
                
                model = PPO.load(best_model_path, env=env, **load_params)
                print(f"Loaded best PPO model from checkpoint")
            
        # Get best mean reward from metrics file
        MEANREWARD = 0.0
        try:
          if os.path.exists(METRICS_FILE_PATH):
            with open(METRICS_FILE_PATH, 'r') as f:
              lines = f.readlines()
              if len(lines) > 1:  # Skip header
                # Find the highest mean reward in any line
                best_explained_variance = 0.0
                for line in lines[1:]:  # Skip header
                  if ',' in line:
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                      try:
                        explained_variance = float(parts[3])
                        best_explained_variance = max(best_explained_variance, explained_variance)
                      except (ValueError, IndexError):
                        continue
                
                MEANREWARD = best_explained_variance
                print(f"best mean reward so far: {MEANREWARD}")
        except Exception as e:
          print(f"Error reading metrics file: {e}")
        
        # Additional training if needed
        ITERATIONS  = 0 #SID: Not training again
        while ((MEANREWARD < MEANREWARDTHRESHOLD) & RETRAIN):
          print(f"Training again since MEANREWARD ({MEANREWARD}) is < MEANREWARDTHRESHOLD ({MEANREWARDTHRESHOLD})")
          
          files = glob.glob(basepath+'/tmp/sb3_log/*')
          for f in files:
            os.remove(f)
          
          # Continue training with callbacks, starting from best model
          # Enhanced continued training
          continue_timesteps = max(BASEMODELITERATIONS, N_STEPS * 5)  # Minimum for continued training
          print(f"Continuing training with {continue_timesteps} timesteps")
          
          try:
              # MPS memory optimization - clear cache before continued training
              if DEVICE == "mps" and torch.backends.mps.is_available():
                  torch.mps.empty_cache()
                  torch.mps.synchronize()
              
              model.learn(
                  total_timesteps=continue_timesteps, 
                  callback=callbacks,
                  reset_num_timesteps=False,  # Don't reset for continued training
                  tb_log_name=f"{SYM}_{prefix}_continued"
              )
              
              # MPS memory optimization - clear cache after continued training
              if DEVICE == "mps" and torch.backends.mps.is_available():
                  torch.mps.empty_cache()
          except Exception as e:
              print(f"Continued training interrupted: {e}")
              print("Proceeding with current model...")
              pass
          
          # After each iteration, find and load the best model
          if SAVE_BEST_MODEL:
            best_model_path = find_best_checkpoint(SYM, prefix, METRICS_FILE=METRICS_FILE)
            if best_model_path:
                print(f"Loading best model checkpoint after iteration {ITERATIONS+1} for {SYM}{prefix}")
                # Base load parameters for retrain iteration
                load_params = {
                    'learning_rate': GLOBALLEARNINGRATE,
                    'n_steps': N_STEPS,
                    'batch_size': BATCH_SIZE,
                    'n_epochs': N_EPOCHS,
                    'gamma': GAMMA,
                    'gae_lambda': GAE_LAMBDA,
                    'clip_range': CLIP_RANGE,
                    'clip_range_vf': CLIP_RANGE_VF,
                    'normalize_advantage': NORAD,
                    'ent_coef': ENT_COEF,
                    'vf_coef': VF_COEF,
                    'max_grad_norm': MAX_GRAD_NORM,
                    'target_kl': TARGET_KL,
                    'stats_window_size': STATS_WINDOW_SIZE
                }
                
                # Add SDE parameters only if enabled
                if USE_SDE:
                    load_params['use_sde'] = True
                    load_params['sde_sample_freq'] = SDE_SAMPLE_FREQ
                
                model = PPO.load(best_model_path, env=env, **load_params)
            
          # Clean up GPU memory after each iteration
          cleanup_gpu_memory()
          
          # Get updated best mean reward
          MEANREWARD = 0.0
          try:
            if os.path.exists(METRICS_FILE_PATH):
              with open(METRICS_FILE_PATH, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:  # Skip header
                  # Find the highest mean reward in any line
                  best_explained_variance = 0.0
                  for line in lines[1:]:  # Skip header
                    if ',' in line:
                      parts = line.strip().split(',')
                      if len(parts) >= 3:
                        try:
                          explained_variance = float(parts[3])
                          best_explained_variance = max(best_explained_variance, explained_variance)
                        except (ValueError, IndexError):
                          continue
                  MEANREWARD = best_explained_variance
                  print(f"Best Mean Reward after iteration {ITERATIONS+1}: {MEANREWARD}")
          except Exception as e:
            print(f"Error reading metrics file after additional training: {e}")
          
          ITERATIONS += 1
          if ITERATIONS >= MAXITERREPEAT:
            print("Reached maximum training iterations")
            break
            
        # Enhanced model saving with performance validation
        if SAVE_BEST_MODEL:
          # Try to find and save the best model from checkpoints
          best_model_path = find_best_checkpoint(SYM, prefix, METRICS_FILE=METRICS_FILE)
          if best_model_path:
            print(f"Using best model from checkpoints for {SYM}{prefix}")
            
            # Validate the best model performance before final save
            try:
              # Load model with proper device specification
              test_model = PPO.load(best_model_path, device=DEVICE)
              
              # Safely reset environment (handle multi-env case)
              try:
                  obs = env.reset()
              except Exception as reset_error:
                  print(f"Environment reset failed, creating new env for validation: {reset_error}")
                  # Create a simple validation environment
                  from stable_baselines3.common.vec_env import DummyVecEnv
                  eval_dfs = dfs.iloc[-50:].reset_index(drop=True) if len(dfs) > 50 else dfs
                  def make_validation_env():
                      env_class = StockTradingEnvOptimized
                      return env_class(eval_dfs, NLAGS, NUMVARS, MAXIMUM_SHORT_VALUE,
                                            INITIAL_ACCOUNT_BALANCE, min(MAX_STEPS, 50),
                                            finalsignalsp, INITIAL_ACCOUNT_BALANCE, 0)
                  eval_env = DummyVecEnv([make_validation_env])
                  obs = eval_env.reset()
                  env = eval_env  # Use the new env for validation
              
              test_reward = 0.0  # Initialize as float
              validation_steps = 0
              
              for _ in range(min(100, len(dfs)//10)):  # Quick validation run
                action, _ = test_model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                
                # Handle reward conversion safely for vectorized environments
                if hasattr(reward, 'item') and hasattr(reward, 'size') and reward.size == 1:
                    # Single value tensor/array
                    reward_val = float(reward.item())
                elif isinstance(reward, (list, tuple, np.ndarray)):
                    # Handle vectorized environments (multiple envs)
                    if len(reward) > 0:
                        reward_val = float(reward[0])  # Take first environment's reward
                    else:
                        reward_val = 0.0
                else:
                    reward_val = float(reward)
                
                test_reward += reward_val
                validation_steps += 1
                
                # Handle done flag for vectorized environments
                if isinstance(done, (list, tuple, np.ndarray)):
                    done_val = done[0] if len(done) > 0 else False
                else:
                    done_val = bool(done)
                
                if done_val:
                  break
              
              # Ensure test_reward is a proper float for formatting
              test_reward = float(test_reward)
              avg_reward = test_reward / max(validation_steps, 1)
              
              print(f"Best model validation - Total reward: {test_reward:.4f}, Avg reward: {avg_reward:.4f}, Steps: {validation_steps}")
              
            except Exception as e:
              print(f"Model validation failed: {str(e)}")
              print("Using current model state instead")
              import traceback
              print(f"Validation error details: {traceback.format_exc()}")
            # Copy the best checkpoint to the final model location
            try:
              final_model_path = basepath + "/models/" + modelfilename + ".zip"
              shutil.copy2(best_model_path, final_model_path)
              print(f"Saved best model to {final_model_path}")
            except Exception as e:
              print(f"Failed to copy best model: {e}")
              print("Saving current model instead")
              model.save(basepath + "/models/" + modelfilename)
          else:
            print(f"No best model found, saving current model for {SYM}{prefix}")
            model.save(basepath + "/models/" + modelfilename)
        else:
          # Just save the final model
          model.save(basepath + "/models/" + modelfilename)
          print(f"Saved current model to {basepath}/models/{modelfilename}")
        # Clean up memory after model training
        cleanup_gpu_memory()
        
    # Finish tracking progress for this symbol
    if progress_tracker:
      progress_tracker.finish_symbol(SYM)

def generateposterior(rdflistp,qtnorm,SYMLISTGIVEN=TESTSYMBOLS,lol=None):
  from StockTradingEnvOptimized import StockTradingEnvOptimized
  USE_OPTIMIZED_ENV = True
  df_test_actions_list = {}
  for SYM in SYMLISTGIVEN: 
    for prefix in ["final"]:
      print('SYM:' + SYM + ' prefix:' + prefix)
      df = rdflistp[SYM+prefix]
      df['currentdate'] = df['currentt'].dt.date
      
      alldates = df['currentdate'].unique()
      traindates = alldates[:int(np.ceil(len(alldates)*TRAIN_MAX))]
      TRAINROWS = df[df.currentdate.isin(traindates)].shape[0]
      
      if prefix=="final":
        finalsignalsp = lol[SYM] 
      else:
        finalsignalsp = globalsignals
      
      if(TOPN>0):
        dropcols = finalsignalsp[TOPN:]
        df = df.drop(dropcols,axis=1)
        finalsignalsp = finalsignalsp[:TOPN]
      
      df_train = df.iloc[:TRAINROWS].reset_index(drop=True)
      df_test = df.iloc[(TRAINROWS+1):].reset_index(drop=True)
      
      try:
        qt = qtnorm[SYM]
      except:
        print("qtnorm not found for "+SYM)
        continue 
      df_train_transformed = df_train.copy()
      df_test_transformed = df_test.copy()
      df_train_transformed[finalsignalsp] = pd.DataFrame(qt.fit_transform(df_train_transformed[finalsignalsp].to_numpy()),columns=finalsignalsp)
      df_test_transformed[finalsignalsp] = pd.DataFrame(qt.transform(df_test_transformed[finalsignalsp].to_numpy()),columns=finalsignalsp)
      
      NUMVARS = len(finalsignalsp)
      df_test_transformed_list = [d.reset_index(drop=True) for _, d in df_test_transformed.groupby('currentdate')] 
      
      print('SYM:' + SYM + ' prefix:' + prefix)
      
      for iter1 in np.arange(1,1+MAXITERPOSTERIOR):
        print(iter1)
        dfs_combined = pd.DataFrame() 
        for dfs in df_test_transformed_list:
          print(np.unique(dfs.currentdate)[0])
          env2 = DummyVecEnv([lambda: StockTradingEnvOptimized(dfs,NLAGS,NUMVARS,MAXIMUM_SHORT_VALUE,INITIAL_ACCOUNT_BALANCE,MAX_STEPS,finalsignalsp,INITIAL_ACCOUNT_BALANCE, INITIAL_SHARES_HELD=0)])
          if(prefix=="final"):
            modelfilename = SYM + 'localmodel'
            print("loading " + modelfilename)
          else:
            modelfilename = 'globalmodel' + '.zip'
            print("loading " + modelfilename)
          model = PPO.load(basepath + "/models/" + modelfilename,env2,verbose=VERBOSITY)
          obs = env2.reset()
          
          actions = []
          quantity = []
          positions = []
          
          for i in np.arange(NLAGS,(dfs.shape[0])):
            action, _states = model.predict(obs,deterministic=DETERMINISTIC)
            with Capturing() as output:
              obs, rewards, done, info = env2.step([action,True])
            actions.append(action[0][0])
            quantity.append(action[0][1])
            positions.append(re.search('Position: (.+?),', output[0]).group(1))

          df_test_actions = dfs.copy()
          df_test_actions['actions'] = [0]*NLAGS + actions
          df_test_actions['quantities'] = [0]*NLAGS + quantity
          df_test_actions['positions'] = [0]*NLAGS + positions
          df_test_actions['positions'] = df_test_actions['positions'].astype(float)
          df_test_actions['buysellhold'] = ["BUY" if x >=BUYTHRESHOLD else "SELL" if x<=SELLTHRESHOLD else "HOLD" for x in df_test_actions['actions']]
          dfs_combined = pd.concat([dfs_combined, df_test_actions], ignore_index=True)
          
        df_test_actions = df_test.copy()
        df_test_actions['actions'] = dfs_combined['actions']
        df_test_actions['quantities'] = dfs_combined['quantities']
        df_test_actions['positions'] = dfs_combined['positions']
        df_test_actions['buysellhold'] = dfs_combined['buysellhold']
        df_test_actions_list[SYM+prefix+str(iter1)] = df_test_actions
  return df_test_actions_list


def closeout_position(df):
    """
    Close out all positions at the end of each trading day
    """
    # Get the last timestamp for each date
    indices_last = df.groupby('currentdate')['currentt'].max().reset_index()
    indices_last.columns = ['currentdate', 'maxt']
    
    # Get the second-to-last timestamp for each date
    df_not_last = df[~df['currentt'].isin(indices_last['maxt'])]
    if not df_not_last.empty:
        indices_before_last = df_not_last.groupby('currentdate')['currentt'].max().reset_index()
        indices_before_last.columns = ['currentdate', 'maxt']
        
        # Process each end-of-day position
        for i in range(len(indices_last)):
            current_date = indices_last.iloc[i]['currentdate']
            last_time = indices_last.iloc[i]['maxt']
            
            # Check if we have a before-last entry for this date
            before_last_rows = indices_before_last[indices_before_last['currentdate'] == current_date]
            if before_last_rows.empty:
                continue
                
            before_last_time = before_last_rows.iloc[0]['maxt']
            
            # Get position at before-last timestamp
            before_last_position_rows = df[(df['currentt'] == before_last_time) & 
                                           (df['currentdate'] == current_date)]
            if before_last_position_rows.empty:
                continue
                
            before_last_position = before_last_position_rows.iloc[0]['positions']
            
            # Update last timestamp row
            df.loc[(df['currentt'] == last_time) & (df['currentdate'] == current_date), 
                   'trade_position_size'] = abs(before_last_position)
            df.loc[(df['currentt'] == last_time) & (df['currentdate'] == current_date), 
                   'positions'] = 0
            df.loc[(df['currentt'] == last_time) & (df['currentdate'] == current_date), 
                   'positionchange'] = -before_last_position
            
            # Set action based on previous position
            if before_last_position > 0:
                df.loc[(df['currentt'] == last_time) & (df['currentdate'] == current_date), 
                       'buysellhold'] = "SELL"
                df.loc[(df['currentt'] == last_time) & (df['currentdate'] == current_date), 
                       'actions'] = -1
            elif before_last_position < 0:
                df.loc[(df['currentt'] == last_time) & (df['currentdate'] == current_date), 
                       'buysellhold'] = "BUY"
                df.loc[(df['currentt'] == last_time) & (df['currentdate'] == current_date), 
                       'actions'] = 1
            else:
                df.loc[(df['currentt'] == last_time) & (df['currentdate'] == current_date), 
                       'buysellhold'] = "HOLD"
                df.loc[(df['currentt'] == last_time) & (df['currentdate'] == current_date), 
                       'actions'] = 0
                
    return df


def plotter(plot_obj, filename):
    """
    Save plot to file
    """
    plt.savefig(os.path.join(basepath+"/plots/", f"{filename}.png"), dpi=300, bbox_inches='tight')
    plt.close()

#Indentation correct and plot explainability fix
def posteriorplots(df_test_actions_list,SYMLISTGIVEN=TESTSYMBOLS,DAYINDEX=0):
  symposterior = {}
  for SYM in SYMLISTGIVEN:
      for prefix in ["final"]:
          pnls = []
          # Iterate through each model iteration
          for iter1 in range(1, MAXITERPOSTERIOR + 1):
              key = f"{SYM}{prefix}{iter1}"
              
              if key not in df_test_actions_list:
                  print(f"Key {key} not found in df_test_actions_list")
                  continue
                  
              # Get the dataframe for this iteration
              rdf_test_actions = df_test_actions_list[key].copy()
              rdf_test_actions['currentdate'] = pd.to_datetime(rdf_test_actions['currentdate']).dt.date
              rdf_test_actions['trade_position_size'] = abs(rdf_test_actions.groupby(['currentdate'])['positions'].diff())
              rdf_test_actions.loc[~rdf_test_actions['currentdate'].duplicated(),'trade_position_size'] = rdf_test_actions.loc[~rdf_test_actions['currentdate'].duplicated(),'positions']
              rdf_test_actions['positionchange'] = rdf_test_actions.groupby(['currentdate'])['positions'].diff()
              rdf_test_actions.loc[~rdf_test_actions['currentdate'].duplicated(),'positionchange'] = rdf_test_actions.loc[~rdf_test_actions['currentdate'].duplicated(),'positions']
              rdf_test_actions = closeout_position(rdf_test_actions)
              conditions = [
                  (rdf_test_actions['positionchange'] > 0),
                  (rdf_test_actions['positionchange'] < 0),
                  (rdf_test_actions['positionchange'] == 0)
              ]
              choices = ['BUY', 'SELL', 'HOLD']
              rdf_test_actions['buysellhold'] = np.select(conditions, choices, default='HOLD')
              rdf_test_actions.loc[rdf_test_actions['trade_position_size'] < 1e-1, 'buysellhold'] = 'HOLD'
              buysell = rdf_test_actions[rdf_test_actions['buysellhold'] != 'HOLD'][
                  ['positions', 'trade_position_size', 'buysellhold', 'vwap2', 
                   'actions', 'quantities', 'currentt', 'currentdate']
              ].copy()
              conditions = [
                  (buysell['buysellhold'] == 'SELL'),
                  (buysell['buysellhold'] == 'BUY')
              ]
              choices = [
                  buysell['vwap2'] * buysell['trade_position_size'],
                  -buysell['vwap2'] * buysell['trade_position_size']
              ]
              buysell['pnl'] = np.select(conditions, choices, default=0)
              
              daily_pnl = buysell.groupby('currentdate')['pnl'].sum()
              mean_pnl = daily_pnl.mean()
              mean_percentage_pnl = np.round(mean_pnl/INITIAL_ACCOUNT_BALANCE*100,2)
              print(f"Mean daily PnL: {mean_pnl} or {mean_percentage_pnl}% for {SYM}{prefix}")
              pnls.append(mean_pnl)
          
          # Store results in symposterior dictionary
          symposterior[f"{SYM}{prefix}"] = pnls
          
          # Create visualization for median performance
          if len(pnls) > 0:
            # Find the iteration with median performance
            pnls_sorted = sorted(pnls)
            median_idx = len(pnls) // 2
            median_pnl = pnls_sorted[median_idx]
            iter1 = pnls.index(median_pnl) + 1  # +1 because iterations are 1-indexed
            
            key = f"{SYM}{prefix}{iter1}"
            if key not in df_test_actions_list:
                print(f"Key {key} not found in df_test_actions_list")
                continue
            
            rdf_test_actions = df_test_actions_list[key].copy()
            rdf_test_actions['currentdate'] = pd.to_datetime(rdf_test_actions['currentdate']).dt.date
            rdf_test_actions['trade_position_size'] = abs(rdf_test_actions.groupby(['currentdate'])['positions'].diff())
            rdf_test_actions.loc[~rdf_test_actions['currentdate'].duplicated(),'trade_position_size'] = rdf_test_actions.loc[~rdf_test_actions['currentdate'].duplicated(),'positions']
            rdf_test_actions['positionchange'] = rdf_test_actions.groupby(['currentdate'])['positions'].diff()
            rdf_test_actions.loc[~rdf_test_actions['currentdate'].duplicated(),'positionchange'] = rdf_test_actions.loc[~rdf_test_actions['currentdate'].duplicated(),'positions']
            rdf_test_actions = closeout_position(rdf_test_actions)
            conditions = [
                (rdf_test_actions['positionchange'] > 0),
                (rdf_test_actions['positionchange'] < 0),
                (rdf_test_actions['positionchange'] == 0)
            ]
            choices = ['BUY', 'SELL', 'HOLD']
            rdf_test_actions['buysellhold'] = np.select(conditions, choices, default='HOLD')
            rdf_test_actions.loc[rdf_test_actions['trade_position_size'] < 1e-1, 'buysellhold'] = 'HOLD'
            buysell = rdf_test_actions[rdf_test_actions['buysellhold'] != 'HOLD'][
                ['positions', 'trade_position_size', 'buysellhold', 'vwap2', 
                 'actions', 'quantities', 'currentt', 'currentdate']
            ].copy()
            conditions = [
                (buysell['buysellhold'] == 'SELL'),
                (buysell['buysellhold'] == 'BUY')
            ]
            choices = [
                buysell['vwap2'] * buysell['trade_position_size'],
                -buysell['vwap2'] * buysell['trade_position_size']
            ]
            buysell['pnl'] = np.select(conditions, choices, default=0)
            
            # Calculate portfolio value and performance
            
            daily_pnl = buysell.groupby('currentdate')['pnl'].sum()
            print(daily_pnl)
            median_pnl = daily_pnl.mean()
            #print(f"Median PnL: {median_pnl}")
            
            unique_dates = np.unique(rdf_test_actions.currentdate)
            if DAYINDEX >= len(unique_dates):
                # If DAYINDEX is out of bounds, skip this iteration
                continue
            SPECIFICDATE = unique_dates[DAYINDEX]
            SPECIFICPNL =  float(daily_pnl[daily_pnl.index==SPECIFICDATE])
            tdf = rdf_test_actions[rdf_test_actions.currentdate==SPECIFICDATE].reset_index(drop=True)
            
            plt.figure(figsize=(14, 14))
            lwidths = np.log10(abs(tdf.positions+1)) + 1
            if(len(np.unique(lwidths))==1):
              lwidthst =[1]*len(lwidths)
            else:
              lwidthst = 1.0 + (1.0/(max(lwidths)-min(lwidths)))*(lwidths-min(lwidths))
            colors = ['green' if xx>0 else 'red' if xx<0 else 'black' for xx in tdf['positions']]
            for piter in range(len(lwidths)-1):
              plt.plot(tdf.currentt[piter:piter+2], tdf.vwap2[piter:piter+2], linewidth=lwidthst[piter],color=colors[piter])
            
            # Create scatter plot for buy/sell signals
            colors = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'black'}
            for action in ['BUY', 'SELL',]:
                mask = tdf['buysellhold'] == action
                if mask.any():
                    plt.scatter(
                        x=tdf.loc[mask, 'currentt'],
                        y=tdf.loc[mask, 'o'],
                        c=colors[action],
                        #s = round(np.log(rdf_test_actions.loc[mask, 'positions']-rdf_test_actions.loc[mask, 'positions'].min()+1)).astype(int),
                        s=tdf.loc[mask, 'trade_position_size'],
                        label=action,
                        alpha=0.7
                    )
            
            print(f"{SYM}{prefix} - Median and SPECIFIC DATE and PNL: {median_pnl:.2f}, {SPECIFICDATE}, {SPECIFICPNL:.2f}")
            plt.title(f"{SYM}{prefix} - Median and SPECIFIC DATE and PNL: {median_pnl:.2f}, {SPECIFICDATE}, {SPECIFICPNL:.2f}")
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.xticks(rotation=25)
            plt.tight_layout()

            # Save the plot
            plot_filename = f"validationplot{SYM.replace('-', '').replace('&', '')}{prefix}{DAYINDEX}"
            plotter(plt, plot_filename)
              
  # Save symposterior results to CSV
  pd.DataFrame.from_dict(symposterior, orient='index').transpose().to_csv(
      os.path.join(basepath, "symposterior.csv")
  )
  return symposterior

def prettyprintpnl(finalprices,portfolio,cashposition,startingnetvalue,positivetrades,negativetrades):
  dfret = pd.DataFrame.from_dict({'prices':finalprices,'positions':portfolio,'cash':cashposition,'buys':positivetrades,'sells':negativetrades})
  dfret['profit'] = dfret['cash'] + dfret['prices']*dfret['positions'] - INITIAL_ACCOUNT_BALANCE
  dfret['totaltrades'] = dfret['buys'] + dfret['sells']
  return dfret


def find_best_checkpoint(SYM, prefix, METRICS_FILE=None):
    """
    Find the best model checkpoint based on mean reward
    
    Args:
        SYM: Symbol for the model
        prefix: Model prefix ("final" or other)
        METRICS_FILE: Path to custom metrics file
        
    Returns:
        Path to the best checkpoint model file
    """
    # Define checkpoint directory and metrics file
    
    try:
        # Check if the file exists
        if not os.path.exists(METRICS_FILE_PATH):
            print(f"Metrics file {METRICS_FILE_PATH} not found")
            raise FileNotFoundError(f"Metrics file {METRICS_FILE_PATH} not found")
            
        # Read metrics from file
        with open(METRICS_FILE_PATH, 'r') as f:
            lines = f.readlines()
            
        if len(lines) <= 1:  # Just header or empty
            print(f"Metrics file {METRICS_FILE_PATH} has no data")
            raise ValueError(f"Metrics file {METRICS_FILE_PATH} has no data")
            
        # Parse timesteps and mean rewards from lines
        # Format expected: timestep,mean_reward
        timesteps = []
        mean_rewards = []
        
        for line in lines[1:]:  # Skip header
            if ',' in line:
                parts = line.strip().split(',')
                if len(parts) >= 2:  # Ensure we have mean_reward column
                    try:
                        timestep = int(parts[0])
                        mean_reward = float(parts[1])  # Mean reward is in 2nd column
                        timesteps.append(timestep)
                        mean_rewards.append(mean_reward)
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing line: {line} - {e}")
                        continue
        
        if not timesteps:
            print(f"No valid data found in metrics file {METRICS_FILE_PATH}")
            raise ValueError(f"No valid data found in metrics file {METRICS_FILE_PATH}")
            
        # Find timestep with highest mean reward
        best_idx = mean_rewards.index(max(mean_rewards))
        best_timestep = timesteps[best_idx]
        best_mean_reward = mean_rewards[best_idx]
        
        print(f"Best mean reward {best_mean_reward:.4f} found at timestep {best_timestep}")
        
        # Find the closest checkpoint file
        checkpoint_files = glob.glob(CHECKPOINT_DIR + "ppo_model_*.zip")
        
        # Extract timesteps from filenames
        checkpoint_steps = []
        for f in checkpoint_files:
            match = re.search(r'ppo_model_(\d+)_steps.zip', os.path.basename(f))
            if match:
                checkpoint_steps.append((int(match.group(1)), f))
        
        if not checkpoint_steps:
            print(f"No checkpoint files found in {CHECKPOINT_DIR}")
            return None
        
        # Find closest checkpoint step to the best timestep
        checkpoint_steps.sort(key=lambda x: abs(x[0] - best_timestep))
        closest_step, best_checkpoint = checkpoint_steps[0]
        
        print(f"Best model found at step {closest_step} with mean reward {best_mean_reward:.4f}")
        
        return best_checkpoint
                
    except Exception as e:
        print(f"Error finding best checkpoint: {e}")
    
    # Fallback to latest checkpoint if metrics can't be used
    checkpoint_files = glob.glob(CHECKPOINT_DIR + "ppo_model_*.zip")
    if checkpoint_files:
        # Get latest checkpoint by timestamp
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        print(f"Using latest checkpoint file: {latest_checkpoint}")
        return latest_checkpoint
    
    print("No checkpoint files found")
    return None


def save_best_model(SYM, prefix="final", METRICS_FILE=None):
    """
    Find the best model checkpoint and save it to the models directory
    
    Args:
        SYM: Symbol for the model
        prefix: Model prefix ("final" or other)
        METRICS_FILE: Optional path to custom metrics file
    
    Returns:
        Path to the saved best model or None if no model was saved
    """
    # Find the best checkpoint
    best_checkpoint = find_best_checkpoint(SYM, prefix, METRICS_FILE)
    
    if best_checkpoint is None:
        print(f"No best checkpoint found for {SYM}{prefix}")
        return None
    
    # Define target filename
    if prefix == "final":
        target_filename = basepath + "/models/" + SYM + 'localmodel' + '.zip'
    else:
        target_filename = basepath + "/models/" + 'globalmodel' + '.zip'
    
    try:
        # Copy the best checkpoint to the models directory
        shutil.copy2(best_checkpoint, target_filename)
        print(f"Saved best model to {target_filename}")
        
        # Load and save the model to ensure proper format if needed
        try:
            model = PPO.load(target_filename)
            model.save(target_filename)
            print(f"Successfully loaded and resaved model to {target_filename}")
        except Exception as e:
            print(f"Warning: Could not load and resave model: {e}")
        
        return target_filename
    except Exception as e:
        print(f"Error saving best model: {e}")
        return None




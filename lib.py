# Core Python libraries
import collections
import datetime as dt
import gc
import glob
import io
import itertools
import json
import math
import os
import random
import re
import socket
import subprocess
import sys
import time
import warnings
import zipfile
from copy import copy
from datetime import datetime, timedelta, date
from io import StringIO
from typing import Tuple

# Data science libraries
import numpy as np
import pandas as pd
import pytz
import scipy
from scipy import stats
from scipy.special import kl_div
from scipy.stats import wasserstein_distance

# Visualization libraries
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import LineCollection

# Machine learning libraries
import gymnasium as gym
import joblib
import onnx
import onnxruntime as ort
import torch
from gymnasium import spaces
from sklearn.preprocessing import QuantileTransformer

# Stable Baselines3 and RL libraries
from stable_baselines3 import PPO, SAC, A2C, DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import (
    BaseCallback, CallbackList, CheckpointCallback
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.sac.policies import SACPolicy
from sb3_contrib import RecurrentPPO

# Third-party libraries
import requests

# Local imports
import kitelogin
from parameters import *

os.chdir(basepath)
# Note: StockTradingEnv2 import moved to avoid circular imports in multiprocessing

warnings.filterwarnings("ignore")
sys.setrecursionlimit(10000)



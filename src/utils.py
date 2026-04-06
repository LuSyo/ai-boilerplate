import argparse
import random
import numpy as np
# import torch
import datetime
import os
import logging
import sys

def parse_args():
  date_str = datetime.datetime.now().strftime('%Y-%m-%d')

  parser = argparse.ArgumentParser(description="CEVAE-HE Training and Testing Pipeline")

  parser.add_argument('--seed', type=int, default=Config.SEED)

  parser.add_argument('--exp_name', type=str, default=date_str)
  parser.add_argument('--run_name', type=str, default=date_str)

  parser.add_argument('--query', type=str, required=True)

  return parser.parse_args()
  
def setup_logger(log_dir, exp_name):
  os.makedirs(log_dir, exist_ok=True)
  log_path = os.path.join(log_dir, f"{exp_name}.log")

  # Create a custom logger
  logger = logging.getLogger(exp_name)
  logger.setLevel(logging.INFO)

  if not logger.handlers:
      # Formatter: Timestamp | Level | Message
      formatter = logging.Formatter(
          '%(asctime)s | %(levelname)s | %(message)s', 
          datefmt='%Y-%m-%d %H:%M:%S'
      )

      # File Handler
      file_handler = logging.FileHandler(log_path)
      file_handler.setFormatter(formatter)
      logger.addHandler(file_handler)

      # Console Handler
      console_handler = logging.StreamHandler(sys.stdout)
      console_handler.setFormatter(formatter)
      logger.addHandler(console_handler)

  return logger

def set_global_seeds(seed):
  random.seed(seed)
  np.random.seed(seed)
  # torch.manual_seed(seed)
  # torch.cuda.manual_seed_all(seed)

class Config:
   DATA_DIR = './data'
   SOURCES_DIR = './data/sources'
   LOG_DIR = './logs'
   RESULTS_DIR = './results'
   EVAL_DIR = './data/eval'

   MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
   # mlflow ui --backend-store-uri sqlite:///mlflow.db to see results

   SEED = 4
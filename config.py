from easydict import EasyDict as edict
import torch

import os
from pathlib import Path

CONFIG = edict()
CONFIG.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONFIG.N_GPUS = torch.cuda.device_count()
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
temp_path = Path(ROOT_DIR)
ROOT_DIR = temp_path.absolute()

CONFIG.TEMP_SUB_EXP_PATH = ROOT_DIR

CONFIG.VIT = edict()
CONFIG.VIT.NAME = "VIT_MODEL"
CONFIG.VIT.BATCH_SIZE = 64
CONFIG.VIT.LOSS = "CROSS_ENTROPY"
CONFIG.VIT.N_EPOCHS = 100
CONFIG.VIT.LR = 0.005
CONFIG.VIT.DS_NAME = "MNIST"
CONFIG.VIT.TYPE = "PATCHES"


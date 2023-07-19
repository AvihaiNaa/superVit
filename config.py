from easydict import EasyDict as edict
import torch

CONFIG = edict()
CONFIG.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONFIG.N_GPUS = torch.cuda.device_count()
CONFIG.TEMP_SUB_EXP_PATH = '/home/avihaina/projects/superVit'

CONFIG.VIT = edict()
CONFIG.VIT.NAME = "VIT_MODEL"
CONFIG.VIT.BATCH_SIZE = 64
CONFIG.VIT.LOSS = "CROSS_ENTROPY"
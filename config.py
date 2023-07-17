from easydict import EasyDict as edict
import torch

CONFIG = edict()
CONFIG.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONFIG.N_GPUS = torch.cuda.device_count()


CONFIG.VIT = edict()
CONFIG.VIT.BATCH_SIZE = 64

import random
import os
import numpy as np
import torch

def set_seed(seed=7):       #函数用于设置随机种子，确保实验的可重复性
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


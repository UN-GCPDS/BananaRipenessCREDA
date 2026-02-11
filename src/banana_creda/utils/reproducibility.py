import torch
import numpy as np
import random
import os

def set_seed(seed: int = 42):
    """
    Fixes seeds for reproducibility of experiments.
    """
    # 1. Basic seeds of Python and Numpy
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    # 2. PyTorch seeds (Requested)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    
    # 3. CuDNN configuration for determinism
    # Nota: Esto puede reducir ligeramente el rendimiento pero asegura consistencia
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"✅ Semillas fijadas en: {seed}")
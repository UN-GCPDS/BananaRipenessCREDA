import torch
import numpy as np
import random
import os

def set_seed(seed: int = 42):
    """
    Fija las semillas para garantizar la reproducibilidad de los experimentos.
    """
    # 1. Semillas básicas de Python y Numpy
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    # 2. Semillas de PyTorch (Solicitadas)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    
    # 3. Configuración de CuDNN para determinismo
    # Nota: Esto puede reducir ligeramente el rendimiento pero asegura consistencia
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"✅ Semillas fijadas en: {seed}")
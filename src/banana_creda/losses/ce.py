import torch.nn as nn
import torch.nn.functional as F

class CELoss(nn.Module):
    """
    Standard Cross-Entropy Loss for baseline classification.
    No Domain Adaptation components are included.
    """
    def __init__(self, config=None, num_classes=None):
        super().__init__()
        # Config y num_classes se reciben para mantener la compatibilidad 
        # de la firma con tu inicialización anterior, aunque no se usen.

    def forward(self, logits, labels):
        return F.cross_entropy(logits, labels)
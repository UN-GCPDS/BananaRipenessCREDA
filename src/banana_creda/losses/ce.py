import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

class CELoss(nn.Module):
    """Standard Cross-Entropy Loss for baseline classification.
    
    No Domain Adaptation components are included in this loss.
    """
    def __init__(self, config: Any = None, num_classes: int | None = None):
        """Initializes the CELoss.
        
        Args:
            config (Optional[Any]): Configuration object (ignored, kept for compatibility).
            num_classes (Optional[int]): Number of classes (ignored, kept for compatibility).
        """
        super().__init__()
        # Config and num_classes are received to maintain signature consistency
        # with DA losses, although they are not used here.

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Computes the cross-entropy loss.
        
        Args:
            logits (torch.Tensor): Predicted class logits.
            labels (torch.Tensor): Ground truth labels.
            
        Returns:
            torch.Tensor: Scalar loss value.
        """
        return F.cross_entropy(logits, labels)
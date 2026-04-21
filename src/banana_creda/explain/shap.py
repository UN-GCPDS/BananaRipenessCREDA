"""
Gradient SHAP Implementation for Banana Ripeness Detection
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
from torch import Tensor
from pathlib import Path
from captum.attr import GradientShap

class SHAP:
    """
    A wrapper for Captum's GradientShap. 
    Uses an averaged baseline from representative samples to reduce background noise.
    """

    def __init__(self, model: nn.Module):
        """
        Initializes the SHAP explainer.
        """
        self.model = model
        # Get device from model parameters
        self.device = next(self.model.parameters()).device
        self.gs = GradientShap(self.model)
        self.overlays = {}

    def _generate_baselines(self, img_stack: Tensor) -> Tensor:
        """
        Generates a baseline by averaging the 12 representative samples.
        The average acts as a neutral reference for the specific class/environment.
        """
        # img_stack shape: [12, 3, 224, 224] -> mean shape: [1, 3, 224, 224]
        avg_baseline = img_stack.mean(dim=0, keepdim=True)
        return avg_baseline.to(self.device)

    def generate_overlays(
        self, 
        samples: Dict[int, Tensor], 
        n_samples: int = 25, 
        stdevs: float = 0.02,
        alpha: float = 0.4
    ) -> None:
        """
        Computes SHAP and overlays the heatmap on the first image of each class stack.
        """
        self.overlays = {}
        cmap = plt.get_cmap('hot') 

        for target_class, img_stack in samples.items():
            # 1. Prepare Input (using the first image of the 12 collected)
            img_tensor = img_stack[0]
            input_img = img_tensor.unsqueeze(0).to(self.device).requires_grad_(True)
            
            # 2. Prepare Baseline (averaging the entire stack of 12)
            baselines = self._generate_baselines(img_stack)

            # 3. Compute Attribution
            # n_samples increased to 25 for better expectation convergence
            attr = self.gs.attribute(
                input_img, 
                baselines=baselines, 
                target=target_class, 
                n_samples=n_samples, 
                stdevs=stdevs
            )

            # 4. Process Attribution (Channel Aggregation)
            attr_combined = torch.sum(torch.abs(attr), dim=1).squeeze().detach().cpu().numpy()

            # 5. Normalize Heatmap
            attr_min, attr_max = attr_combined.min(), attr_combined.max()
            attr_norm = (attr_combined - attr_min) / (attr_max - attr_min + 1e-8)
            
            heatmap = cmap(attr_norm)[..., :3] 
            heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float() 

            # 6. Overlay Logic
            img_vis = img_tensor.detach().cpu()
            img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min() + 1e-8)
            
            overlaid = (1 - alpha) * img_vis + alpha * heatmap
            self.overlays[target_class] = overlaid

    def save_overlays(self, output_dir: str) -> None:
        """
        Saves the SHAP overlaid images.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        for target_class, overlaid in self.overlays.items():
            save_file = out_path / f"shap_class_{target_class}.png"
            plt.imsave(save_file, overlaid.permute(1, 2, 0).numpy())
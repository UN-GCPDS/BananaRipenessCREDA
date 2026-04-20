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
    Computes fine-grained pixel attributions compared to a baseline distribution.
    """

    def __init__(self, model: nn.Module):
        """
        Initializes the SHAP explainer.

        Args:
            model (nn.Module): The pretrained BananaModel instance.
        """
        self.model = model
        # Get device from model parameters
        self.device = next(self.model.parameters()).device
        self.gs = GradientShap(self.model)
        self.overlays = {}

    def _generate_baselines(self, input_img: Tensor, n_baseline_samples: int = 5) -> Tensor:
        """
        Generates a baseline distribution. 
        Commonly uses black images (zeros) or random noise.
        """
        # Create a distribution of black images as the reference
        return torch.zeros((n_baseline_samples,) + input_img.shape[1:]).to(self.device)

    def generate_overlays(
        self, 
        samples: Dict[int, Tensor], 
        n_samples: int = 5, 
        stdevs: float = 0.01,
        alpha: float = 0.5
    ) -> None:
        """
        Computes Gradient SHAP attributions and creates overlays.

        Args:
            samples (Dict[int, Tensor]): Class indices and image tensors.
            n_samples (int): Number of random samples for expectations.
            stdevs (float): Noise standard deviation added to inputs.
            alpha (float): Blending factor for the overlay.
        """
        self.overlays = {}
        cmap = plt.get_cmap('hot') # 'hot' or 'jet' are good for pixel attribution

        for target_class, img_tensor in samples.items():
            # 1. Prepare Input and Baselines
            input_img = img_tensor.unsqueeze(0).to(self.device).requires_grad_(True)
            baselines = self._generate_baselines(input_img)

            # 2. Compute Attribution
            # Dimensions match input: [1, 3, 224, 224]
            attr = self.gs.attribute(
                input_img, 
                baselines=baselines, 
                target=target_class, 
                n_samples=n_samples, 
                stdevs=stdevs
            )

            # 3. Process Attribution for Visualization
            # SHAP returns [C, H, W]. We aggregate across channels (sum of absolute values)
            # to get a single heatmap indicating 'importance' regardless of color.
            attr_combined = torch.sum(torch.abs(attr), dim=1).squeeze().detach().cpu().numpy()

            # 4. Normalize Heatmap
            attr_min, attr_max = attr_combined.min(), attr_combined.max()
            attr_norm = (attr_combined - attr_min) / (attr_max - attr_min + 1e-8)
            
            heatmap = cmap(attr_norm)[..., :3]  # Drop Alpha channel
            heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float() 

            # 5. Overlay Logic
            img_vis = img_tensor.detach().cpu()
            # Normalize original image to [0, 1] for blending
            img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min() + 1e-8)
            
            overlaid = (1 - alpha) * img_vis + alpha * heatmap
            self.overlays[target_class] = overlaid

    def save_overlays(self, output_dir: str) -> None:
        """
        Saves the SHAP overlaid images to the specified directory.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        for target_class, overlaid in self.overlays.items():
            save_file = out_path / f"shap_class_{target_class}.png"
            plt.imsave(save_file, overlaid.permute(1, 2, 0).numpy())
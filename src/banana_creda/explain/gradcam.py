"""
Grad-CAM Implementation for Banana Ripeness Detection
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from torch import Tensor
from torchvision import models
from pathlib import Path
from captum.attr import LayerGradCam, LayerAttribution

class GradCAM:
    """
    A robust wrapper for Captum's LayerGradCam supporting ResNet, ViT, 
    EfficientNet, and MobileNet architectures.
    """

    def __init__(self, model: nn.Module, model_name: str):
        """
        Initializes the explainer by identifying the appropriate target layer.

        Args:
            model (nn.Module): The pretrained model instance.
            model_name (str): One of ['resnet34', 'vit_b_16', 'efficientnet_b0', 'mobilenet_v3_large'].
        """
        self.model = model
        self.model_name = model_name.lower()
        self.target_layer = self._get_target_layer()
        self.lgc = LayerGradCam(self.model, self.target_layer)

    def _get_target_layer(self) -> nn.Module:
        """
        Maps the model name to the recommended last structural layer for GradCAM.
        """
        if "resnet34" in self.model_name:
            return self.model.layer4
        elif "vit_b_16" in self.model_name:
            return self.model.encoder.ln
        elif "efficientnet_b0" in self.model_name:
            return self.model.features[-1]
        elif "mobilenet_v3_large" in self.model_name:
            return self.model.features[-1]
        else:
            raise ValueError(f"Architecture {self.model_name} not explicitly supported.")

    def _reshape_transform(self, attr: Tensor) -> Tensor:
        """
        Reshapes ViT tokens back into a spatial grid if necessary.
        """
        if "vit" in self.model_name:
            # Shape is [Batch, 1, 197]. 197 = 1 (CLS) + 14x14 (Patches)
            # Remove CLS token and reshape to 14x14
            attr = attr[:, :, 1:].reshape(attr.shape[0], 1, 14, 14)
        return attr

    def generate_overlays(
        self, 
        samples: Dict[int, Tensor], 
        alpha: float = 0.5, 
        image_size: int = 224
    ) -> Dict[int, Tensor]:
        """
        Computes GradCAM and overlays the heatmap on the original images.

        Args:
            samples (Dict[int, Tensor]): Keys are target class indices, 
                                         values are image tensors [C, H, W].
            alpha (float): Blending factor (0 = only image, 1 = only heatmap).
            image_size (int): Spatial size for upsampling.

        Returns:
            Dict[int, Tensor]: Dictionary with class indices and overlaid tensors [3, H, W].
        """
        self.overlays = {}
        cmap = plt.get_cmap('jet')
        image_size = (image_size, image_size)

        for target_class, img_tensor in samples.items():
            # Add batch dimension [1, C, H, W]
            input_img = img_tensor.unsqueeze(0).requires_grad_(True).to(self.model.device)

            # 1. Compute Attribution
            attr = self.lgc.attribute(input_img, target=target_class, relu_attributions=True)
            
            # 2. Architecturespecific reshaping
            attr = self._reshape_transform(attr)
            
            # 3. Upsample to original image size
            upsampled_attr = LayerAttribution.interpolate(
                attr, image_size, interpolate_mode='bilinear'
            ).squeeze().detach().cpu().numpy()

            # 4. Normalize heatmap for colormap
            upsampled_attr = (upsampled_attr - upsampled_attr.min()) / (upsampled_attr.max() - upsampled_attr.min() + 1e-8)
            heatmap = cmap(upsampled_attr)[..., :3]  # Drop Alpha channel from CMAP
            heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float() # [3, H, W]

            # 5. Overlay logic (assumes img_tensor is in [0, 1] range)
            # If image is normalized, we un-normalize or clamp for visualization
            img_vis = img_tensor.detach().cpu()
            img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min() + 1e-8)
            
            overlaid = (1 - alpha) * img_vis + alpha * heatmap
            self.overlays[target_class] = overlaid

    def save_overlays(self, output_dir: str) -> None:
        """
        Saves the overlaid images to the specified directory.

        Args:
            output_dir (str): Directory to save the overlaid images.
        """
        for target_class, overlaid in self.overlays.items():
            save_file = Path(output_dir) / f"overlay_{target_class}.png"
            plt.imsave(save_file, overlaid.permute(1, 2, 0).numpy())
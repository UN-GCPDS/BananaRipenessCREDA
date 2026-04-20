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
    EfficientNet, and MobileNet architectures within the BananaModel framework.
    """

    def __init__(self, model: nn.Module, model_name: str):
        """
        Initializes the explainer by identifying the appropriate target layer.
        """
        self.model = model
        self.model_name = model_name.lower()
        # Get device from model parameters
        self.device = next(self.model.parameters()).device
        
        self.target_layer = self._get_target_layer()
        self.lgc = LayerGradCam(self.model, self.target_layer)
        self.overlays = {}

    def _get_target_layer(self) -> nn.Module:
        """
        Maps the model name to the specific index in the BananaModel.encoder.
        Based on backbones.py architecture.
        """
        try:
            if "resnet" in self.model_name:
                # encoder[4] is nn.Sequential(layer4, avgpool)
                return self.model.encoder[4][0] 
            elif "vit" in self.model_name:
                # encoder[4] is nn.Sequential(blocks 9-11, ln)
                return self.model.encoder[4][-1]
            elif "efficientnet" in self.model_name:
                # encoder[4] is nn.Sequential(blocks 7-8, conv_head)
                return self.model.encoder[4][-1]
            elif "mobilenet" in self.model_name:
                # encoder[4] is nn.Sequential(blocks 13-16, conv_head)
                return self.model.encoder[4][-1]
            else:
                raise ValueError(f"Architecture {self.model_name} not supported.")
        except (IndexError, AttributeError) as e:
            raise ValueError(f"Could not find target layer for {self.model_name}: {e}")

    def _reshape_transform(self, attr: Tensor) -> Tensor:
        """
        Reshapes ViT tokens back into a spatial grid.
        """
        if "vit" in self.model_name:
            # Captum returns [Batch, 1, 197] for the LayerNorm layer
            # 197 tokens = 1 CLS + 196 (14x14 patches)
            if attr.shape[-1] == 197:
                attr = attr[:, :, 1:].reshape(attr.shape[0], 1, 14, 14)
        return attr

    def generate_overlays(
        self, 
        samples: Dict[int, Tensor], 
        alpha: float = 0.5, 
        image_size: int = 224
    ) -> None:
        """
        Computes GradCAM and overlays the heatmap on the original images.
        """
        self.overlays = {}
        cmap = plt.get_cmap('jet')
        spatial_size = (image_size, image_size)

        for target_class, img_tensor in samples.items():
            # Ensure input is on the correct device
            input_img = img_tensor.unsqueeze(0).to(self.device).requires_grad_(True)

            # 1. Compute Attribution
            attr = self.lgc.attribute(input_img, target=target_class, relu_attributions=True)
            
            # 2. Architecture-specific reshaping (for ViT)
            attr = self._reshape_transform(attr)
            
            # 3. Upsample to original image size
            upsampled_attr = LayerAttribution.interpolate(
                attr, spatial_size, interpolate_mode='bilinear'
            ).squeeze().detach().cpu().numpy()

            # 4. Normalize heatmap
            attr_min, attr_max = upsampled_attr.min(), upsampled_attr.max()
            upsampled_attr = (upsampled_attr - attr_min) / (attr_max - attr_min + 1e-8)
            
            heatmap = cmap(upsampled_attr)[..., :3]  # Drop Alpha channel
            heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float() 

            # 5. Overlay logic
            img_vis = img_tensor.detach().cpu()
            img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min() + 1e-8)
            
            overlaid = (1 - alpha) * img_vis + alpha * heatmap
            self.overlays[target_class] = overlaid

    def save_overlays(self, output_dir: str) -> None:
        """
        Saves the overlaid images to the specified directory.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        for target_class, overlaid in self.overlays.items():
            save_file = out_path / f"gradcam_class_{target_class}.png"
            # Ensure proper range for imsave
            plt.imsave(save_file, overlaid.permute(1, 2, 0).numpy())
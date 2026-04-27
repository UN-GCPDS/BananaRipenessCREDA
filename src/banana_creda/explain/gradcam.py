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
        self.device = next(self.model.parameters()).device
        
        self.target_layer = self._get_target_layer()
        self.lgc = LayerGradCam(self.model, self.target_layer)
        self.overlays = {}

    def _get_target_layer(self) -> nn.Module:
        """
        Maps the model name to the specific layer in BananaModel.encoder.

        For ViT we must hook the last transformer block (encoder[4][-2], i.e.
        blocks[11]), NOT the final LayerNorm. The LayerNorm outputs [B, 768]
        after Captum's gradient pooling, which has no spatial structure.
        blocks[11] outputs [B, 197, 768] — 197 tokens (1 CLS + 14x14 patches)
        that can be reshaped into a spatial heatmap.
        """
        try:
            if "resnet" in self.model_name:
                # encoder[4] is nn.Sequential(layer4, avgpool)
                return self.model.encoder[4][0]
            elif "vit" in self.model_name:
                # encoder[4] is nn.Sequential(blocks[9], blocks[10], blocks[11], ln)
                # Hook blocks[11] (index -2), which outputs [B, 197, 768]
                return self.model.encoder[4][-2]
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

    def _reshape_vit_attr(self, attr: Tensor) -> Tensor:
        """
        Reshapes a ViT attribution tensor into a 2D spatial grid [B, 1, H, W].

        LayerGradCam on a transformer block returns [B, 1, N] where N=197
        (1 CLS token + 14x14=196 patch tokens for vit_b_16).
        We drop the CLS token and reshape patches into a square grid.
        """
        print(f"[GradCAM DEBUG] Raw ViT attr shape: {attr.shape}")

        if attr.dim() != 3:
            raise ValueError(f"Expected 3D ViT attribution [B, 1, N], got {attr.shape}")

        n_tokens = attr.shape[-1]

        # Strip CLS token if sequence length is not a perfect square
        grid_size = int(n_tokens ** 0.5)
        if grid_size * grid_size != n_tokens:
            attr = attr[:, :, 1:]          # drop CLS → [B, 1, 196]
            n_tokens = attr.shape[-1]
            grid_size = int(n_tokens ** 0.5)

        if grid_size * grid_size != n_tokens:
            raise ValueError(
                f"ViT attribution length {n_tokens} is not a perfect square after "
                f"stripping the CLS token. Cannot reshape into a spatial grid."
            )

        attr = attr.reshape(attr.shape[0], 1, grid_size, grid_size)
        print(f"[GradCAM DEBUG] Reshaped ViT attr: {attr.shape}")
        return attr

    def generate_overlays(
        self,
        samples: Dict[int, Tensor],
        alpha: float = 0.5,
        image_size: int = 224
    ) -> None:
        """
        Computes GradCAM and overlays the heatmap on the original images.
        Uses the first image of the provided stack for each class.
        """
        self.overlays = {}
        cmap = plt.get_cmap('jet')
        spatial_size = (image_size, image_size)

        for target_class, img_stack in samples.items():
            img_tensor = img_stack[0]
            input_img = img_tensor.unsqueeze(0).to(self.device).requires_grad_(True)

            # 1. Compute attribution
            attr = self.lgc.attribute(input_img, target=target_class, relu_attributions=True)

            # 2. Reshape ViT sequence tokens → spatial grid [B, 1, 14, 14]
            if "vit" in self.model_name:
                attr = self._reshape_vit_attr(attr)

            # 3. Upsample to original image size
            upsampled_attr = LayerAttribution.interpolate(
                attr, spatial_size, interpolate_mode='bilinear'
            ).squeeze().detach().cpu().numpy()

            # 4. Normalize heatmap to [0, 1]
            attr_min, attr_max = upsampled_attr.min(), upsampled_attr.max()
            upsampled_attr = (upsampled_attr - attr_min) / (attr_max - attr_min + 1e-8)

            heatmap = cmap(upsampled_attr)[..., :3]   # drop alpha → [H, W, 3]
            heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float()

            # 5. Blend heatmap with original image
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
            plt.imsave(save_file, overlaid.permute(1, 2, 0).numpy())
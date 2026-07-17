"""
Grad-CAM Implementation for Banana Ripeness Detection
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from torch import Tensor
from pathlib import Path
from captum.attr import LayerGradCam, LayerAttribution


class GradCAM:
    """
    A robust wrapper for Captum's LayerGradCam supporting ResNet, ViT,
    EfficientNet, and MobileNet architectures within the BananaModel framework.

    For ViT, the last Linear layer of the final EncoderBlock's MLP is hooked
    (encoder[4][2].mlp[3]), whose output is [B, 197, 768]. We use
    attr_dim_summation=False so Captum does not average over tokens, then
    manually average over features (768) and reshape to [B, 1, 14, 14].
    """

    def __init__(self, model: nn.Module, model_name: str):
        self.model = model
        self.model_name = model_name.lower()
        self.device = next(self.model.parameters()).device
        self.target_layer = self._get_target_layer()
        self.lgc = LayerGradCam(self.model, self.target_layer)
        self.overlays = {}

    def _get_target_layer(self) -> nn.Module:
        """
        Returns the layer to hook based on the model architecture.

        ViT: encoder[4][2].mlp[3] — last Linear of the MLP in block 11.
             Output: [B, 197, 768] (197 tokens x 768 features).
        CNN: last convolutional/conv_head layer of encoder[4].
             Output: [B, C, H, W] — directly spatial.
        """
        try:
            if "resnet" in self.model_name:
                return self.model.encoder[4][-2]
            elif "vit" in self.model_name:
                # mlp[3] = last Linear of the MLP in the last EncoderBlock
                return self.model.encoder[4][2].mlp[3]
            elif "efficientnet" in self.model_name:
                return self.model.encoder[4][-2]
            elif "mobilenet" in self.model_name:
                return self.model.encoder[4][-2]
            else:
                raise ValueError(f"Architecture {self.model_name} is not supported.")
        except (IndexError, AttributeError) as e:
            raise ValueError(f"Could not find target layer for {self.model_name}: {e}")

    def _postprocess_vit_attr(self, attr: Tensor) -> Tensor:
        """
        Converts ViT attribution from [B, 197, 768] to [B, 1, 14, 14].

        Steps:
          1. attr_dim_summation=False → Captum returns [B, 197, 768]
          2. Mean over features (dim=-1) → [B, 197]
          3. Strip CLS token (index 0)   → [B, 196]
          4. Reshape into spatial grid   → [B, 1, 14, 14]
        """
        attr = attr.mean(dim=-1, keepdim=False)        # [B, 197]
        attr = attr[:, 1:]                              # [B, 196] — strip CLS
        attr = attr.reshape(attr.shape[0], 1, 14, 14)  # [B, 1, 14, 14]
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
            if "vit" in self.model_name:
                # attr_dim_summation=False preserves the token dimension
                attr = self.lgc.attribute(
                    input_img,
                    target=target_class,
                    relu_attributions=True,
                    attr_dim_summation=False
                )
                attr = self._postprocess_vit_attr(attr)   # → [B, 1, 14, 14]
            else:
                attr = self.lgc.attribute(
                    input_img,
                    target=target_class,
                    relu_attributions=True
                )

            # 2. Upsample to original image size
            upsampled_attr = LayerAttribution.interpolate(
                attr, spatial_size, interpolate_mode='bilinear'
            ).squeeze().detach().cpu().numpy()

            # 3. Normalize heatmap to [0, 1]
            attr_min, attr_max = upsampled_attr.min(), upsampled_attr.max()
            upsampled_attr = (upsampled_attr - attr_min) / (attr_max - attr_min + 1e-8)

            heatmap = cmap(upsampled_attr)[..., :3]   # drop alpha → [H, W, 3]
            heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float()

            # 4. Blend heatmap with original image
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
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


class _TransposeWrapper(nn.Module):
    """
    Wraps a ViT EncoderBlock and transposes its output from
    [B, N, C] → [B, C, N] so that Captum's LayerGradCam pools
    over the channel dim (C) and preserves N as the spatial dim.
    Without this, Captum pools over N and returns [B, 1, 768]
    (one value per feature) instead of [B, 1, 196] (one value per token).
    """
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x).transpose(1, 2)   # [B, N, C] → [B, C, N]


class GradCAM:
    """
    A robust wrapper for Captum's LayerGradCam supporting ResNet, ViT,
    EfficientNet, and MobileNet architectures within the BananaModel framework.
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
        Maps the model name to the specific layer in BananaModel.encoder.

        For ViT, encoder[4] = Sequential(blocks[9], blocks[10], blocks[11], LayerNorm).
        We wrap blocks[11] (index -2) in _TransposeWrapper so Captum receives
        [B, C, N] and pools over C, yielding [B, 1, 196] — one value per patch token.
        """
        try:
            if "resnet" in self.model_name:
                return self.model.encoder[4][0]

            elif "vit" in self.model_name:
                # Replace encoder[4][-2] in-place with the transpose wrapper
                # so the hooked layer produces [B, C, N] for Captum.
                last_block = self.model.encoder[4][-2]   # EncoderBlock (blocks[11])
                wrapped = _TransposeWrapper(last_block)
                # Patch it into the Sequential so the forward pass still works
                children = list(self.model.encoder[4].children())
                children[-2] = wrapped
                self.model.encoder[4] = nn.Sequential(*children)
                return wrapped

            elif "efficientnet" in self.model_name:
                return self.model.encoder[4][-1]

            elif "mobilenet" in self.model_name:
                return self.model.encoder[4][-1]

            else:
                raise ValueError(f"Architecture {self.model_name} not supported.")

        except (IndexError, AttributeError) as e:
            raise ValueError(f"Could not find target layer for {self.model_name}: {e}")

    def _reshape_vit_attr(self, attr: Tensor) -> Tensor:
        """
        Reshapes ViT attribution [B, 1, N] → [B, 1, H, W].

        After the _TransposeWrapper, Captum returns [B, 1, 196] where
        196 = 14x14 patch tokens (no CLS token in this model).
        """
        print(f"[GradCAM DEBUG] Raw ViT attr shape: {attr.shape}")

        n_tokens = attr.shape[-1]
        grid_size = int(n_tokens ** 0.5)

        # Strip CLS token if present (sequence not a perfect square)
        if grid_size * grid_size != n_tokens:
            attr = attr[:, :, 1:]
            n_tokens = attr.shape[-1]
            grid_size = int(n_tokens ** 0.5)

        if grid_size * grid_size != n_tokens:
            raise ValueError(
                f"ViT attribution length {n_tokens} is not a perfect square "
                f"after stripping the CLS token. Got {attr.shape}."
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

            # 2. Reshape ViT tokens → spatial grid [B, 1, 14, 14]
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
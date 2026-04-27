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

    For ViT, LayerGradCam cannot be used directly because EncoderBlock outputs
    [B, N, C] (tokens-first) while Captum expects [B, C, ...] (channels-first).
    Instead, we register manual forward/backward hooks on the last EncoderBlock
    to capture activations and gradients, then compute GradCAM by hand.
    """

    def __init__(self, model: nn.Module, model_name: str):
        self.model = model
        self.model_name = model_name.lower()
        self.device = next(self.model.parameters()).device

        self.target_layer = self._get_target_layer()

        # For non-ViT models, use Captum as normal
        if "vit" not in self.model_name:
            self.lgc = LayerGradCam(self.model, self.target_layer)
        else:
            self.lgc = None   # ViT uses manual hooks instead

        self.overlays = {}

    def _get_target_layer(self) -> nn.Module:
        """
        Maps the model name to the specific layer in BananaModel.encoder.

        ViT: encoder[4][-2] = blocks[11] (last EncoderBlock, before LayerNorm).
             Outputs [B, N, C] = [B, 196, 768].
        """
        try:
            if "resnet" in self.model_name:
                return self.model.encoder[4][0]
            elif "vit" in self.model_name:
                return self.model.encoder[4][-2]   # EncoderBlock (blocks[11])
            elif "efficientnet" in self.model_name:
                return self.model.encoder[4][-1]
            elif "mobilenet" in self.model_name:
                return self.model.encoder[4][-1]
            else:
                raise ValueError(f"Architecture {self.model_name} not supported.")
        except (IndexError, AttributeError) as e:
            raise ValueError(f"Could not find target layer for {self.model_name}: {e}")

    def _compute_vit_gradcam(self, input_img: Tensor, target_class: int) -> Tensor:
        """
        Manually computes GradCAM for ViT using forward/backward hooks.

        EncoderBlock outputs [B, N, C]:
          - N = 196 patch tokens (14x14 grid, no CLS in this model)
          - C = 768 hidden dim

        GradCAM formula:
          1. activations A: [B, N, C]
          2. gradients  G: [B, N, C]
          3. weights = mean(G, dim=1) → [B, C]          (pool over tokens)
          4. cam = relu(sum(weights * A, dim=-1)) → [B, N]  (weight channels)
          5. reshape N → 14x14, upsample to [224, 224]
        """
        activations = {}
        gradients = {}

        def fwd_hook(module, inp, out):
            activations['value'] = out  # [B, N, C]

        def bwd_hook(module, grad_in, grad_out):
            gradients['value'] = grad_out[0]  # [B, N, C]

        fwd_handle = self.target_layer.register_forward_hook(fwd_hook)
        bwd_handle = self.target_layer.register_full_backward_hook(bwd_hook)

        try:
            self.model.zero_grad()
            output = self.model(input_img)                        # [B, num_classes]
            score = output[0, target_class]
            score.backward()
        finally:
            fwd_handle.remove()
            bwd_handle.remove()

        A = activations['value']   # [B, N, C]
        G = gradients['value']     # [B, N, C]

        # Pool gradients over token dim → importance weight per channel
        weights = G.mean(dim=1)    # [B, C]

        # Weighted sum over channels → one scalar per token
        cam = (weights.unsqueeze(1) * A).sum(dim=-1)   # [B, N]
        cam = torch.relu(cam)                           # ReLU

        # Reshape flat token sequence → 2D spatial grid
        n_tokens = cam.shape[-1]
        grid_size = int(n_tokens ** 0.5)
        if grid_size * grid_size != n_tokens:
            raise ValueError(
                f"Cannot reshape {n_tokens} tokens into a square grid. "
                f"Expected a perfect square (e.g. 196 = 14x14)."
            )
        cam = cam.reshape(cam.shape[0], 1, grid_size, grid_size)  # [B, 1, 14, 14]
        print(f"[GradCAM DEBUG] ViT CAM shape before upsample: {cam.shape}")
        return cam

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

            # 1. Compute attribution map [B, 1, H, W]
            if "vit" in self.model_name:
                attr = self._compute_vit_gradcam(input_img, target_class)
            else:
                attr = self.lgc.attribute(
                    input_img, target=target_class, relu_attributions=True
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
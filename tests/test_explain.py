import pytest
pytest.importorskip("captum")

import torch
import numpy as np
from banana_creda.models.backbones import BananaModel
from banana_creda.config import ModelConfig
from banana_creda.explain.gradcam import GradCAM
from banana_creda.explain.shap import SHAP

@pytest.mark.parametrize("backbone", ["resnet", "efficientnet", "mobilenetv3", "vit"])
def test_gradcam_and_shap(backbone):
    config = ModelConfig(backbone=backbone, num_classes=4, pretrained=False)
    model = BananaModel(config)
    model.eval()

    # Create dummy samples
    # Dict[target_class, img_stack] where img_stack is [12, 3, 224, 224]
    samples = {
        0: torch.randn(12, 3, 224, 224),
        1: torch.randn(12, 3, 224, 224),
    }

    # Test GradCAM
    try:
        explainer = GradCAM(model, backbone)
        
        # Verify the target layer is not an AvgPool layer
        target_layer_name = explainer.target_layer.__class__.__name__
        print(f"Backbone: {backbone}, Target layer: {target_layer_name}")
        assert "AvgPool" not in target_layer_name, f"Target layer for {backbone} should not be pooling layer: {target_layer_name}"
        
        explainer.generate_overlays(samples, alpha=0.5, image_size=224)
        assert len(explainer.overlays) == 2
        for cls_idx, overlaid in explainer.overlays.items():
            assert overlaid.shape == (3, 224, 224)
            # Verify the overlaid heatmap is not flat/uniform (which would happen if it's 1x1 interpolated)
            # We can check that the min and max of the visualised output are not the same, or that standard deviation > 0.01.
            assert not torch.allclose(overlaid, overlaid[0, 0, 0]), f"Overlaid image for {backbone} is completely uniform!"
    except Exception as e:
        pytest.fail(f"GradCAM failed for {backbone}: {e}")

    # Test SHAP
    try:
        shap_explainer = SHAP(model)
        shap_explainer.generate_overlays(samples, n_samples=3, stdevs=0.02)
        assert len(shap_explainer.overlays) == 2
        for cls_idx, overlaid in shap_explainer.overlays.items():
            assert overlaid.shape == (3, 224, 224)
            assert not torch.allclose(overlaid, overlaid[0, 0, 0]), f"SHAP overlaid image for {backbone} is completely uniform!"
    except Exception as e:
        pytest.fail(f"SHAP failed for {backbone}: {e}")

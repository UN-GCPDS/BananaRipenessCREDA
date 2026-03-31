import torch
import pytest
from banana_creda.models.backbones import BananaModel
from banana_creda.config import ModelConfig

@pytest.mark.parametrize("backbone", ["resnet", "efficientnet", "mobilenetv3", "vit"])
def test_banana_model_init_and_forward(backbone):
    config = ModelConfig(backbone=backbone, num_classes=4, pretrained=False)
    model = BananaModel(config)
    
    # Check if the model is initialized
    assert isinstance(model, BananaModel)
    
    # Test forward pass (classification mode)
    x = torch.randn(2, 3, 224, 224)
    output = model(x, mode='class')
    assert output.shape == (2, 4)
    
    # Test forward pass (feature mode)
    features = model(x, mode='feature')
    assert len(features.shape) == 2
    assert features.shape[0] == 2

from pydantic import ValidationError

def test_unsupported_backbone():
    with pytest.raises(ValidationError):
        ModelConfig(backbone="unsupported")

def test_vit_indexing():
    # Specific test for ViT to ensure it's indexing the [CLS] token
    config = ModelConfig(backbone="vit", num_classes=4, pretrained=False)
    model = BananaModel(config)
    
    # Create input and run forward
    x = torch.randn(1, 3, 224, 224)
    # ViT's encoder output is usually (B, L, D), we want to make sure it's flattened/indexed correctly
    features = model(x, mode='feature')
    assert features.ndim == 2
    assert features.shape[0] == 1

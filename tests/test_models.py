import torch
from banana_creda.models.backbones import BananaModel

def test_model_forward_modes(mock_config):
    model = BananaModel(mock_config.model)
    img = torch.randn(1, 3, 224, 224)
    
    # Test feature mode
    feats = model(img, mode='feature')
    assert feats.shape == (1, 512), f"Expected (1, 512), got {feats.shape}"
    
    # Test class mode
    logits = model(img, mode='class')
    assert logits.shape == (1, 4), f"Expected (1, 4), got {logits.shape}"
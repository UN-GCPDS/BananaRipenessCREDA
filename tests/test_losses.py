import torch
from banana_creda.losses.creda import CREDALoss

def test_creda_loss_output_shape(mock_config, sample_batch):
    features, logits, labels = sample_batch
    criterion = CREDALoss(mock_config.training, num_classes=4)
    
    # Simulate source and target with the same synthetic data
    loss, loss_dict = criterion(features, logits, labels, features, logits)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0
    assert "loss_creda" in loss_dict

def test_creda_numerical_stability(mock_config):
    """Verifies that the loss is stable with extreme values."""
    criterion = CREDALoss(mock_config.training, num_classes=4)
    
    # Case: Identical features (zero distance)
    feat = torch.ones(4, 512)
    logits = torch.randn(4, 4)
    labels = torch.tensor([0, 0, 1, 1])
    
    loss, _ = criterion(feat, logits, labels, feat, logits)
    assert not torch.isnan(loss), "The loss produced NaN with constant data"
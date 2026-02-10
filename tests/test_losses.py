import torch
from banana_creda.losses.creda import CREDALoss

def test_creda_loss_output_shape(mock_config, sample_batch):
    features, logits, labels = sample_batch
    criterion = CREDALoss(mock_config.training, num_classes=4)
    
    # Simular source y target con los mismos datos sintéticos
    loss, loss_dict = criterion(features, logits, labels, features, logits)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0
    assert "loss_creda" in loss_dict

def test_creda_numerical_stability(mock_config):
    """Verifica que la pérdida sea estable con valores extremos."""
    criterion = CREDALoss(mock_config.training, num_classes=4)
    
    # Caso: Características idénticas (distancia cero)
    feat = torch.ones(4, 512)
    logits = torch.randn(4, 4)
    labels = torch.tensor([0, 0, 1, 1])
    
    loss, _ = criterion(feat, logits, labels, feat, logits)
    assert not torch.isnan(loss), "La pérdida produjo NaN con datos constantes"
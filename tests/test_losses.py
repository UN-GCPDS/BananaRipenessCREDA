import torch
import pytest
from banana_creda.losses.creda import CREDALoss
from banana_creda.config import TrainConfig

def test_creda_loss_init():
    config = TrainConfig(lambda_creda=0.1, use_uncertainty=True)
    loss_fn = CREDALoss(config, num_classes=4)
    assert loss_fn.lambda_creda == 0.1
    assert loss_fn.num_classes == 4

def test_creda_loss_forward():
    config = TrainConfig(lambda_creda=0.1, use_uncertainty=True, sigma='auto')
    num_classes = 4
    loss_fn = CREDALoss(config, num_classes)
    
    # Dummy source data
    # (B=8, D=128)
    features_s = torch.randn(8, 128)
    logits_s = torch.randn(8, 4)
    labels_s = torch.randint(0, 4, (8,))
    
    # Dummy target data
    # (B=8, D=128)
    features_t = torch.randn(8, 128)
    logits_t = torch.randn(8, 4)
    
    # Ensure some valid classes for CREDA
    labels_s[0:2] = 0
    labels_s[2:4] = 1
    # We can't guarantee target pseudo labels, but let's see. 
    # Usually, we'd mock argmax if needed, but let's test it as-is first.
    
    total_loss, metrics = loss_fn(features_s, logits_s, labels_s, features_t, logits_t)
    
    assert torch.is_tensor(total_loss)
    assert "total_loss" in metrics
    assert "loss_cls" in metrics
    assert "loss_creda" in metrics

def test_creda_loss_no_uncertainty():
    config = TrainConfig(lambda_creda=0.5, use_uncertainty=False, sigma=1.0)
    num_classes = 2
    loss_fn = CREDALoss(config, num_classes)
    
    features_s = torch.randn(4, 64)
    logits_s = torch.randn(4, 2)
    labels_s = torch.tensor([0, 0, 1, 1])
    
    features_t = torch.randn(4, 64)
    logits_t = torch.randn(4, 2)
    
    total_loss, metrics = loss_fn(features_s, logits_s, labels_s, features_t, logits_t)
    assert isinstance(metrics["loss_creda"], float)

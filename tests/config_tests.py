import pytest
import torch
from banana_creda.config import ExperimentConfig

@pytest.fixture
def mock_config():
    """Retorna una configuración mínima para pruebas."""
    return ExperimentConfig.from_yaml("configs/base_experiment.yaml")

@pytest.fixture
def sample_batch():
    """Genera datos sintéticos: (features, logits, labels)."""
    batch_size = 8
    num_classes = 4
    feat_dim = 512
    
    features = torch.randn(batch_size, feat_dim)
    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    return features, logits, labels
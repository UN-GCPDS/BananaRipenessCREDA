import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor

from typing import List, Tuple
from banana_creda.config import ModelConfig

class FeatureWrapper(nn.Module):
    """Wrapper class to extract features from a specific node in a PyTorch model.
    
    Attributes:
        extractor (nn.Module): The underlying feature extraction module.
    """
    def __init__(self, model: nn.Module, node_name: str):
        """Initializes the FeatureWrapper.
        
        Args:
            model (nn.Module): Pretrained model to wrap.
            node_name (str): The name of the node from which to extract features.
        """
        super().__init__()
        self.extractor = create_feature_extractor(model, return_nodes={node_name: 'out'})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to extract features.
        
        Args:
            x (torch.Tensor): Input image tensor.
            
        Returns:
            torch.Tensor: Extracted features from the specified node.
        """
        return self.extractor(x)['out']

def get_backbone_components(config: ModelConfig) -> Tuple[List[nn.Module], nn.Module, bool]:
    """Retrieves the encoder layers and classifier for a specific backbone.
    
    This function initializes only the necessary layers to avoid redundancy.
    
    Args:
        config (ModelConfig): Configuration object specifying backbone and hyperparameters.
        
    Returns:
        Tuple[List[nn.Module], nn.Module, bool]: 
            List of encoder layers, the classification head, and a flag for ViT indexing.
            
    Raises:
        ValueError: If the backbone name in config is not supported.
    """
    bb_name = config.backbone.lower()
    weights = "IMAGENET1K_V1" if config.pretrained else None
    
    if bb_name == "resnet":
        model = models.resnet34(weights=weights)
        modules = list(model.children())
        encoder_layers = [
            nn.Sequential(*modules[:4]),
            modules[4],
            modules[5],
            modules[6],
            nn.Sequential(*modules[7:9])
        ]
        classifier = nn.Sequential(
            nn.Dropout(p=config.dropout_rate, inplace=True),
            nn.Linear(model.fc.in_features, config.num_classes)
        )
        return encoder_layers, classifier, False

    elif bb_name == "vit":
        model = models.vit_b_16(weights=weights)
        blocks = model.encoder.layers
        encoder_layers = [
            FeatureWrapper(model, 'encoder.dropout'),
            nn.Sequential(*blocks[0:3]),
            nn.Sequential(*blocks[3:6]),
            nn.Sequential(*blocks[6:9]),
            nn.Sequential(*blocks[9:12], model.encoder.ln)
        ]
        classifier = nn.Sequential(
            nn.Dropout(p=config.dropout_rate, inplace=True),
            nn.Linear(model.heads[0].in_features, config.num_classes)
        )
        return encoder_layers, classifier, True

    elif bb_name == "efficientnet":
        model = models.efficientnet_b0(weights=weights)
        modules = list(model.children())
        encoder_layers = [
            modules[0][0],
            nn.Sequential(*modules[0][1:3]),
            nn.Sequential(*modules[0][3:5]),
            nn.Sequential(*modules[0][5:7]),
            nn.Sequential(*modules[0][7:9], modules[1])
        ]
        classifier = modules[-1]
        classifier[0] = nn.Dropout(p=config.dropout_rate, inplace=True)
        classifier[1] = nn.Linear(classifier[1].in_features, config.num_classes)
        return encoder_layers, classifier, False

    elif bb_name == "mobilenetv3":
        model = models.mobilenet_v3_large(weights=weights)
        modules = list(model.children())
        encoder_layers = [
            modules[0][0],
            nn.Sequential(*modules[0][1:5]),
            nn.Sequential(*modules[0][5:9]),
            nn.Sequential(*modules[0][9:13]),
            nn.Sequential(*modules[0][13:17], modules[1])
        ]
        classifier = modules[-1]
        classifier[2] = nn.Dropout(p=config.dropout_rate, inplace=True)
        classifier[3] = nn.Linear(classifier[3].in_features, config.num_classes)
        return encoder_layers, classifier, False

    else:
        raise ValueError(f"Backbone '{bb_name}' not supported for Banana model")

class BananaModel(nn.Module):
    """Modular neural network architecture for Domain Adaptation tasks.
    
    Supports multiple backbones and provides dual forward pass (features/logits).
    
    Attributes:
        config (ModelConfig): Configuration settings.
        encoder (nn.Module): Core feature extraction layers.
        classifier (nn.Module): Final classification layers.
        needs_vit_indexing (bool): Flag indicating if ViT-style class token indexing is needed.
    """
    def __init__(self, config: ModelConfig):
        """Initializes the BananaModel.
        
        Args:
            config (ModelConfig): Neural network configuration.
        """
        super(BananaModel, self).__init__()
        self.config = config
        
        encoder_layers, self.classifier, self.needs_vit_indexing = get_backbone_components(config)
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x: torch.Tensor, mode: str = 'class') -> torch.Tensor:
        """Executes the forward pass of the model.
        
        Args:
            x (torch.Tensor): Input batch of images.
            mode (str): Execution mode, either 'class' for logits or 'feature' for deep features.
            
        Returns:
            torch.Tensor: Logits or feature vectors depending on the mode.
        """
        features = self.encoder(x)

        if self.needs_vit_indexing:
            features = features[:, 0]

        if len(features.shape) > 2:
            features = torch.flatten(features, 1)

        if mode == 'feature':
            return features
        
        return self.classifier(features)

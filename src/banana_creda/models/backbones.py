import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor

from banana_creda.config import ModelConfig

class FeatureWrapper(nn.Module):
    """
    Wrapper class to extract features from a specific node in a PyTorch model.
    """
    def __init__(self, model, node_name):
        super().__init__()
        self.extractor = create_feature_extractor(model, return_nodes={node_name: 'out'})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extractor(x)['out']

def get_backbone_components(config: ModelConfig):
    """
    Returns the encoder and classifier for a given backbone configuration.
    This function avoids redundant model loading.
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
    """
    Modular neural network architecture designed for Domain Adaptation tasks.
    """
    def __init__(self, config: ModelConfig):
        super(BananaModel, self).__init__()
        self.config = config
        
        self.encoder, self.classifier, self.needs_vit_indexing = get_backbone_components(config)

        self.encoder = nn.Sequential(*self.encoder)

    def forward(self, x: torch.Tensor, mode: str = 'class'):
        """
        Performs the forward pass through the network.
        """
        features = self.encoder(x)

        if self.needs_vit_indexing:
            features = features[:, 0]

        if len(features.shape) > 2:
            features = torch.flatten(features, 1)

        if mode == 'feature':
            return features
        
        return self.classifier(features)

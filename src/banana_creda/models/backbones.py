import torch
import torch.nn as nn
import torchvision.models as models
from banana_creda.config import ModelConfig

class BananaModel(nn.Module):
    """
    Modular neural network architecture designed for Domain Adaptation tasks.

    This class provides a unified interface for various backbone architectures 
    (ResNet34, ViT-B/16, EfficientNet-B0, MobileNetV3-Large) and handles the 
    extraction of features versus classification outputs.

    Attributes:
        config (ModelConfig): Configuration object containing model parameters.
        backbone (nn.Module): The feature extraction portion of the network.
        classifier (nn.Module): The classification head of the network.
    """
    def __init__(self, config: ModelConfig):
        """
        Initializes the BananaModel with a specific configuration.

        Args:
            config (ModelConfig): Model configuration including backbone type,
                pretraining status, and number of classes.
        """
        super(BananaModel, self).__init__()
        self.config = config
        
        # 1. Get the backbone and its output dimension
        self.backbone, self.classifier = self._get_backbone_classifier()

    def _get_backbone_classifier(self) -> tuple[nn.Module, nn.Module]:
        """
        Factory method to initialize the backbone and classifier components.

        Depending on the configuration, it loads a pre-defined architecture, 
        optionally fetches pretrained weights, and splits the model into 
        feature extraction and classification modules. It also adjusts the 
        final linear layer to match the number of classes.

        Returns:
            tuple[nn.Module, nn.Module]: A tuple containing (backbone, classifier).
        
        Raises:
            ValueError: If the specified backbone name is not supported.
        """
        bb_name = self.config.backbone.lower()
        weights = "IMAGENET1K_V1" if self.config.pretrained else None
        
        if bb_name == "resnet":
            model = models.resnet34(weights=weights)
            modules = list(model.children())
            backbone = nn.Sequential(*modules[:-1])
            classifier = nn.Sequential(
                nn.Dropout(p=self.config.dropout_rate),
                nn.Linear(model.fc.in_features, self.config.num_classes)
            )
            
        elif bb_name == "vit":
            model = models.vit_b_16(weights=weights)
            modules = list(model.children())
            backbone = nn.Sequential(*modules[:-1])
            classifier = nn.Sequential(
                nn.Sequential(
                    nn.Dropout(p=self.config.dropout_rate),
                    nn.Linear(model.heads[0].in_features, self.config.num_classes)
                )
            )
            
        elif bb_name == "efficientnet":
            model = models.efficientnet_b0(weights=weights)
            modules = list(model.children())
            backbone = nn.Sequential(*modules[:-1])
            classifier = nn.Sequential(*modules[-1:])
            classifier[0][0] = nn.Dropout(p=self.config.dropout_rate, inplace=True)
            classifier[0][1] = nn.Linear(classifier[0][1].in_features, self.config.num_classes)
            
        elif bb_name == "mobilenetv3":
            model = models.mobilenet_v3_large(weights=weights)
            modules = list(model.children())
            backbone = nn.Sequential(*modules[:-1])
            classifier = nn.Sequential(*modules[-1:])
            classifier[0][2] = nn.Dropout(p=self.config.dropout_rate, inplace=True)
            classifier[0][3] = nn.Linear(classifier[0][3].in_features, self.config.num_classes)
            
        else:
            raise ValueError(f"Backbone '{bb_name}' not supported for Banana model")
            
        return backbone, classifier

    def forward(self, x: torch.Tensor, mode: str = 'class'):
        """
        Performs the forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor (usually image batch).
            mode (str): Execution mode. 
                - 'class': Returns the final classification scores (logits).
                - 'feature': Returns the latent embedding (feature vector).
                Defaults to 'class'.

        Returns:
            torch.Tensor: Either the features of shape (B, D) or the logits of 
                shape (B, num_classes).
        """
        # Extract features
        features = self.backbone(x)
        
        # Some architectures (ViT) already return the flattened vector,
        # others (CNNs) return [B, C, 1, 1]. Flatten for safety.
        if len(features.shape) > 2:
            features = torch.flatten(features, 1)

        if mode == 'feature':
            return features
        
        return self.classifier(features)
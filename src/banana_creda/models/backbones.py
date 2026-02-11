import torch
import torch.nn as nn
import torchvision.models as models
from banana_creda.config import ModelConfig

class BananaModel(nn.Module):
    """
    Modular architecture for Domain Adaptation.
    Supports ResNet, ViT, EfficientNet and MobileNet.
    """
    def __init__(self, config: ModelConfig):
        super(BananaModel, self).__init__()
        self.config = config
        
        # 1. Get the backbone and its output dimension
        self.backbone, self.num_features = self._get_backbone_and_dims()
        
        # 2. Common classifier (Head)
        self.classifier = nn.Linear(self.num_features, self.config.num_classes)

    def _get_backbone_and_dims(self) -> tuple[nn.Module, int]:
        """
        Backbone factory: Initializes the architecture and extracts the 
        dimension of the latent embedding.
        """
        bb_name = self.config.backbone.lower()
        weights = "IMAGENET1K_V1" if self.config.pretrained else None
        
        if bb_name == "resnet":
            model = models.resnet34(weights=weights)
            num_in = model.fc.in_features
            model.fc = nn.Identity() # Remove the original head
            
        elif bb_name == "vit":
            model = models.vit_b_16(weights=weights)
            # ViT-B-16 has a head in model.heads.head   
            num_in = model.heads.head.in_features
            model.heads = nn.Identity()
            
        elif bb_name == "efficientnet":
            model = models.efficientnet_b0(weights=weights)
            # EfficientNet has the head in model.classifier[1]
            num_in = model.classifier[1].in_features
            model.classifier = nn.Identity()
            
        elif bb_name == "mobilenetv3":
            model = models.mobilenet_v3_large(weights=weights)
            # MobileNetV3 has a complex head in .classifier
            num_in = model.classifier[3].in_features
            model.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Backbone '{bb_name}' not supported for Banana model")
            
        return model, num_in

    def forward(self, x: torch.Tensor, mode: str = 'class'):
        """
        Forward pass consistent for all architectures.
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
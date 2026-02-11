import torch
import torch.nn as nn
import torchvision.models as models
from banana_creda.config import ModelConfig

class BananaModel(nn.Module):
    """
    Arquitectura modular para Domain Adaptation.
    Soporta ResNet, ViT, EfficientNet y MobileNet.
    """
    def __init__(self, config: ModelConfig):
        super(BananaModel, self).__init__()
        self.config = config
        
        # 1. Obtener el backbone y su dimensión de salida
        self.backbone, self.num_features = self._get_backbone_and_dims()
        
        # 2. Clasificador común (Head)
        self.classifier = nn.Linear(self.num_features, self.config.num_classes)

    def _get_backbone_and_dims(self) -> tuple[nn.Module, int]:
        """
        Fábrica de backbones: Inicializa la arquitectura y extrae la 
        dimensión del embedding latente.
        """
        bb_name = self.config.backbone.lower()
        weights = "IMAGENET1K_V1" if self.config.pretrained else None
        
        if bb_name == "resnet":
            model = models.resnet34(weights=weights)
            num_in = model.fc.in_features
            model.fc = nn.Identity() # Removemos el head original
            
        elif bb_name == "vit":
            model = models.vit_b_16(weights=weights)
            # ViT-B-16 tiene un head en model.heads.head
            num_in = model.heads.head.in_features
            model.heads = nn.Identity()
            
        elif bb_name == "efficientnet":
            model = models.efficientnet_b0(weights=weights)
            # EfficientNet tiene el head en model.classifier[1]
            num_in = model.classifier[1].in_features
            model.classifier = nn.Identity()
            
        elif bb_name == "mobilenetv3":
            model = models.mobilenet_v3_large(weights=weights)
            # MobileNetV3 tiene un head complejo en .classifier
            num_in = model.classifier[3].in_features
            model.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Backbone '{bb_name}' no soportado para bananos.")
            
        return model, num_in

    def forward(self, x: torch.Tensor, mode: str = 'class'):
        """
        Forward pass consistente para todas las arquitecturas.
        """
        # Extraer características
        features = self.backbone(x)
        
        # Algunas arquitecturas (ViT) ya devuelven el vector aplanado,
        # otras (CNNs) devuelven [B, C, 1, 1]. Aplanamos por seguridad.
        if len(features.shape) > 2:
            features = torch.flatten(features, 1)

        if mode == 'feature':
            return features
        
        return self.classifier(features)
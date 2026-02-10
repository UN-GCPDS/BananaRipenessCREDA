import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights
from banana_creda.config import ModelConfig

class BananaModel(nn.Module):
    """
    Wrapper de ResNet optimizado para Domain Adaptation.
    Separa el extractor de características del clasificador.
    """
    def __init__(self, config: ModelConfig):
        super(BananaModel, self).__init__()
        self.config = config

        # 1. Inicializar Backbone (Extractor de características)
        if self.config.backbone == "resnet34":
            weights = ResNet34_Weights.IMAGENET1K_V1 if self.config.pretrained else None
            base_model = resnet34(weights=weights)
            num_features = base_model.fc.in_features
            # Removemos la última capa fc
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        else:
            raise NotImplementedError(f"Backbone {self.config.backbone} no implementado.")

        # 2. Clasificador (Head)
        self.classifier = nn.Linear(num_features, self.config.num_classes)

    def forward(self, x: torch.Tensor, mode: str = 'class'):
        """
        Args:
            x: Tensor de entrada [B, 3, H, W]
            mode: 
                'feature' -> Retorna el embedding (vector de características) [B, 512]
                'class' -> Retorna los logits de clasificación [B, num_classes]
        """
        # Extraer características: [B, 512, 1, 1]
        features = self.backbone(x)
        
        # Aplanar para el clasificador: [B, 512]
        features = torch.flatten(features, 1)

        if mode == 'feature':
            return features
        
        # Obtener logits
        logits = self.classifier(features)
        return logits
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor

class FeatureWrapper(nn.Module):
    """
    Wrapper class to extract features from a specific node in a PyTorch model.

    This class is particularly useful for Vision Transformers (ViT) where the 
    standard output is a sequence of patch embeddings plus a class token. 
    By specifying a node name, we can extract features from intermediate layers 
    or specific parts of the network.

    Attributes:
        extractor (nn.Module): The feature extractor created using 
            torchvision.models.feature_extraction.create_feature_extractor.
    """
    def __init__(self, model, node_name):
        """
        Initializes the FeatureWrapper.

        Args:
            model (nn.Module): The base model from which to extract features.
            node_name (str): The name of the node to extract features from. 
                This name corresponds to the key in the dictionary returned by 
                the extractor.
        """
        super().__init__()
        self.extractor = create_feature_extractor(model, return_nodes={node_name: 'out'})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ 
        Performs the forward pass through the feature extractor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Extracted features.
        """
        return self.extractor(x)['out']

class BananaModel(nn.Module):
    """
    Modular neural network architecture designed for Domain Adaptation tasks.

    This class provides a unified interface for various backbone architectures 
    (ResNet34, ViT-B/16, EfficientNet-B0, MobileNetV3-Large) and handles the 
    extraction of features versus classification outputs.

    Attributes:
        config (ModelConfig): Configuration object containing model parameters.
        layer_0 (nn.Module): The first layer of the backbone.
        layer_1 (nn.Module): The second layer of the backbone.
        layer_2 (nn.Module): The third layer of the backbone.
        layer_3 (nn.Module): The fourth layer of the backbone.
        layer_4 (nn.Module): The fifth layer of the backbone.
        classifier (nn.Module): The classification head of the network.
    """
    def __init__(self, config):
        """
        Initializes the BananaModel with a specific configuration.

        Args:
            config (ModelConfig): Model configuration including backbone type,
                pretraining status, and number of classes.
        """
        super(BananaModel, self).__init__()
        self.config = config
        
        # 1. Get the layers and classifier  
        self.layer_0, self.layer_1, self.layer_2, self.layer_3, self.layer_4, self.classifier = self._get_layers_classifier()

    def _get_layers_classifier(self) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module, nn.Module, nn.Module]:
        """
        Factory method to initialize the layers and classifier components.

        Depending on the configuration, it loads a pre-defined architecture, 
        optionally fetches pretrained weights, and splits the model into 
        feature extraction and classification modules. It also adjusts the 
        final linear layer to match the number of classes.

        Returns:
            tuple[nn.Module, nn.Module, nn.Module, nn.Module, nn.Module, nn.Module]: 
            A tuple containing (layer_0, layer_1, layer_2, layer_3, layer_4, classifier).
        
        Raises:
            ValueError: If the specified backbone name is not supported.
        """
        bb_name = self.config.backbone.lower()
        weights = "IMAGENET1K_V1" if self.config.pretrained else None
        
        if bb_name == "resnet":
            model = models.resnet34(weights=weights)
            modules = list(model.children())
            layer_0 = nn.Sequential(*modules[:4])
            layer_1 = modules[4]
            layer_2 = modules[5]
            layer_3 = modules[6]
            layer_4 = nn.Sequential(*modules[7:9])
            classifier = nn.Sequential(nn.Dropout(p=self.config.dropout_rate, inplace=True),
                                       nn.Linear(model.fc.in_features, self.config.num_classes))
            
        elif bb_name == "vit":
            model = models.vit_b_16(weights=weights)
            blocks = model.encoder.layers
            layer_0 = FeatureWrapper(model, 'encoder.dropout')
            layer_1 = nn.Sequential(*blocks[0:3])
            layer_2 = nn.Sequential(*blocks[3:6])
            layer_3 = nn.Sequential(*blocks[6:9])
            layer_4 =  nn.Sequential(*blocks[9:12],model.encoder.ln)
            classifier = nn.Sequential(
                nn.Dropout(p=self.config.dropout_rate, inplace=True),
                nn.Linear(model.heads[0].in_features, self.config.num_classes)
            )
            
        elif bb_name == "efficientnet":
            model = models.efficientnet_b0(weights=weights)
            modules = list(model.children())
            layer_0 = modules[0][0]
            layer_1 = nn.Sequential(*modules[0][1:3])
            layer_2 = nn.Sequential(*modules[0][3:5])
            layer_3 = nn.Sequential(*modules[0][5:7])
            layer_4 = nn.Sequential(*modules[0][7:9],modules[1])
            classifier = modules[-1]
            classifier[0] = nn.Dropout(p=self.config.dropout_rate, inplace=True)
            classifier[1] = nn.Linear(classifier[1].in_features, self.config.num_classes)
            
        elif bb_name == "mobilenetv3":
            model = models.mobilenet_v3_large(weights=weights)
            modules = list(model.children())
            layer_0 = modules[0][0]
            layer_1 = nn.Sequential(*modules[0][1:5])
            layer_2 = nn.Sequential(*modules[0][5:9])
            layer_3 = nn.Sequential(*modules[0][9:13])
            layer_4 = nn.Sequential(*modules[0][13:17],modules[1])
            classifier = modules[-1]
            classifier[2] = nn.Dropout(p=self.config.dropout_rate, inplace=True)
            classifier[3] = nn.Linear(classifier[3].in_features, self.config.num_classes)
            
        else:
            raise ValueError(f"Backbone '{bb_name}' not supported for Banana model")
            
        return layer_0, layer_1, layer_2, layer_3, layer_4, classifier

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
        features = self.layer_0(x)
        features = self.layer_1(features)
        features = self.layer_2(features)
        features = self.layer_3(features)   
        features = self.layer_4(features)

        if self.config.backbone.lower() == 'vit':
            features = features[:,0]

        if len(features.shape) > 2:
            features = torch.flatten(features, 1)

        if mode == 'feature':
            return features
        
        return self.classifier(features)
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Literal
import yaml
from pathlib import Path

class DataConfig(BaseModel):
    """Configuration for data loading and transformation."""
    orig_data_dir: str
    synth_data_dir: str
    batch_size: int = Field(default=32, gt=0)
    img_size: int = Field(default=224, gt=0)
    num_workers: int = Field(default=4)
    imagenet_mean: List[float] = [0.485, 0.456, 0.406]
    imagenet_std: List[float] = [0.229, 0.224, 0.225]

class ModelConfig(BaseModel):
    """Configuration for model architecture."""  
    num_classes: int = 4
    pretrained: bool = True
    backbone: Literal["resnet", "vit", "efficientnet", "mobilenetv3"] = "resnet"

class TrainConfig(BaseModel):
    """Configuration for training and Domain Adaptation."""
    epochs: int = Field(default=10, gt=0)
    lr: float = Field(default=1e-4, gt=0)
    gamma: float = 0.94
    # Warmup logic to stabilize initial training
    warmup: bool = True
    warmup_epochs: int = 5
    warmup_threshold: float = 0.9
    # CREDA hyperparameters
    lambda_creda: float = 1.0
    lambda_entropy: float = 0.0  # Desactivado según tu base_experiment.yaml
    use_uncertainty: bool = True
    sigma: Union[float, str] = "auto"
    # Infrastructure
    use_amp: bool = True
    device: str = "cuda"
    seed: int = 42 # Seed for scientific reproducibility

class ExperimentMetadata(BaseModel):
    """Metadata for version control and project outputs."""
    name: str = "base_experiment"
    version: int = 1
    output_dir: str = "outputs/experiment_1"

class ExperimentConfig(BaseModel):
    """Global schema that unites all configurations for validation with Pydantic."""
    data: DataConfig
    model: ModelConfig
    training: TrainConfig
    experiment: ExperimentMetadata

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "ExperimentConfig":
        """Loads and validates the configuration from a YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
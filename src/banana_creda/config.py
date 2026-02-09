from pydantic import BaseModel, Field
from typing import List, Optional, Union
import yaml
from pathlib import Path

class DataConfig(BaseModel):
    orig_data_dir: str
    synth_data_dir: str
    batch_size: int = Field(default=32, gt=0)
    img_size: int = Field(default=224, gt=0)
    num_workers: int = Field(default=2)
    imagenet_mean: List[float] = [0.485, 0.456, 0.406]
    imagenet_std: List[float] = [0.229, 0.224, 0.225]

class ModelConfig(BaseModel):
    num_classes: int = 4
    pretrained: bool = True
    backbone: str = "resnet34"

class TrainConfig(BaseModel):
    epochs: int = Field(default=15, gt=0)
    lr: float = Field(default=1e-4, gt=0)
    gamma: float = 0.94
    lambda_creda: float = 1.0
    lambda_entropy: float = 0.05
    use_uncertainty: bool = True
    sigma: Union[float, str] = "auto"
    use_amp: bool = True
    device: str = "cuda"

class ExperimentConfig(BaseModel):
    """Esquema global que une todas las configuraciones."""
    data: DataConfig
    model: ModelConfig
    training: TrainConfig

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "ExperimentConfig":
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
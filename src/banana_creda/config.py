from pydantic import BaseModel, Field
from typing import Literal
import yaml
from pathlib import Path

class DataConfig(BaseModel):
    """Configuration for data loading and transformation."""
    source_data_dir: Path
    target_data_dir: Path | None = None
    synthetic_data_dir: Path | None = None
    batch_size: int = Field(default=32, gt=0)
    img_size: int = Field(default=224, gt=0)
    num_workers: int = Field(default=4, ge=0)
    imagenet_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    use_lime_on_target: bool | None = None

class ModelConfig(BaseModel):
    """Configuration for model architecture."""  
    num_classes: int = Field(default=4, gt=0)
    pretrained: bool = True
    backbone: Literal["resnet", "vit", "efficientnet", "mobilenetv3"] = "resnet"
    dropout_rate: float = Field(default=0.2, ge=0.0, le=1.0)

class TrainConfig(BaseModel):
    """Configuration for training and Domain Adaptation."""
    epochs: int = Field(default=10, gt=0)
    transfer_epochs: int | None = None
    
    lr: float = Field(default=1e-4, gt=0)
    transfer_lr: float | None = Field(default=1e-5, gt=0)
    epochs_phases: list[int] | None = None
    gamma: float = Field(default=0.94, gt=0.0, le=1.0)
    
    # Warm-up logic
    warmup: bool | None = None
    warmup_epochs: int | None = None
    warmup_threshold: float | None = None
    
    # CREDA static hyperparameters
    lambda_creda: float | None = None
    
    use_uncertainty: bool | None = None
    sigma: float | str | None = None
    use_amp: bool = True
    device: Literal["cuda", "cpu", "mps"] = "cuda"
    seed: int = 42

class ExperimentMetadata(BaseModel):
    """Metadata for version control and project outputs."""
    name: str = "base_experiment"
    version: int = Field(default=1, gt=0)
    output_dir: Path = Path("outputs/experiment_1")

class ExperimentConfig(BaseModel):
    """Global schema that unites all configurations."""
    data: DataConfig
    model: ModelConfig
    training: TrainConfig
    experiment: ExperimentMetadata

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "ExperimentConfig":
        """Loads and validates the configuration from a YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
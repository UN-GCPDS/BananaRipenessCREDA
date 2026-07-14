from pydantic import BaseModel, Field
from typing import Literal
import yaml
from pathlib import Path

class DataConfig(BaseModel):
    """Configuration for data loading and transformation.

    Attributes:
        source_data_dir (Path): Path to the source domain dataset.
        target_data_dir (Path | None): Path to the target domain dataset.
        synthetic_data_dir (Path | None): Path to the synthetic dataset.
        batch_size (int): Number of samples per batch.
        img_size (int): Size to which images will be resized.
        num_workers (int): Number of subprocesses used for data loading.
        imagenet_mean (tuple[float, float, float]): ImageNet mean for normalization.
        imagenet_std (tuple[float, float, float]): ImageNet std for normalization.
        use_lime_on_target (bool | None): Flag to use LIME variation on target data.
        use_augmentation (bool): Whether to apply data augmentations to training dataset.
    """
    source_data_dir: Path
    target_data_dir: Path | None = None
    synthetic_data_dir: Path | None = None
    batch_size: int = Field(default=32, gt=0)
    img_size: int = Field(default=224, gt=0)
    num_workers: int = Field(default=4, ge=0)
    imagenet_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    use_lime_on_target: bool | None = None
    use_augmentation: bool = True

class ModelConfig(BaseModel):
    """Configuration for model architecture.

    Attributes:
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to use a pretrained backbone.
        backbone (Literal): Name of the architecture to use as backbone.
        dropout_rate (float): Dropout probability for the classification head.
    """  
    num_classes: int = Field(default=4, gt=0)
    pretrained: bool = True
    backbone: Literal["resnet", "vit", "efficientnet", "mobilenetv3"] = "resnet"
    dropout_rate: float = Field(default=0.2, ge=0.0, le=1.0)

class TrainConfig(BaseModel):
    """Configuration for training and Domain Adaptation.

    Attributes:
        epochs (int): Total number of training epochs.
        transfer_epochs (int | None): Epochs allocated for transfer learning.
        lr (float): Learning rate for the optimizer.
        transfer_lr (float | None): Learning rate for transfer learning phase.
        epochs_phases (list[int] | None): Epoch thresholds for training phases.
        gamma (float): Learning rate decay factor.
        warmup (bool | None): Enable learning rate warm-up.
        warmup_epochs (int | None): Number of warm-up epochs.
        warmup_threshold (float | None): Accuracy threshold for warm-up completion.
        lambda_creda (float | None): Weight for the CREDA loss term.
        use_uncertainty (bool | None): Flag to weight samples by uncertainty.
        sigma (float | str | None): Bandwidth for the RBF kernel.
        use_amp (bool): Whether to use Automatic Mixed Precision.
        device (Literal): Computing device (cpu, cuda, mps).
        seed (int): Random seed for reproducibility.
    """
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
    """Metadata for version control and project outputs.

    Attributes:
        name (str): Unique name for the experiment.
        version (int): Iteration number of the experiment.
        output_dir (Path): Directory to save logs, models, and visualizations.
        save_results (bool): Whether to save the results of the experiment.
    """
    name: str = "base_experiment"
    version: int = Field(default=1, ge=0)
    output_dir: Path = Path("outputs/experiment_1")
    save_results: bool = False

class ExperimentConfig(BaseModel):
    """Global schema that unites all configurations.

    Attributes:
        data (DataConfig): Data-related settings.
        model (ModelConfig): Architecture-related settings.
        training (TrainConfig): Training-related settings.
        experiment (ExperimentMetadata): Metadata for tracking.
    """
    data: DataConfig
    model: ModelConfig
    training: TrainConfig
    experiment: ExperimentMetadata

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "ExperimentConfig":
        """Loads and validates the configuration from a YAML file.
        
        Args:
            yaml_path (str | Path): Path to the YAML configuration file.
            
        Returns:
            ExperimentConfig: Validated local configuration object.
        """
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        # Clean output_dir to prevent absolute paths starting with /outputs
        if "experiment" in config_dict and "output_dir" in config_dict["experiment"]:
            out_dir = str(config_dict["experiment"]["output_dir"]).replace("\\", "/")
            if out_dir.startswith("/outputs"):
                config_dict["experiment"]["output_dir"] = out_dir.lstrip("/")
                
        return cls(**config_dict)

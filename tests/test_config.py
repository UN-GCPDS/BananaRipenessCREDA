import pytest
from pathlib import Path
import yaml
from banana_creda.config import (
    DataConfig,
    ModelConfig,
    TrainConfig,
    ExperimentMetadata,
    ExperimentConfig
)

def test_data_config_valid():
    config = DataConfig(
        source_data_dir=Path("data/source"),
        target_data_dir=Path("data/target"),
        batch_size=16
    )
    assert config.batch_size == 16
    assert config.img_size == 224

def test_data_config_invalid():
    with pytest.raises(ValueError):
        DataConfig(source_data_dir=Path("data/source"), batch_size=-1)

def test_model_config_defaults():
    config = ModelConfig()
    assert config.num_classes == 4
    assert config.backbone == "resnet"

def test_experiment_config_from_yaml(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    yaml_file = d / "config.yaml"
    
    config_dict = {
        "data": {
            "source_data_dir": "data/source",
            "target_data_dir": "data/target",
            "batch_size": 32
        },
        "model": {
            "num_classes": 4,
            "backbone": "vit"
        },
        "training": {
            "epochs": 10,
            "lr": 0.001
        },
        "experiment": {
            "name": "test_exp",
            "version": 1,
            "output_dir": "outputs/test"
        }
    }
    
    with open(yaml_file, "w") as f:
        yaml.dump(config_dict, f)
        
    config = ExperimentConfig.from_yaml(yaml_file)
    assert config.model.backbone == "vit"
    assert config.experiment.name == "test_exp"
    assert isinstance(config.data.source_data_dir, Path)

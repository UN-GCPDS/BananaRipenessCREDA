"""

"""
import torch
import argparse
from pathlib import Path

# Imports from the modular package
from banana_creda.config import ExperimentConfig
from banana_creda.data.loader import BananaDataLoader
from banana_creda.models.backbones import BananaModel
from banana_creda.utils.reproducibility import set_seed

def run_explainability(config_path: str, model_path: str, output_dir: str = None) -> None:
    """
    """
    # 1. Load Configuration and Setup Environment
    cfg = ExperimentConfig.from_yaml(config_path)

    if output_dir is not None:
        cfg.experiment.output_dir = output_dir

    if cfg.training.seed is not None:
        set_seed(cfg.training.seed)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    
    # 2. Data Setup (Prioritizing Target Test data for explainability)
    data_manager = BananaDataLoader(cfg.data)
    
    # IMPORTANT: Source domain loader does not require lighting variation (LIME) labels
    _, _, src_test, _ = data_manager.get_split_loaders(cfg.data.source_data_dir)
    
    # IMPORTANT: Target domain loader requires lighting variation (LIME) labels if use_lime_on_target is True
    _, _, tgt_test, _ = data_manager.get_split_loaders(
        cfg.data.target_data_dir, 
        cfg.data.use_lime_on_target
    )
    
    # 3. Model Initialization and Loading Weights
    print(f"Loading pre-trained model from: {model_path}")
    model = BananaModel(cfg.model).to(device)
    
    # Load state dict with map_location for CPU/GPU flexibility
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 4. Final Explainability and Scientific Reports
    output_dir = Path(cfg.experiment.output_dir + "/explainability")
    output_dir.mkdir(parents=True, exist_ok=True)

    
    
    
    
    print(f"\nexplainability completed successfully. Results saved in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Banana-CREDA Explainability Script")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained .pth model file")
    parser.add_argument("--output_dir", type=str, default=None, help="Override the output directory defined in the YAML config")
    
    args = parser.parse_args()
    run_explainability(args.config, args.model, args.output_dir)

"""
Command-line script for running explainability analysis on a trained model.

This script loads a pre-trained model, generates representative samples from the 
source test set, and produces Grad-CAM visualizations to explain the model's 
decisions.
"""
import torch
import argparse
from pathlib import Path

# Imports from the modular package
from banana_creda.config import ExperimentConfig
from banana_creda.data.loader import BananaDataLoader
from banana_creda.models.backbones import BananaModel
from banana_creda.utils.reproducibility import set_seed
from banana_creda.explain.gradcam import GradCAM
from banana_creda.utils.representative_samples import representative_samples

def run_explainability(config_path: str, model_path: str, output_dir: str = None) -> None:
    """
    Runs the explainability analysis pipeline.
    """
    # 1. Load Configuration and Setup Environment
    cfg = ExperimentConfig.from_yaml(config_path)

    if output_dir is not None:
        cfg.experiment.output_dir = output_dir

    if cfg.training.seed is not None:
        set_seed(cfg.training.seed)
    
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    
    # 2. Data Setup (Both Source and Target)
    data_manager = BananaDataLoader(cfg.data)
    _, _, src_test, _ = data_manager.get_split_loaders(cfg.data.source_data_dir)
    _, _, tgt_test, _ = data_manager.get_split_loaders(
        cfg.data.target_data_dir, 
        cfg.data.use_lime_on_target
    )
    
    # 3. Model Initialization
    print(f"Loading model: {model_path}")
    model = BananaModel(cfg.model).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 4. Path Setup using pathlib
    explain_root = Path(cfg.experiment.output_dir) / "explainability"
    explain_root.mkdir(parents=True, exist_ok=True)

    # 5. Run Analysis Loop for Both Domains
    domains = [
        ("source", src_test),
        ("target", tgt_test)
    ]

    for domain_name, loader in domains:
        print(f"\n>>> Analyzing Domain: {domain_name.upper()}")
        domain_dir = explain_root / domain_name
        domain_dir.mkdir(parents=True, exist_ok=True)

        # Generate Samples
        samples, found_all = representative_samples(loader, cfg.model.num_classes)
        if not samples:
            print(f"Warning: No samples found for {domain_name}. Skipping.")
            continue
        
        if not found_all:
            print(f"Note: Only found {len(samples)}/{cfg.model.num_classes} classes for {domain_name}.")

        # Initialize Grad-CAM
        explainer = GradCAM(model, cfg.model.backbone)
        
        print(f"Generating and saving overlays for {domain_name}...")
        explainer.generate_overlays(samples, alpha=0.5, image_size=224)
        explainer.save_overlays(str(domain_dir))

    print(f"\nAll explainability reports saved in: {explain_root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Banana-CREDA Explainability Script")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained .pth model file")
    parser.add_argument("--output_dir", type=str, default=None, help="Override the output directory defined in the YAML config")
    
    args = parser.parse_args()
    run_explainability(args.config, args.model, args.output_dir)

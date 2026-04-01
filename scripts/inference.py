"""
Command-line script for running inference and evaluation on a trained model.

This script loads a pre-trained model, prepares data loaders for both source 
and target domains, performs classification on the target test set, and 
generates performance metrics and latent space visualizations.
"""
import torch
import argparse
from pathlib import Path

# Imports from the modular package
from banana_creda.config import ExperimentConfig
from banana_creda.data.loader import BananaDataLoader
from banana_creda.models.backbones import BananaModel
from banana_creda.utils.visualizer import BananaVisualizer
from banana_creda.utils.metrics import MetricTracker

def run_inference(config_path: str, model_path: str, output_dir: str = None) -> None:
    """Runs inference and generates evaluation reports for a trained model.

    Args:
        config_path (str): Path to the YAML configuration file.
        model_path (str): Path to the .pth file containing trained weights.
        output_dir (str, optional): Override the output directory defined in the YAML config.
    """
    # 1. Load Configuration and Setup Environment
    cfg = ExperimentConfig.from_yaml(config_path)

    if output_dir is not None:
        cfg.experiment.output_dir = output_dir

    if cfg.training.seed is not None:
        set_seed(cfg.training.seed)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    
    # 2. Data Setup (Prioritizing Target Test data for evaluation)
    data_manager = BananaDataLoader(cfg.data)
    
    # IMPORTANT: Source domain loader does not require lighting variation (LIME) labels
    src_train, src_val, src_test, class_names = data_manager.get_split_loaders(cfg.data.source_data_dir)
    
    # IMPORTANT: Target domain loader requires lighting variation (LIME) labels if use_lime_on_target is True
    tgt_train, tgt_val, tgt_test, _ = data_manager.get_split_loaders(
        cfg.data.target_data_dir, 
        cfg.data.use_lime_on_target
    )

    source_loaders = {'train': src_train, 'validation': src_val, 'test': src_test}
    target_loaders = {'train': tgt_train, 'validation': tgt_val, 'test': tgt_test}
    
    # 3. Model Initialization and Loading Weights
    print(f"Loading pre-trained model from: {model_path}")
    model = BananaModel(cfg.model).to(device)
    
    # Load state dict with map_location for CPU/GPU flexibility
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 4. Final Evaluation and Scientific Reports
    output_dir = cfg.experiment.output_dir + "/inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    viz = BananaVisualizer(device=device, output_dir=str(output_dir))
    
    print("\nExecuting evaluation on Target Test Set...")

    y_true_np, y_pred_np, y_probs_np, _, _ = viz._get_inference_data(model, target_loaders['test'], save_results=cfg.experiment.save_results)

    # Compute and print Full Scientific Report (Precision, Recall, F1, Support)
    metrics = MetricTracker.compute_full_metrics(
        torch.from_numpy(y_pred_np), 
        torch.from_numpy(y_true_np), 
        len(class_names), 
        device
    )

    MetricTracker.print_full_report("Inference Run: Target Test Set", metrics, class_names)
    
    print("\nGenerating Advanced Visualizations...")
    
    # 1. Confusion Matrix
    viz.plot_confusion_matrix(model, target_loaders['test'], class_names, "Inference Target Test - Confusion Matrix")

    # 2. ROC Curve
    viz.plot_roc_curve(model, target_loaders['test'], class_names, "Inference Target Test - ROC Curve")

    # 3. Domain Alignment
    viz.plot_umap(model, source_loaders['test'], target_loaders['test'], "Inference Target Test - Domain Alignment")
    
    # 4. Qualitative Latent Space with lighting variation analysis (LIME)
    viz.plot_umap_with_images(
        model=model, 
        source_loader=source_loaders['test'], 
        target_loader=target_loaders['test'], 
        class_names=class_names, 
        prefix="Inference Target Test - Latent Space",
        image_zoom=0.07,
        min_dist_plots=0.45,
        use_lime=cfg.data.use_lime_on_target,
    )
    
    print(f"\nInference completed successfully. Results saved in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Banana-CREDA Inference Script")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained .pth model file")
    parser.add_argument("--output_dir", type=str, default=None, help="Override the output directory defined in the YAML config")
    
    args = parser.parse_args()
    run_inference(args.config, args.model, args.output_dir)

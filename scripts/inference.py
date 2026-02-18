import torch
import argparse
from pathlib import Path

# Imports from the modular package
from banana_creda.config import ExperimentConfig
from banana_creda.data.loader import BananaDataLoader
from banana_creda.models.backbones import BananaModel
from banana_creda.utils.visualizer import BananaVisualizer
from banana_creda.utils.metrics import MetricTracker

def run_inference(config_path: str, model_path: str):
    # 1. Load Configuration and Setup Environment
    cfg = ExperimentConfig.from_yaml(config_path)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    
    # 2. Data Setup (Prioritizing Target Test data for evaluation)
    data_manager = BananaDataLoader(cfg.data)
    
    # We load both loaders to support the alignment visualization discussed in your research
    _, _, src_test, class_names = data_manager.get_split_loaders(cfg.data.synth_data_dir)
    _, _, tgt_test, _ = data_manager.get_split_loaders(cfg.data.orig_data_dir)
    
    # 3. Model Initialization and Loading Weights
    print(f"Loading pre-trained model from: {model_path}")
    model = BananaModel(cfg.model).to(device)
    
    # Load state dict with map_location for CPU/GPU flexibility
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 4. Final Evaluation and Scientific Reports
    output_dir = Path(cfg.experiment.output_dir + "/inference")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    viz = BananaVisualizer(device=device, output_dir=str(output_dir))
    
    print("\nExecuting evaluation on Target Test Set...")

    # Get raw inference data (y_true, y_pred, y_probs, features, images, domains, lightings)
    # Using the extended signature to capture LIME variants and domain info
    inference_results = viz._get_inference_data(model, tgt_test)
    y_true_np, y_pred_np = inference_results[0], inference_results[1]

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
    viz.plot_confusion_matrix(model, tgt_test, class_names, "Inference_Confusion_Matrix")

    # 2. ROC Curve
    viz.plot_roc_curve(model, tgt_test, class_names, "Inference_ROC")
    
    # 3. Qualitative Latent Space with LIME Lassos
    # This reflects your requirement to group images by lighting variation (LIME_01 to LIME_07)
    viz.plot_umap_with_images(
        model=model, 
        source_loader=src_test, 
        target_loader=tgt_test, 
        class_names=class_names, 
        prefix="Latent_Space",
        image_zoom=0.15,
        min_dist_plots=1.9,
        use_lime=False  # Triggers the Convex Hull grouping based on filenames
    )
    
    print(f"\nInference completed successfully. Results saved in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Banana-CREDA Inference Script")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained .pth model file")
    
    args = parser.parse_args()
    run_inference(args.config, args.model)
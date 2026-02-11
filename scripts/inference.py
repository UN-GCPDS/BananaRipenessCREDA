import torch
import argparse
from pathlib import Path

# Imports del paquete
from banana_creda.config import ExperimentConfig
from banana_creda.models.backbones import BananaModel
from banana_creda.data.loader import BananaDataLoader
from banana_creda.utils.visualizer import BananaVisualizer
from banana_creda.utils.metrics import MetricTracker

def main(config_path: str, model_path: str):
    # 1. Load Configuration and Device
    cfg = ExperimentConfig.from_yaml(config_path)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    
    # 2. Load Data (Only Target Test for real inference)
    data_manager = BananaDataLoader(cfg.data)
    # Nota: get_split_loaders devuelve (train, val, test, names)
    _, _, tgt_test, class_names = data_manager.get_split_loaders(cfg.data.orig_data_dir)
    
    # 3. Load Model with Trained Weights
    print(f"Cargando modelo desde: {model_path}")
    model = BananaModel(cfg.model).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 4. Initialize Visualizer
    output_dir = "outputs/inference_results"
    viz = BananaVisualizer(device=device, output_dir=output_dir)
    
    print("Ejecutando inferencia y generando visualizaciones...")
    
    # 5. Generate Visualizations
    viz.plot_confusion_matrix(model, tgt_test, class_names, "Inference_Run")
    viz.plot_roc_curve(model, tgt_test, class_names, "Inference_Run")
    
    # Qualitative analysis with images in the latent space
    viz.plot_umap_with_images(
        model=model, 
        loader=tgt_test, 
        class_names=class_names, 
        prefix="Inference_Run_Samples"
    )
    
    # 6. Calculate Full Metrics (The report you requested)
    print("\nCalculating detailed statistical metrics...")
    
    # Extract the necessary data using the visualizer helper
    y_true, y_pred, _, _, _ = viz._get_inference_data(model, tgt_test)
    
    # Convert to tensors for the tracker
    metrics = MetricTracker.compute_full_metrics(
        preds=torch.from_numpy(y_pred),
        labels=torch.from_numpy(y_true),
        num_classes=len(class_names),
        device=device
    )
    
    # Print the scientific report
    MetricTracker.print_full_report("Target Domain (Original) INFERENCE", metrics, class_names)
    
    print(f"\nInference completed. Files saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for Banana-CREDA")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--model", type=str, required=True, help="Path to model_final.pth")
    args = parser.parse_args()
    main(args.config, args.model)
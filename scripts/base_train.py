"""
Command-line script for training a baseline classification model on the source domain.

This script loads configuration, prepares data loaders for the source domain, 
initializes the baseline model and trainer, and executes the training loop.
It concludes by generating performance reports and saving the model weights.
"""
import argparse
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from pathlib import Path
from typing import Dict, List

# Imports from the modular package
from banana_creda.config import ExperimentConfig
from banana_creda.data.loader import BananaDataLoader
from banana_creda.models.backbones import BananaModel
from banana_creda.utils.visualizer import BananaVisualizer
from banana_creda.utils.reproducibility import set_seed
from banana_creda.utils.metrics import MetricTracker

# Baseline-specific imports
from banana_creda.losses.ce import CELoss
from banana_creda.training.base_trainer import BaselineTrainer

def run_baseline_experiment(config_path: str) -> None:
    """Runs the baseline training and evaluation experiment.

    Args:
        config_path (str): Path to the YAML configuration file.
    """
    # 1. Load Configuration & Setup
    cfg = ExperimentConfig.from_yaml(config_path)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    
    if cfg.training.seed is not None:
        set_seed(cfg.training.seed)
        
    output_path = Path(cfg.experiment.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 2. Data Setup (ONLY Source/Original data)
    data_manager = BananaDataLoader(cfg.data)
    # Ensure that cfg.data.source_data_dir points to your "Real Dataset" (without variations)
    src_train, src_val, src_test, class_names = data_manager.get_split_loaders(cfg.data.source_data_dir)

    # 3. Model, Loss, and Optimizer Initialization
    model = BananaModel(cfg.model).to(device)
    criterion = CELoss().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr, weight_decay=1e-5)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=cfg.training.gamma)

    # 4. Training Execution
    trainer = BaselineTrainer(
        model=model,
        train_loader=src_train,
        val_loader=src_val,
        criterion=criterion,
        optimizer=optimizer,
        config=cfg.training,
        device=device
    )
    
    print(f"\n Starting BASELINE experiment on {device}...")
    trained_model, history = trainer.fit(scheduler=scheduler)

    # 5. Final Evaluation on Source Test Set (To verify the baseline works on its own domain)
    viz = BananaVisualizer(device=device, output_dir=str(output_path))
    
    print("\n Generating Final Statistical Reports on SOURCE Test Set...")
    # Get raw inference data from the source test set
    y_true_np, y_pred_np, y_probs_np, _, _ = viz._get_inference_data(
        trained_model, 
        src_test
    ) 

    metrics = MetricTracker.compute_full_metrics(
        torch.from_numpy(y_pred_np), 
        torch.from_numpy(y_true_np), 
        len(class_names), 
        device
    )
    MetricTracker.print_full_report("Source Domain BASELINE TEST", metrics, class_names)
    
    # Quantitative Visualizations
    viz.plot_confusion_matrix(trained_model, src_test, class_names, "Baseline Source Test")
    viz.plot_roc_curve(trained_model, src_test, class_names, "Baseline Source Test")
    
    # 6. Save the trained weights
    save_file = Path(output_path) / "model_final.pth"
    torch.save(trained_model.state_dict(), save_file)
    print(f"\n Experiment completed. Results saved in: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Banana Baseline Training Script (Source Only)")
    parser.add_argument("--config", type=str, default="configs/base_experiment.yaml", help="Path to YAML config")
    args = parser.parse_args()
    
    run_baseline_experiment(args.config)

"""
Command-line script for training the Domain Adaptation (DA) model using CREDA.

This script manages the full Domain Adaptation experiment, including loading 
source and target datasets, initializing the CREDA model and trainer, 
executing the training loop, and generating comprehensive visualization 
and statistical reports on the target domain.
"""
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
from pathlib import Path

# Imports from the modular package
from banana_creda.config import ExperimentConfig
from banana_creda.data.loader import BananaDataLoader
from banana_creda.models.backbones import BananaModel
from banana_creda.losses.creda import CREDALoss
from banana_creda.training.da_trainer import BananaTrainer
from banana_creda.utils.visualizer import BananaVisualizer
from banana_creda.utils.reproducibility import set_seed
from banana_creda.utils.metrics import MetricTracker

def run_experiment(config_path: str) -> None:
    """Executes the Domain Adaptation experiment using the CREDA algorithm.

    Args:
        config_path (str): Path to the YAML configuration file.
    """
    # 1. Load Configuration (Validation with Pydantic)
    cfg = ExperimentConfig.from_yaml(config_path)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    
    # Reproducibility
    if cfg.training.seed is not None:
        set_seed(cfg.training.seed)
    
    # 2. Data Setup (Source: Normal / Target: Varying Illumination)
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

    # 3. Model, Loss, and Optimizer Initialization
    model = BananaModel(cfg.model).to(device)
    criterion = CREDALoss(cfg.training, cfg.model.num_classes).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr, weight_decay=1e-5)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=cfg.training.gamma)

    # 4. Training Execution
    trainer = BananaTrainer(
        model=model,
        source_loaders=source_loaders,
        target_loaders=target_loaders,
        criterion=criterion,
        optimizer=optimizer,
        config=cfg.training,
        device=device
    )
    
    print(f" Starting CREDA experiment on {device}...")
    trained_model = trainer.fit(scheduler=scheduler)

    # 5. Final Evaluation and Scientific Reports
    output_path = Path(cfg.experiment.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    viz = BananaVisualizer(device=device, output_dir=str(output_path))
    
    print("\n Generating Final Statistical Reports...")

    # Get raw inference data from the test set
    y_true_np, y_pred_np, y_probs_np, _, _ = viz._get_inference_data(trained_model, target_loaders['test']) 

    # Compute and print Full Scientific Report (Precision, Recall, F1, Support)
    # We convert back to tensors for the MetricTracker logic
    metrics = MetricTracker.compute_full_metrics(
        torch.from_numpy(y_pred_np), 
        torch.from_numpy(y_true_np), 
        len(class_names), 
        device
    )
    MetricTracker.print_full_report("Target Domain FINAL TEST", metrics, class_names)
    
    print("\n Generating Visualizations...")
    
    # Quantitative: Confusion Matrix
    viz.plot_confusion_matrix(trained_model, target_loaders['test'], class_names, "Target Test - Confusion Matrix")

    # Quantitative: ROC Curve and AUC
    viz.plot_roc_curve(trained_model, target_loaders['test'], class_names, "Target Test - ROC Curve")
    
    # Save best model weights
    save_file = Path(output_path) / "model_final.pth"
    torch.save(trained_model.state_dict(), save_file)
    print(f"\n Experiment completed. Results saved in: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Banana-CREDA Training Script")
    parser.add_argument("--config", type=str, default="configs/base_experiment.yaml", help="Path to YAML config")
    args = parser.parse_args()
    run_experiment(args.config)
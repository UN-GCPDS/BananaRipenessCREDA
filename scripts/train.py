import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
from pathlib import Path

# Imports del paquete modular que hemos construido
from banana_creda.config import ExperimentConfig
from banana_creda.data.loader import BananaDataLoader
from banana_creda.models.backbones import BananaModel
from banana_creda.losses.creda import CREDALoss
from banana_creda.training.trainer import BananaTrainer
from banana_creda.utils.visualizer import BananaVisualizer
from banana_creda.utils.reproducibility import set_seed
from banana_creda.utils.metrics import MetricTracker

def run_experiment(config_path: str):
    # 1. Load Configuration (Validation with Pydantic)
    cfg = ExperimentConfig.from_yaml(config_path)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    if cfg.training.seed is not None:
        set_seed(cfg.training.seed)
    
    # 2. Data Setup (Separation of Source/Target domains)
    data_manager = BananaDataLoader(cfg.data)
    src_train, src_val, src_test, class_names = data_manager.get_split_loaders(cfg.data.synth_data_dir)
    tgt_train, tgt_val, tgt_test, _ = data_manager.get_split_loaders(cfg.data.orig_data_dir)
    
    source_loaders = {'train': src_train, 'validation': src_val, 'test': src_test}
    target_loaders = {'train': tgt_train, 'validation': tgt_val, 'test': tgt_test}

    # 3. Model, CREDA Loss, and Optimizer Initialization
    model = BananaModel(cfg.model).to(device)
    criterion = CREDALoss(cfg.training, cfg.model.num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr, weight_decay=1e-5)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=cfg.training.gamma)

    # 4. Training (Trainer with AMP and Warm-up support)
    trainer = BananaTrainer(
        model=model,
        source_loaders=source_loaders,
        target_loaders=target_loaders,
        criterion=criterion,
        optimizer=optimizer,
        config=cfg.training
    )
    
    print(f"Starting CREDA experiment in {device}...")
    trained_model = trainer.fit(scheduler=scheduler)

    # 5. Final Evaluation and Advanced Visualizations
    output_path = cfg.experiment.output_dir
    viz = BananaVisualizer(device=device, output_dir=output_path)
    
    print("\nGenerating Final Reports and Visualizations of Latent Space...")

    # Test data
    y_true, y_pred, y_probs, _, _ = viz._get_inference_data(trained_model, target_loaders['test']) 

    # Complete metrics
    metrics = MetricTracker.compute_full_metrics(y_pred, y_true, len(class_names), device)
    MetricTracker.print_full_report("Target Domain FINAL TEST", metrics, class_names)
    
    # Quantitative metrics
    viz.plot_confusion_matrix(trained_model, target_loaders['test'], class_names, "Target_Test")

    # ROC curve
    viz.plot_roc_curve(trained_model, target_loaders['test'], class_names, "Target_Test")
    
    # Domain alignment (red points vs blue points)
    viz.plot_umap(trained_model, source_loaders['test'], target_loaders['test'], "Domain_Alignment")
    
    # Qualitative analysis with images in the latent space
    # We execute it on the Target (Original) domain to see what the network is learning
    viz.plot_umap_with_images(
        model=trained_model, 
        loader=target_loaders['test'], 
        class_names=class_names, 
        prefix="Target_Latent_Space",
        min_dist_plots=0.15, # Adjust to avoid image overlap
        image_zoom=0.07      # Adjust according to your image resolution
    )
    
    # Save best model weights
    save_file = Path(output_path) / "model_final.pth"
    torch.save(trained_model.state_dict(), save_file)
    print(f"Experiment completed. Results and models saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base_experiment.yaml")
    args = parser.parse_args()
    run_experiment(args.config)
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
from pathlib import Path

# Imports del paquete modular que hemos construido
from banana_creda.config import ExperimentConfig
from banana_creda.data.loader import BananaDataLoader
from banana_creda.models.resnet import BananaModel
from banana_creda.losses.creda import CREDALoss
from banana_creda.training.trainer import BananaTrainer
from banana_creda.utils.visualizer import BananaVisualizer

def run_experiment(config_path: str):
    # 1. Cargar Configuración (Validación con Pydantic)
    cfg = ExperimentConfig.from_yaml(config_path)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    
    # 2. Setup de Datos (Separación de dominios Source/Target)
    data_manager = BananaDataLoader(cfg.data)
    src_train, src_val, src_test, class_names = data_manager.get_split_loaders(cfg.data.synth_data_dir)
    tgt_train, tgt_val, tgt_test, _ = data_manager.get_split_loaders(cfg.data.orig_data_dir)
    
    source_loaders = {'train': src_train, 'validation': src_val, 'test': src_test}
    target_loaders = {'train': tgt_train, 'validation': tgt_val, 'test': tgt_test}

    # 3. Inicializar Modelo, Pérdida CREDA y Optimizer
    model = BananaModel(cfg.model).to(device)
    criterion = CREDALoss(cfg.training, cfg.model.num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr, weight_decay=1e-5)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=cfg.training.gamma)

    # 4. Motor de Entrenamiento (Trainer con soporte AMP y Warm-up)
    trainer = BananaTrainer(
        model=model,
        source_loaders=source_loaders,
        target_loaders=target_loaders,
        criterion=criterion,
        optimizer=optimizer,
        config=cfg.training
    )
    
    print(f"Iniciando experimento CREDA en {device}...")
    trained_model = trainer.fit(scheduler=scheduler)

    # 5. Evaluación Final y Visualización Avanzada
    output_path = "outputs/experiment_1"
    viz = BananaVisualizer(device=device, output_dir=output_path)
    
    print("\nGenerando Reportes Finales y Visualizaciones de Espacio Latente...")
    
    # Métricas cuantitativas tradicionales
    viz.plot_confusion_matrix(trained_model, target_loaders['test'], class_names, "Target_Test")
    
    # Alineación global de dominios (puntos rojos vs azules)
    viz.plot_umap(trained_model, source_loaders['test'], target_loaders['test'], "Domain_Alignment")
    
    # NUEVO: Análisis cualitativo con imágenes en el espacio latente
    # Lo ejecutamos sobre el dominio Target (Original) para ver qué está aprendiendo la red
    viz.plot_umap_with_images(
        model=trained_model, 
        loader=target_loaders['test'], 
        class_names=class_names, 
        prefix="Target_Latent_Space",
        min_dist_plots=0.15, # Ajusta para evitar que las fotos se encimen demasiado
        image_zoom=0.07      # Ajusta según la resolución de tus imágenes
    )
    
    # Guardar pesos del mejor modelo encontrado
    save_file = Path(output_path) / "model_final.pth"
    torch.save(trained_model.state_dict(), save_file)
    print(f"Experimento completado. Resultados y modelos guardados en {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base_experiment.yaml")
    args = parser.parse_args()
    run_experiment(args.config)
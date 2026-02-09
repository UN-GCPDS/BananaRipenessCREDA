import argparse
from banana_creda.config import ExperimentConfig

def main():
    parser = argparse.ArgumentParser(description="CREDA Training Script")
    parser.add_argument("--config", type=str, default="configs/base_experiment.yaml", help="Path to YAML config")
    args = parser.parse_args()

    # Carga y validación automática con Pydantic
    cfg = ExperimentConfig.from_yaml(args.config)
    
    print(f"Configuración cargada: {cfg.model.backbone} para {cfg.model.num_classes} clases.")
    print(f"Entrenando en: {cfg.training.device}")

    # Aquí vendrá la inicialización de los componentes en los siguientes pasos...

if __name__ == "__main__":
    main()
import torch
import argparse
from banana_creda.config import ExperimentConfig
from banana_creda.models.backbones import BananaModel
from banana_creda.data.loader import BananaDataLoader
from banana_creda.utils.visualizer import BananaVisualizer

def main(config_path: str, model_path: str):
    cfg = ExperimentConfig.from_yaml(config_path)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    
    # Cargar solo datos de test originales
    data_manager = BananaDataLoader(cfg.data)
    _, _, tgt_test, class_names = data_manager.get_split_loaders(cfg.data.orig_data_dir)
    
    # Cargar Modelo
    model = BananaModel(cfg.model).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Visualizar
    viz = BananaVisualizer(device=device, output_dir="outputs/inference_results")
    viz.plot_confusion_matrix(model, tgt_test, class_names, "Inference_Run")
    print("Inferencia finalizada. Resultados en outputs/inference_results")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    main(args.config, args.model)
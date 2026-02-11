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
    # 1. Cargar Configuración y Device
    cfg = ExperimentConfig.from_yaml(config_path)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    
    # 2. Cargar Datos (Solo Target Test para inferencia real)
    data_manager = BananaDataLoader(cfg.data)
    # Nota: get_split_loaders devuelve (train, val, test, names)
    _, _, tgt_test, class_names = data_manager.get_split_loaders(cfg.data.orig_data_dir)
    
    # 3. Cargar Modelo con Pesos Entrenados
    print(f"📦 Cargando modelo desde: {model_path}")
    model = BananaModel(cfg.model).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 4. Inicializar Visualizador
    output_dir = "outputs/inference_results"
    viz = BananaVisualizer(device=device, output_dir=output_dir)
    
    print("🚀 Ejecutando inferencia y generando visualizaciones...")
    
    # 5. Generar Visualizaciones
    viz.plot_confusion_matrix(model, tgt_test, class_names, "Inference_Run")
    viz.plot_roc_curve(model, tgt_test, class_names, "Inference_Run")
    
    # UMAP Cualitativo (Muestras reales en el espacio latente)
    viz.plot_umap_with_images(
        model=model, 
        loader=tgt_test, 
        class_names=class_names, 
        prefix="Inference_Run_Samples"
    )
    
    # 6. CÁLCULO DE FULL METRICS (El reporte que pediste)
    print("\n🔬 Calculando métricas estadísticas detalladas...")
    
    # Extraemos los datos necesarios usando el helper del visualizador
    y_true, y_pred, _, _, _ = viz._get_inference_data(model, tgt_test)
    
    # Convertimos a tensores para el tracker
    metrics = MetricTracker.compute_full_metrics(
        preds=torch.from_numpy(y_pred),
        labels=torch.from_numpy(y_true),
        num_classes=len(class_names),
        device=device
    )
    
    # Imprimir el reporte estilo científico
    MetricTracker.print_full_report("Target Domain (Original) INFERENCE", metrics, class_names)
    
    print(f"\n✅ Inferencia finalizada. Archivos guardados en: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de Inferencia para Banana-CREDA")
    parser.add_argument("--config", type=str, required=True, help="Ruta al config.yaml")
    parser.add_argument("--model", type=str, required=True, help="Ruta al model_final.pth")
    args = parser.parse_args()
    main(args.config, args.model)
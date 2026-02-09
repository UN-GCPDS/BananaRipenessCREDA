import kagglehub
from pathlib import Path
from banana_creda.config import ExperimentConfig

def download():
    # Solo necesitamos la config de datos para saber dónde guardarlos
    # pero usamos ExperimentConfig para mantener consistencia
    try:
        print("Autenticando y descargando dataset de Kaggle...")
        path = kagglehub.dataset_download('lucasiturriago/bananaripeness')
        print(f"Dataset disponible en: {path}")
        
        # Aquí podrías añadir lógica para mover o crear symlinks 
        # hacia las carpetas definidas en tu YAML si fuera necesario.
    except Exception as e:
        print(f"Error al descargar: {e}")

if __name__ == "__main__":
    download()
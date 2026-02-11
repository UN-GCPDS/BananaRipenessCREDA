import kagglehub
from pathlib import Path
from banana_creda.config import ExperimentConfig

def download():
    # We only need the data config to know where to save the data
    # but we use ExperimentConfig to maintain consistency
    try:
        print("Authenticating and downloading dataset from Kaggle...")
        path = kagglehub.dataset_download('lucasiturriago/bananaripeness')
        print(f"Dataset available at: {path}")
        
        # You can add logic to move or create symlinks 
        # towards the folders defined in your YAML if needed.
    except Exception as e:
        print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    download()
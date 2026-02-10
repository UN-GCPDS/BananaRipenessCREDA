# Banana Ripeness Classification using CREDA

This project implements a Domain Adaptation approach for classifying banana ripeness stages using **Class-Regularized Entropy Domain Adaptation (CREDA)**. The goal is to adapt a model trained on synthetic data (Source Domain) to perform well on real-world banana images (Target Domain), where labels might be scarce or unavailable.

## Features

- **Domain Adaptation**: Utilizes CREDA to align feature distributions between synthetic and real domains.
- **Backbone**: Uses a ResNet (e.g., ResNet-34) pretrained on ImageNet as the feature extractor.
- **Uncertainty Awareness**: Incorporates uncertainty weighting to improve alignment robustness.
- **Visualization**: Includes tools for visualizing Confusion Matrices and UMAP embeddings of the latent space.
- **Configurable**: Driven by `yaml` configuration files for flexible experimentation.

## Project Structure

```
BananaRipenessCREDA/
├── configs/                # Configuration files (YAML)
│   └── base_experiment.yaml
├── data/                   # Data storage (if not using external paths)
├── notebooks/              # Jupyter notebooks for exploration
├── scripts/                # Executable run scripts
│   ├── train.py            # Main training script
│   ├── inference.py        # Inference script
│   └── download_data.py    # Data download utility
├── src/                    # Source code
│   └── banana_creda/       # Main package
│       ├── data/           # Data loading logic
│       ├── losses/         # CREDA loss implementation
│       ├── models/         # Model architectures
│       ├── training/       # Training loop
│       ├── utils/          # Metrics and visualization
│       └── config.py       # Configuration schemas
├── tests/                  # Unit tests
├── README.md               # Project documentation
└── pyproject.toml          # Project dependencies and metadata
```

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/BananaRipenessCREDA.git
    cd BananaRipenessCREDA
    ```

2.  **Install dependencies**:
    It is recommended to use a virtual environment (e.g., `venv` or `conda`).
    ```bash
    pip install -e .
    ```
    This will install `banana_creda` in editable mode along with all required packages listed in `pyproject.toml`.

    For development dependencies (testing, linting):
    ```bash
    pip install -e .[dev]
    ```

## Configuration

The experiment configuration is defined in `configs/base_experiment.yaml`. You can modify this file or create new ones for different experiments.

Key parameters include:

-   **data**:
    -   `orig_data_dir`: Path to the real-world dataset (Target).
    -   `synth_data_dir`: Path to the synthetic dataset (Source).
    -   `batch_size`: Batch size for training.
-   **model**:
    -   `backbone`: ResNet architecture (e.g., `resnet18`, `resnet34`).
    -   `num_classes`: Number of ripeness stages (default: 4).
-   **training**:
    -   `epochs`: Number of training epochs.
    -   `lr`: Learning rate.
    -   `lambda_creda`: Weight for the CREDA loss term.
    -   `lambda_entropy`: Weight for the entropy minimization term.
    -   `sigma`: Gaussian kernel width ('auto' for heuristic).

## Usage

### Training

To start training the domain adaptation model:

```bash
python scripts/train.py --config configs/base_experiment.yaml
```

This script will:
1.  Load the configuration.
2.  Initialize the datasets and model.
3.  Train the model using the Source (labeled) and Target (unlabeled) data.
4.  Save the best model to `outputs/experiment_1/model_final.pth`.
5.  Generate evaluation plots (Confusion Matrix, UMAP) in `outputs/experiment_1`.

### Inference

To run inference on the target test set with a trained model:

```bash
python scripts/inference.py --config configs/base_experiment.yaml --model outputs/experiment_1/model_final.pth
```

### Testing

To run the unit tests:

```bash
pytest tests/
```

## Outputs

The training process generates the following outputs in the specified output directory:

-   **`model_final.pth`**: Saved model weights.
-   **`Target_Test_cm.png`**: Confusion Matrix on the real-world test set.
-   **`Final_Alignment_umap_alignment.png`**: UMAP visualization showing the alignment between Source and Target domains.

## License

This project is licensed under the MIT License.

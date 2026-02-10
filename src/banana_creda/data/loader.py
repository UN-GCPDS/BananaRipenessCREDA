import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Dict, Tuple, List, Optional
from banana_creda.config import DataConfig

class BananaDataLoader:
    """
    Gestiona la carga de datos para los dominios Original (Target) 
    y Sintético (Source).
    """
    def __init__(self, config: DataConfig):
        self.config = config
        self.transforms = self._get_transforms()

    def _get_transforms(self) -> Dict[str, transforms.Compose]:
        """Define los pipelines de transformación para entrenamiento e inferencia."""
        return {
            'train': transforms.Compose([
                transforms.Resize((self.config.img_size, self.config.img_size)),
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(self.config.imagenet_mean, self.config.imagenet_std)
            ]),
            'inference': transforms.Compose([
                transforms.Resize((self.config.img_size, self.config.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(self.config.imagenet_mean, self.config.imagenet_std)
            ]),
        }

    def get_split_loaders(self, data_dir: str) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
        """
        Crea los DataLoaders para train, validation y test de un directorio específico.
        """
        splits = ['train', 'validation', 'test']
        loaders = {}
        
        # Primero cargamos el dataset de entrenamiento para obtener las clases
        train_path = os.path.join(data_dir, 'train')
        temp_dataset = datasets.ImageFolder(root=train_path)
        class_names = temp_dataset.classes

        for split in splits:
            transform_type = 'train' if split == 'train' else 'inference'
            dataset_path = os.path.join(data_dir, split)
            
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"No se encontró la carpeta: {dataset_path}")

            dataset = datasets.ImageFolder(
                root=dataset_path,
                transform=self.transforms[transform_type]
            )

            loaders[split] = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=(split == 'train'),
                num_workers=self.config.num_workers,
                pin_memory=True,
                # Optimizaciones de rendimiento para GPUs modernas
                prefetch_factor=2 if self.config.num_workers > 0 else None,
                persistent_workers=True if self.config.num_workers > 0 else False
            )

        return loaders['train'], loaders['validation'], loaders['test'], class_names
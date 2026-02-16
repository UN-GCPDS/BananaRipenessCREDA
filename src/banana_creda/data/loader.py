import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Dict, Tuple, List
from banana_creda.config import DataConfig


class BananaImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, return_lime_variant=False):
        super().__init__(root=root, transform=transform)
        self.return_lime_variant = return_lime_variant

    def _extract_lime_variant(self, path: str):
        filename = os.path.basename(path)
        name, _ = os.path.splitext(filename)

        try:
            variant = name.split("_")[-1]
            return int(variant)
        except Exception:
            return None

    def __getitem__(self, index):
        image, label = super().__getitem__(index)

        if not self.return_lime_variant:
            return image, label

        path, _ = self.samples[index]
        lime_variant = self._extract_lime_variant(path)

        return image, label, lime_variant


class BananaDataLoader:
    """
    Manages the loading of data for the Original (Target)
    and Synthetic (Source) domains.
    """
    def __init__(self, config: DataConfig):
        self.config = config
        self.transforms = self._get_transforms()

    def _get_transforms(self) -> Dict[str, transforms.Compose]:
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

    def get_split_loaders(self, data_dir: str, return_lime_variant: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
        """
        Returns the data loaders for the train, validation, and test splits.
        
        Args:
            data_dir: Path to the data directory.
            return_lime_variant: Whether to return the lime variant.
        
        Returns:
            Tuple of data loaders for train, validation, and test splits.
        """
        splits = ['train', 'validation', 'test']
        loaders = {}

        train_path = os.path.join(data_dir, 'train')
        temp_dataset = datasets.ImageFolder(root=train_path)
        class_names = temp_dataset.classes

        for split in splits:
            transform_type = 'train' if split == 'train' else 'inference'
            dataset_path = os.path.join(data_dir, split)

            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Directory not found: {dataset_path}")

            dataset = BananaImageFolder(
                root=dataset_path,
                transform=self.transforms[transform_type],
                return_lime_variant=return_lime_variant
            )

            loaders[split] = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=(split == 'train'),
                num_workers=self.config.num_workers,
                pin_memory=True,
                prefetch_factor=2 if self.config.num_workers > 0 else None,
                persistent_workers=True if self.config.num_workers > 0 else False
            )

        return loaders['train'], loaders['validation'], loaders['test'], class_names

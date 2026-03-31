import os
import pytest
from pathlib import Path
from PIL import Image
import torch
from banana_creda.data.loader import BananaImageFolder, BananaDataLoader
from banana_creda.config import DataConfig

def create_fake_dataset(base_dir: Path):
    """
    Creates a fake ImageNet-style dataset for testing.
    Structure:
    base_dir/
      train/
        class1/
          img_0.jpg
          img_1_5.jpg  # 5 is the LIME variant
        class2/
          img_2.jpg
      validation/
        class1/
          img_3.jpg
      test/
        class1/
          img_4.jpg
    """
    for split in ['train', 'validation', 'test']:
        split_dir = base_dir / split
        for cls in ['class1', 'class2']:
            cls_dir = split_dir / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a simple 10x10 RGB image
            img = Image.new('RGB', (10, 10), color='red')
            
            if split == 'train' and cls == 'class1':
                # These have specific variants
                img.save(cls_dir / "img_0_0.jpg")
                img.save(cls_dir / "img_1_5.jpg")
            else:
                # Default to variant 0
                img.save(cls_dir / "img_test_0.jpg")

def test_banana_image_folder_extraction(tmp_path):
    # ImageFolder needs a real directory structure
    create_fake_dataset(tmp_path)
    folder = BananaImageFolder(root=str(tmp_path / "train"), return_lime_variant=True)
    
    assert folder._extract_lime_variant("some/path/img_1_5.jpg") == 5
    assert folder._extract_lime_variant("some/path/img_0.jpg") == 0
    assert folder._extract_lime_variant("some/path/plain.jpg") is None

def test_banana_data_loader(tmp_path):
    create_fake_dataset(tmp_path)
    
    config = DataConfig(
        source_data_dir=tmp_path,
        batch_size=2,
        img_size=224,
        num_workers=0  # Prevents multiprocessing issues in tests
    )
    
    loader_manager = BananaDataLoader(config)
    train_loader, val_loader, test_loader, class_names = loader_manager.get_split_loaders(str(tmp_path))
    
    assert "class1" in class_names
    assert "class2" in class_names
    assert len(train_loader.dataset) == 3 # 2 in class1, 1 in class2 (from "img_test.jpg")
    
    # Check if we can get a batch
    images, labels = next(iter(train_loader))
    assert images.shape == (2, 3, 224, 224)
    assert labels.shape == (2,)

def test_banana_image_folder_with_lime(tmp_path):
    create_fake_dataset(tmp_path)
    
    config = DataConfig(
        source_data_dir=tmp_path,
        batch_size=2,
        img_size=224,
        num_workers=0
    )
    
    loader_manager = BananaDataLoader(config)
    train_loader, _, _, _ = loader_manager.get_split_loaders(str(tmp_path), return_lime_variant=True)
    
    # Fetch batch - Note: we need to ensure images have suffixes for this to work
    # In create_fake_dataset, only class1 in train has them.
    # We might need to handle the None case in loader.py if it's expected to have non-LIME images mixed in.
    images, labels, lime_variants = next(iter(train_loader))
    assert isinstance(lime_variants, torch.Tensor)
    assert len(lime_variants) == 2

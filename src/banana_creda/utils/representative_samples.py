import torch
from torch import Tensor
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List

def representative_samples(
    loader: DataLoader, 
    num_classes: int,
    samples_per_class: int = 12
) -> Tuple[Dict[int, Tensor], bool]:
    """
    Iterates through a DataLoader to find multiple images for each class.

    This function tracks the number of samples collected per class and exits 
    early once the quota (samples_per_class) is met for every class.

    Args:
        loader (DataLoader): The PyTorch DataLoader to iterate over.
        num_classes (int): Total number of unique classes.
        samples_per_class (int): Number of images to collect per class (default 12).

    Returns:
        Tuple[Dict[int, Tensor], bool]: 
            - A dictionary where keys are class indices (int) and values are 
              stacked tensors of shape [samples_per_class, C, H, W].
            - A boolean indicating if all classes reached the required count.
    """
    # Storage for collected image tensors
    storage: Dict[int, List[Tensor]] = {i: [] for i in range(num_classes)}
    
    # Track how many samples we still need for each class
    counts = torch.zeros(num_classes, dtype=torch.long)
    
    # Pre-generate class targets for vectorized comparison
    class_targets_base = torch.arange(num_classes).view(-1, 1)

    for batch in loader:
        images, labels = batch[0], batch[1]
        device = labels.device
        
        # 1. Vectorized comparison: [num_classes, batch_size]
        class_targets = class_targets_base.to(device)
        is_class_present = (labels == class_targets)

        # 2. Iterate only through classes that still need samples
        for cls_idx in range(num_classes):
            current_count = counts[cls_idx].item()
            if current_count >= samples_per_class:
                continue

            # Find where this class exists in the current batch
            match_indices = torch.where(is_class_present[cls_idx])[0]
            
            if len(match_indices) > 0:
                # Determine how many more we need
                needed = samples_per_class - current_count
                # Take only what is available or what is needed
                to_take = match_indices[:needed]
                
                for idx in to_take:
                    storage[cls_idx].append(images[idx].detach().cpu())
                
                counts[cls_idx] += len(to_take)

        # 3. Early exit if all classes have reached the samples_per_class quota
        if (counts >= samples_per_class).all():
            break

    # 4. Post-process: Stack lists into single tensors and check completeness
    final_samples: Dict[int, Tensor] = {}
    all_found = True

    for cls_idx, img_list in storage.items():
        if len(img_list) > 0:
            final_samples[cls_idx] = torch.stack(img_list)
        
        if len(img_list) < samples_per_class:
            all_found = False

    return final_samples, all_found
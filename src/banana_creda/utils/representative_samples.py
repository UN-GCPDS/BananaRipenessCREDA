import torch
from torch import Tensor
from torch.utils.data import DataLoader
from typing import Dict, Tuple

def representative_samples(
    loader: DataLoader, 
    num_classes: int, 
) -> Tuple[Dict[int, Tensor], bool]:
    """
    Iterates through a DataLoader to find the first occurring image for each class.

    This function uses vectorized broadcasting to identify indices within batches
    and exits early as soon as at least one image for every class has been collected.

    Args:
        loader (DataLoader): The PyTorch DataLoader to iterate over.
        num_classes (int): The total number of unique classes in the dataset.

    Returns:
        Tuple[Dict[int, Tensor], bool]: 
            - A dictionary where keys are class indices (int) and values are 
              the corresponding first-found image tensors.
            - A boolean indicating if all classes were successfully found (True) 
              or if the loader exhausted before finding everything (False).
    """
    # Storage for found images and tracking mask
    found_images: Dict[int, Tensor] = {}
    missing_classes_mask = torch.ones(num_classes, dtype=torch.bool)
    
    # Pre-generate targets once
    class_targets_base = torch.arange(num_classes)

    for batch in loader:
        # Handle cases where loader might return (img, label, ...)
        images, labels = batch[0], batch[1]
        device = labels.device

        # 1. Vectorized comparison on the correct device
        # class_targets: [num_classes, 1], labels: [batch_size]
        class_targets = class_targets_base.to(device).view(-1, 1)
        is_class_present = (labels == class_targets)
        
        # 2. Filter mask to only look for 'missing' classes in this batch
        valid_in_batch = is_class_present & missing_classes_mask.to(device).view(-1, 1)
        
        # 3. Find first occurrence in this batch for each missing class
        presence_in_batch = valid_in_batch.any(dim=1)
        
        if presence_in_batch.any():
            # Get the first index for each class found
            indices = valid_in_batch.float().argmax(dim=1)
            
            # Extract and store images for the newly found classes
            found_indices = torch.where(presence_in_batch)[0]
            for cls_idx in found_indices.tolist():
                img_idx = int(indices[cls_idx])
                # Store on CPU to keep GPU memory free for the model
                found_images[cls_idx] = images[img_idx].detach().cpu()
                
            # Update our tracker (move presence back to CPU mask)
            missing_classes_mask[presence_in_batch.cpu()] = False

        # 4. Early exit if all classes are found
        if not missing_classes_mask.any():
            return found_images, True

    return found_images, False
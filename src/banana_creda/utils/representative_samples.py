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
    # Pre-generate class targets for broadcasting: shape [num_classes, 1]
    class_targets = torch.arange(num_classes).view(-1, 1)
    
    # Storage for found images and tracking set
    found_images: Dict[int, Tensor] = {}
    missing_classes_mask = torch.ones(num_classes, dtype=torch.bool)

    for batch in loader:
        images, labels = batch

        # 1. Vectorized comparison: [num_classes, batch_size]
        # Only check for classes we haven't found yet to save computation
        is_class_present = (labels == class_targets)
        
        # 2. Filter mask to only look for 'missing' classes in this batch
        valid_in_batch = is_class_present & missing_classes_mask.view(-1, 1)
        
        # 3. Find first occurrence in this batch for each missing class
        # presence_in_batch: [num_classes] bool tensor
        presence_in_batch = valid_in_batch.any(dim=1)
        
        if presence_in_batch.any():
            # Get the first index for each class that was found in this batch
            indices = valid_in_batch.float().argmax(dim=1)
            
            # Extract and store images for the newly found classes
            found_indices = torch.where(presence_in_batch)[0]
            for cls_idx in found_indices.tolist():
                img_idx = int(indices[cls_idx])
                found_images[cls_idx] = images[img_idx].detach().cpu()
                
            # Update our tracker: remove found classes from the 'missing' mask
            missing_classes_mask[presence_in_batch] = False

        # 4. Early exit if the 'missing' mask is all False (all found)
        if not missing_classes_mask.any():
            return found_images, True

    # If we reach here, some classes were not found in the entire loader
    return found_images, False
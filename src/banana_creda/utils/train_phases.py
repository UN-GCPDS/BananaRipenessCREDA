import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional, Set
from banana_creda.config import TrainConfig

def get_training_phase(
    model: nn.Module, 
    phase: int, 
    config: TrainConfig
) -> Tuple[optim.Optimizer, Optional[optim.lr_scheduler.LRScheduler]]:
    """Applies gradual unfreezing to the model based on the specified training phase.
    
    This function implements a curriculum learning strategy by progressively 
    unfreezing layers from the top (classifier) down to the bottom (encoder) 
    of the network. It assigns different learning rates to different parameter groups.

    Args:
        model (nn.Module): The neural network model to be configured. Must have 
            `encoder` (nn.Sequential) and `classifier` attributes.
        phase (int): The current training phase (1 to 4).
        config (TrainConfig): Configuration object containing learning rates and 
            scheduler hyperparameters.

    Returns:
        Tuple[optim.Optimizer, Optional[LRScheduler]]: 
            A tuple containing the configured Adam optimizer and an optional 
            learning rate scheduler.
    """
    scheduler = None
    
    match phase:
        case 1:
            # Phase 1: Train ONLY the classifier. Freeze the entire encoder.
            for param in model.encoder.parameters():
                param.requires_grad = False
                
            for param in model.classifier.parameters():
                param.requires_grad = True

            params = [
                {'params': model.classifier.parameters(), 'lr': config.lr}
            ]
            optimizer = optim.Adam(params)

        case 2:
            # Phase 2: Unfreeze normalization layers (BatchNorm/LayerNorm) + classifier.
            for param in model.encoder.parameters():
                param.requires_grad = False

            norm_params = []
            for m in model.encoder.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                    for param in m.parameters():
                        param.requires_grad = True
                    norm_params.extend(list(m.parameters()))

            for param in model.classifier.parameters():
                param.requires_grad = True

            params = [
                {'params': model.classifier.parameters(), 'lr': config.lr},
                {'params': norm_params, 'lr': config.transfer_lr}
            ]
            optimizer = optim.Adam(params)

        case 3:
            # Phase 3: Unfreeze the last block of the encoder + norm layers + classifier.
            for param in model.encoder.parameters():
                param.requires_grad = False

            added_params: Set[nn.Parameter] = set()

            # Unfreeze the last block (index -1)
            last_block_params = list(model.encoder[-1].parameters())
            for param in last_block_params:
                param.requires_grad = True
                added_params.add(param)

            norm_params = []
            for m in model.encoder.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                    for param in m.parameters():
                        if param not in added_params:
                            param.requires_grad = True
                            norm_params.append(param)
                            added_params.add(param)

            for param in model.classifier.parameters():
                param.requires_grad = True

            params = [
                {'params': model.classifier.parameters(), 'lr': config.lr},
                {'params': norm_params, 'lr': config.transfer_lr},
                {'params': last_block_params, 'lr': config.transfer_lr}
            ]
            optimizer = optim.Adam(params)

        case 4:
            # Phase 4: Unfreeze the last two blocks + norm layers + classifier. Add Scheduler.
            for param in model.encoder.parameters():
                param.requires_grad = False

            added_params: Set[nn.Parameter] = set()

            # Unfreeze the penultimate block (index -2)
            penultimate_block_params = list(model.encoder[-2].parameters())
            for param in penultimate_block_params:
                param.requires_grad = True
                added_params.add(param)

            # Unfreeze the last block (index -1)
            last_block_params = list(model.encoder[-1].parameters())
            for param in last_block_params:
                param.requires_grad = True
                added_params.add(param)

            norm_params = []
            for m in model.encoder.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                    for param in m.parameters():
                        if param not in added_params:
                            param.requires_grad = True
                            norm_params.append(param)
                            added_params.add(param)

            for param in model.classifier.parameters():
                param.requires_grad = True

            params = [
                {'params': model.classifier.parameters(), 'lr': config.lr},
                {'params': norm_params, 'lr': config.transfer_lr},
                {'params': penultimate_block_params, 'lr': config.transfer_lr},
                {'params': last_block_params, 'lr': config.transfer_lr}
            ]
            optimizer = optim.Adam(params)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)
        
        case _:
            # Default fallback: Unfreeze the entire model for full fine-tuning.
            for param in model.parameters():
                param.requires_grad = True
            
            # Using the fine-tuning LR for the whole model to prevent catastrophic forgetting.
            optimizer = optim.Adam(model.parameters(), lr=config.lr)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)

    return optimizer, scheduler
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Dict, Any, List
from tqdm import tqdm

from banana_creda.utils.train_phases import get_training_phase
from banana_creda.utils.metrics import MetricTracker
from banana_creda.utils.formatter import format_time

class TransferTrainer:
    """Trainer class for transfer learning experiments without Domain Adaptation.

    Handles multi-phase training (e.g., freezing/unfreezing layers), validation, 
    metric tracking, and best checkpoint recovery.

    Attributes:
        model (nn.Module): The neural network model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        config (Any): Configuration object containing training hyperparameters.
        device (torch.device): Computing device.
        best_val_acc (float): Highest validation accuracy achieved.
        best_model_weights (Optional[Dict[str, torch.Tensor]]): State dict of the best model.
        history (Dict[str, List[float]]): Record of losses and accuracies over epochs.
        optimizer (optim.Optimizer): Optimization algorithm (initialized in phases).
        scheduler (Optional[LRScheduler]): Learning rate scheduler (initialized in phases).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        config: Any,
        device: torch.device
    ) -> None:
        """Initializes the TransferTrainer.

        Args:
            model (nn.Module): The model to train.
            train_loader (DataLoader): Source of training samples.
            val_loader (DataLoader): Source of validation samples.
            criterion (nn.Module): Loss criterion.
            config (Any): Training configuration object.
            device (torch.device): Device to run training on.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.config = config
        self.device = device

        # State tracking
        self.best_val_acc: float = 0.0
        self.best_model_weights: Optional[Dict[str, torch.Tensor]] = copy.deepcopy(model.state_dict())
        
        # Metrics history
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def fit(self) -> Tuple[nn.Module, Dict[str, List[float]]]:
        """Executes the full multi-phase training and validation process.

        Returns:
            Tuple[nn.Module, Dict[str, List[float]]]: 
                The model with the best validation weights and the training history.
        """
        epochs: int = self.config.transfer_epochs
        src_classes = self.train_loader.dataset.classes
        phases = getattr(self.config, 'epochs_phases', None)
        if (not isinstance(phases, list) or 
            len(phases) != 4 or 
            not all(isinstance(x, int) for x in phases)):
            
            print("Warning: 'epochs_phases' not valid or missing in config.")
            print("Using default configuration: [0, 5, 10, 15]")
            epochs_phases = [0, 5, 10, 15]
        else:
            epochs_phases = phases

        total_train_start = time.time()

        for epoch in range(epochs):
            # Check if a new training phase should start
            if epoch in epochs_phases:
                phase_idx = epochs_phases.index(epoch) + 1
                self.optimizer, self.scheduler = get_training_phase(self.model, phase_idx, self.config)
            
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 40)
            
            # 1. Training Phase
            train_loss, train_acc, epoch_time = self._train_epoch()
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            formatted_time = format_time(epoch_time)
            print(f"[Train] Time: {formatted_time} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
            
            # 2. Validation Phase (using MetricTracker and evaluate)
            val_acc = self.evaluate(self.val_loader, src_classes, "Val")
            self.history['val_loss'].append(0.0)  # Can be calculated if needed
            self.history['val_acc'].append(val_acc)
            
            # 3. Learning Rate Scheduling
            if self.scheduler is not None:
                self.scheduler.step()
                
            # 4. Checkpoint saving
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_weights = copy.deepcopy(self.model.state_dict())
                print(f"New best model found! (Val Acc: {self.best_val_acc:.4f})")

        total_time = time.time() - total_train_start
        print(f"\n{' TRAINING COMPLETE ':=^50}")
        print(f"Total Duration: {format_time(total_time)}")
        print(f"Best Accuracy: {self.best_val_acc:.4f}")
        print("="*50)
        
        # Restore the best weights before returning
        if self.best_model_weights is not None:
            self.model.load_state_dict(self.best_model_weights)
            
        return self.model, self.history

    def _train_epoch(self) -> Tuple[float, float, float]:
        """Runs a single epoch of transfer training.

        Returns:
            Tuple[float, float, float]: 
                Average batch loss, overall accuracy, and total epoch time in seconds.
        """
        self.model.train()
        running_loss: float = 0.0
        
        # Accumulators for MetricTracker
        all_preds: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []
        
        epoch_start = time.time()
        pbar = tqdm(self.train_loader, desc="Training (Transfer)", leave=False)
        
        for batch in pbar:
            imgs: torch.Tensor = batch[0].to(self.device)
            labels: torch.Tensor = batch[1].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            logits: torch.Tensor = self.model(imgs, mode='class')
            loss: torch.Tensor = self.criterion(logits, labels)
            
            # Backward pass & optimization
            loss.backward()
            self.optimizer.step()
            
            # Batch metrics
            batch_size: int = imgs.size(0)
            running_loss += loss.item() * batch_size
            
            # Store predictions for accuracy calculation
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.detach())
            all_labels.append(labels.detach())
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        epoch_time = time.time() - epoch_start
        epoch_loss: float = running_loss / len(self.train_loader.dataset)
        
        # Calculate global accuracy using MetricTracker
        num_classes = len(self.train_loader.dataset.classes)
        overall_acc, _ = MetricTracker.compute_accuracy(
            torch.cat(all_preds), torch.cat(all_labels), num_classes
        )
        
        return epoch_loss, overall_acc, epoch_time

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, class_names: List[str], prefix: str = "Val") -> float:
        """Evaluates the model on a given dataset.

        Args:
            loader (DataLoader): The dataset to evaluate.
            class_names (List[str]): List of human-readable class names.
            prefix (str): Label for the evaluation phase.

        Returns:
            float: Overall classification accuracy.
        """
        self.model.eval()
        all_preds: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []
        
        pbar = tqdm(loader, desc=f"Evaluating ({prefix})", leave=False)
        
        for batch in pbar:
            imgs: torch.Tensor = batch[0].to(self.device)
            labels: torch.Tensor = batch[1].to(self.device)
            
            # Forward pass
            logits: torch.Tensor = self.model(imgs, mode='class')
            
            # Store predictions
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds)
            all_labels.append(labels)

        # Compute metrics via MetricTracker
        overall_acc, per_class_acc = MetricTracker.compute_accuracy(
            torch.cat(all_preds), torch.cat(all_labels), len(class_names)
        )
        
        MetricTracker.print_summary(prefix, overall_acc, per_class_acc, class_names)
        
        return overall_acc
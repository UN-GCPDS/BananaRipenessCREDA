import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Dict, Any, List
from tqdm import tqdm

from banana_creda.utils.metrics import MetricTracker

class BaselineTrainer:
    """
    Trainer class for baseline classification models without Domain Adaptation.
    Handles the training loop, validation, metric tracking, and best checkpoint saving.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        config: Any,
        device: torch.device
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = device

        # State tracking
        self.best_val_acc: float = 0.0
        self.best_model_weights: Optional[Dict[str, torch.Tensor]] = copy.deepcopy(model.state_dict())
        
        # Metrics history
        self.history: Dict[str, list] = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def _format_time(self, seconds: float) -> str:
        """Converts seconds to MM:SS format."""
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"

    def fit(
        self, 
        scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
    ) -> Tuple[nn.Module, Dict[str, list]]:
        
        epochs: int = getattr(self.config, 'epochs', 10) # Fallback to 10 if not defined
        src_classes = self.train_loader.dataset.classes
        total_train_start = time.time()

        for epoch in range(epochs):
            lr_current = self.optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch+1}/{epochs} | LR: {lr_current:.6f}")
            print("-" * 40)
            
            # 1. Training Phase
            train_loss, train_acc, epoch_time = self._train_epoch()
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            formatted_time = self._format_time(epoch_time)
            print(f"[Train] Time: {formatted_time} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
            
            # 2. Validation Phase (Utilizando MetricTracker y evaluate)
            val_acc = self.evaluate(self.val_loader, src_classes, "Val")
            self.history['val_loss'].append(0.0) # Se puede omitir o calcular si necesitas plotearlo
            self.history['val_acc'].append(val_acc)
            
            # 3. Learning Rate Scheduling
            if scheduler is not None:
                scheduler.step()
                
            # 4. Checkpoint saving
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_weights = copy.deepcopy(self.model.state_dict())
                print(f"New best model found! (Val Acc: {self.best_val_acc:.4f})")

        total_time = time.time() - total_train_start
        print(f"\n{' TRAINING COMPLETE ':=^50}")
        print(f"Total Duration: {self._format_time(total_time)}")
        print(f"Best Source Accuracy: {self.best_val_acc:.4f}")
        print("="*50)
        
        # Restore the best weights before returning
        if self.best_model_weights is not None:
            self.model.load_state_dict(self.best_model_weights)
            
        return self.model, self.history

    def _train_epoch(self) -> Tuple[float, float, float]:
        """
        Runs a single epoch of training.
        Returns: Average loss, Average Accuracy, Epoch time in seconds.
        """
        self.model.train()
        running_loss: float = 0.0
        
        # Para el cálculo de accuracy con MetricTracker
        all_preds = []
        all_labels = []
        
        epoch_start = time.time()
        pbar = tqdm(self.train_loader, desc="Training (Baseline)", leave=False)
        
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
            
            # Guardamos predicciones para MetricTracker
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.detach())
            all_labels.append(labels.detach())
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        epoch_time = time.time() - epoch_start
        epoch_loss: float = running_loss / len(self.train_loader.dataset)
        
        # Calculamos el accuracy global con MetricTracker
        num_classes = len(self.train_loader.dataset.classes)
        overall_acc, _ = MetricTracker.compute_accuracy(
            torch.cat(all_preds), torch.cat(all_labels), num_classes
        )
        
        return epoch_loss, overall_acc, epoch_time

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, class_names: List[str], prefix: str = "Val") -> float:
        """
        Runs validation and prints the summary using MetricTracker.
        Returns the overall accuracy.
        """
        self.model.eval()
        all_preds, all_labels = [], []
        
        pbar = tqdm(loader, desc=f"Evaluating ({prefix})", leave=False)
        
        for batch in pbar:
            imgs: torch.Tensor = batch[0].to(self.device)
            labels: torch.Tensor = batch[1].to(self.device)
            
            # Forward pass
            logits: torch.Tensor = self.model(imgs, mode='class')
            
            # Guardamos predicciones
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds)
            all_labels.append(labels)

        # Usar MetricTracker
        overall_acc, per_class_acc = MetricTracker.compute_accuracy(
            torch.cat(all_preds), torch.cat(all_labels), len(class_names)
        )
        
        MetricTracker.print_summary(prefix, overall_acc, per_class_acc, class_names)
        
        return overall_acc
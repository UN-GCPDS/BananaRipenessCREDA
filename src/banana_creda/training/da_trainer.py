import torch
import time
import copy
import math
from itertools import cycle
from collections import defaultdict
from torch.amp import autocast, GradScaler
from typing import Dict, List, Optional
from tqdm import tqdm # <-- Importamos tqdm

from banana_creda.utils.metrics import MetricTracker
from banana_creda.config import TrainConfig

class BananaTrainer:
    """Trainer class for Domain Adaptation using the CREDA algorithm.

    Handles the training loop with dual-domain batching, uncertainty-weighted 
    alignment, and automated mixed precision (AMP).

    Attributes:
        model (nn.Module): The neural network model.
        source_loaders (Dict[str, DataLoader]): DataLoaders for the source domain.
        target_loaders (Dict[str, DataLoader]): DataLoaders for the target domain.
        criterion (nn.Module): Loss function (usually CREDALoss).
        optimizer (optim.Optimizer): Optimization algorithm.
        config (TrainConfig): Configuration object with training hyperparameters.
        device (torch.device): Computing device (cpu/cuda/mps).
        use_amp (bool): Whether to use Automatic Mixed Precision.
        scaler (GradScaler): Gradient scaler for AMP.
        best_acc (float): Highest target validation accuracy achieved.
        best_model_wts (Dict[str, torch.Tensor]): State dict of the best model.
    """
    def __init__(
        self, 
        model: nn.Module, 
        source_loaders: Dict[str, DataLoader], 
        target_loaders: Dict[str, DataLoader], 
        criterion: nn.Module, 
        optimizer: optim.Optimizer, 
        config: TrainConfig
    ):
        """Initializes the BananaTrainer.

        Args:
            model (nn.Module): The model to train.
            source_loaders (Dict[str, DataLoader]): Source domain data loaders.
            target_loaders (Dict[str, DataLoader]): Target domain data loaders.
            criterion (nn.Module): Loss criterion.
            optimizer (optim.Optimizer): Optimizer for parameter updates.
            config (TrainConfig): Training configuration.
        """
        self.model = model
        self.source_loaders = source_loaders
        self.target_loaders = target_loaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = torch.device(config.device)
        
        # AMP setup
        self.use_amp = config.use_amp and (self.device.type == 'cuda')
        self.scaler = GradScaler(enabled=self.use_amp)
        
        # Warm-up logic initialization
        self.criterion.lambda_creda = 0.0 if config.warmup else config.lambda_creda
        
        if config.warmup:
            print(f"Warm-up enabled (Threshold: {config.warmup_threshold})")
            
        self.best_acc = 0.0
        self.best_model_wts = copy.deepcopy(model.state_dict())

    def _format_time(self, seconds: float) -> str:
        """Converts seconds into a readable MM:SS format.

        Args:
            seconds (float): Total seconds to format.

        Returns:
            str: Formatted time string (MM:SS).
        """
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"

    def train_epoch(self) -> Dict[str, float]:
        """Runs a single epoch of Domain Adaptation training.

        Returns:
            Dict[str, float]: Dictionary containing average losses and epoch time.
        """
        self.model.train()
        running_losses = defaultdict(float)
        total_samples = 0
        
        len_s, len_t = len(self.source_loaders['train']), len(self.target_loaders['train'])
        num_batches = max(len_s, len_t)
        
        src_iter = iter(self.source_loaders['train']) if len_s >= len_t else cycle(self.source_loaders['train'])
        tgt_iter = cycle(self.target_loaders['train']) if len_s >= len_t else iter(self.target_loaders['train'])

        epoch_start = time.time()
        
        # tqdm progress bar for the training loop
        pbar = tqdm(range(num_batches), desc="Training (DA)", leave=False)
        
        for _ in pbar:
            src_batch = next(src_iter)
            img_s, lbl_s = src_batch[:2]

            tgt_batch = next(tgt_iter)
            img_t = tgt_batch[0]
            
            img_s, lbl_s, img_t = img_s.to(self.device), lbl_s.to(self.device), img_t.to(self.device)
            self.optimizer.zero_grad()

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                feat_s = self.model(img_s, mode='feature')
                logit_s = self.model(img_s, mode='class')
                feat_t = self.model(img_t, mode='feature')
                logit_t = self.model(img_t, mode='class')
                loss, loss_dict = self.criterion(feat_s, logit_s, lbl_s, feat_t, logit_t)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = img_s.size(0)
            total_samples += batch_size
            for k, v in loss_dict.items():
                running_losses[k] += v * batch_size

            # Update progress bar with the current total loss
            current_loss = loss_dict.get('total_loss', loss.item())
            pbar.set_postfix({'loss': f"{current_loss:.4f}"})

        metrics = {k: v / total_samples for k, v in running_losses.items()}
        metrics['epoch_time'] = time.time() - epoch_start
        return metrics

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, class_names: List[str], prefix: str = "Val") -> float:
        """Evaluates the model on a given dataset.

        Args:
            loader (DataLoader): The dataset loader to evaluate.
            class_names (List[str]): List of human-readable class names.
            prefix (str): Label for the evaluation phase.

        Returns:
            float: Overall classification accuracy.
        """
        self.model.eval()
        all_preds, all_labels = [], []
        
        # tqdm progress bar for the evaluation loop
        pbar = tqdm(loader, desc=f"Evaluating ({prefix})", leave=False)
        
        for batch in pbar:
            imgs, labels = batch[:2]
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            logits = self.model(imgs, mode='class')
            all_preds.append(torch.max(logits, 1)[1])
            all_labels.append(labels)

        overall_acc, per_class_acc = MetricTracker.compute_accuracy(
            torch.cat(all_preds), torch.cat(all_labels), len(class_names)
        )
        
        MetricTracker.print_summary(prefix, overall_acc, per_class_acc, class_names)
        return overall_acc

    def fit(self, scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None) -> torch.nn.Module:
        """Executes the full Domain Adaptation training process.

        Args:
            scheduler (Optional[LRScheduler]): Learning rate scheduler.

        Returns:
            nn.Module: The model with the best target validation weights.
        """
        src_classes = self.source_loaders['train'].dataset.classes
        warmup_done = not self.config.warmup
        warmup_end_epoch = 0 if warmup_done else -1
        total_train_start = time.time()

        print(f"Initial Lambda CREDA loss value: {self.criterion.lambda_creda}")
        print(f"Config Lambda CREDA value: {self.config.lambda_creda}")

        for epoch in range(self.config.epochs):
            lr_current = self.optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch+1}/{self.config.epochs} | LR: {lr_current:.6f}")
            print("-" * 40)

            # 1. Train
            train_metrics = self.train_epoch()
            formatted_time = self._format_time(train_metrics['epoch_time'])
            
            loss_total = train_metrics.get('total_loss', 0.0)
            loss_cls = train_metrics.get('loss_cls', 0.0)
            loss_creda = train_metrics.get('loss_creda', 0.0)
            
            print(f"[Train] Time: {formatted_time} | Loss: {loss_total:.4f} | Cls loss: {loss_cls:.4f} | CREDA loss: {loss_creda:.4f}")

            # 2. Evaluate
            _ = self.evaluate(self.source_loaders['validation'], src_classes, "Src Val")
            val_acc_tgt = self.evaluate(self.target_loaders['validation'], src_classes, "Tgt Val")

            # 3. Checkpoint Logic
            if (not self.config.warmup or warmup_done) and val_acc_tgt > self.best_acc:
                self.best_acc = val_acc_tgt
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                print(f"New best model found! (Target Val Acc: {self.best_acc:.4f})")

            # 4. Warm-up Logic
            if self.config.warmup and not warmup_done:
                if epoch >= self.config.warmup_epochs or val_acc_tgt >= self.config.warmup_threshold: 
                    warmup_done = True
                    warmup_end_epoch = epoch
                    print(f"\n Warm-up completed at epoch {epoch+1}: Domain Alignment Activated.")

            # 5. Exponential Lambda Adjustment
            if warmup_done:
                remaining_epochs = self.config.epochs - warmup_end_epoch
                p = (epoch + 1 - warmup_end_epoch) / remaining_epochs if remaining_epochs > 0 else 1.0
                
                p = max(0.0, min(1.0, p))
                gamma = 2.0
                alpha = 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0
                
                self.criterion.lambda_creda = self.config.lambda_creda * alpha
                print(f"New CREDA lambda = {self.criterion.lambda_creda:.4f}")

            if scheduler is not None: 
                scheduler.step()

        total_time = time.time() - total_train_start
        print(f"\n{' TRAINING COMPLETE ':=^50}")
        print(f"Total Duration: {self._format_time(total_time)}")
        print(f"Best Target Accuracy: {self.best_acc:.4f}")
        print("="*50)

        # Restore best weights before returning
        self.model.load_state_dict(self.best_model_wts)
        return self.model
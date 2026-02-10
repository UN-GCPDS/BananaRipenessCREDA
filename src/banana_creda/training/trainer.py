import torch
import time
import copy
from collections import defaultdict
from torch.amp import autocast, GradScaler
from typing import Dict, Any, List
from banana_creda.utils.metrics import MetricTracker
from banana_creda.config import TrainConfig

class BananaTrainer:
    def __init__(self, model, source_loaders, target_loaders, criterion, optimizer, config: TrainConfig):
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
        
        # Warm-up logic inyectada desde config
        self.criterion.lambda_creda = 0.0 if config.warmup else config.lambda_creda
        self.criterion.lambda_entropy = 0.0 if config.warmup else config.lambda_entropy 
        self.history = defaultdict(list)
        self.best_acc = 0.0
        self.best_model_wts = copy.deepcopy(model.state_dict())

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        running_losses = defaultdict(float)
        total_samples = 0
        
        from itertools import cycle # Import local para evitar overhead
        len_s, len_t = len(self.source_loaders['train']), len(self.target_loaders['train'])
        num_batches = max(len_s, len_t)
        
        src_iter = iter(self.source_loaders['train']) if len_s >= len_t else cycle(self.source_loaders['train'])
        tgt_iter = cycle(self.target_loaders['train']) if len_s >= len_t else iter(self.target_loaders['train'])

        start_time = time.time()
        for _ in range(num_batches):
            img_s, lbl_s = next(src_iter)
            img_t, _ = next(tgt_iter)
            
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

        metrics = {k: v / total_samples for k, v in running_losses.items()}
        metrics['time'] = time.time() - start_time
        return metrics

    def evaluate(self, loader, class_names, prefix="Val"):
        self.model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                logits = self.model(imgs, mode='class')
                all_preds.append(torch.max(logits, 1)[1])
                all_labels.append(labels)

        overall_acc, per_class_acc = MetricTracker.compute_accuracy(
            torch.cat(all_preds), torch.cat(all_labels), len(class_names), self.device
        )
        MetricTracker.print_report(prefix, overall_acc, per_class_acc, class_names)
        return overall_acc

    def fit(self, scheduler=None):
        src_classes = self.source_loaders['train'].dataset.classes
        
        for epoch in range(self.config.epochs):
            print(f"\nEpoch {epoch+1}/{self.config.epochs} | LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            train_metrics = self.train_epoch()
            print(f"[Train] Loss: {train_metrics['total_loss']:.4f} | CREDA: {train_metrics['loss_creda']:.4f}")

            val_acc_src = self.evaluate(self.source_loaders['validation'], src_classes, "Src Val")
            val_acc_tgt = self.evaluate(self.target_loaders['validation'], src_classes, "Tgt Val")

            # Warm-up condition (Basado en tu lógica)
            if self.config.warmup:
                if epoch >= self.config.warmup_epochs and val_acc_tgt >= self.config.warmup_threshold: 
                    self.criterion.config.lambda_creda = self.config.lambda_creda
                    self.criterion.config.lambda_entropy = self.config.lambda_entropy

            if val_acc_tgt > self.best_acc:
                self.best_acc = val_acc_tgt
                self.best_model_wts = copy.deepcopy(self.model.state_dict())

            if scheduler: scheduler.step()

        self.model.load_state_dict(self.best_model_wts)
        return self.model
import torch
import numpy as np
from typing import List, Tuple

class MetricTracker:
    """Calcula y almacena métricas de rendimiento por clase y globales."""
    
    @staticmethod
    def compute_accuracy(preds: torch.Tensor, labels: torch.Tensor, num_classes: int, device: torch.device) -> Tuple[float, List[float]]:
        confusion_matrix = torch.zeros(num_classes, num_classes, device=device)
        for t, p in zip(labels.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
            
        per_class_acc = []
        for i in range(num_classes):
            tp = confusion_matrix[i, i].item()
            total = confusion_matrix[i, :].sum().item()
            per_class_acc.append(tp / total if total > 0 else 0.0)
            
        total_correct = confusion_matrix.diag().sum().item()
        total_samples = confusion_matrix.sum().item()
        overall_acc = total_correct / total_samples if total_samples > 0 else 0.0
        
        return overall_acc, per_class_acc

    @staticmethod
    def print_report(prefix: str, overall_acc: float, per_class_acc: List[float], class_names: List[str]):
        print(f"\n[{prefix}] Overall Acc: {overall_acc:.4f}")
        print(f"  > {'Class':<20} | {'Accuracy':<10}")
        print(f"  {'-'*35}")
        for i, acc in enumerate(per_class_acc):
            name = class_names[i] if i < len(class_names) else f"Class {i}"
            print(f"    {name:<20} | {acc:.4f}")
        print("-" * 50)
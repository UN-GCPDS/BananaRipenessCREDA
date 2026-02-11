import torch
from typing import List, Dict, Tuple, Optional

class MetricTracker:
    @staticmethod
    def compute_accuracy(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> Tuple[float, List[float]]:
        """
        FAST PATH: Optimized for the training loop.
        Computes global and per-class accuracy using boolean masks.
        """
        # Global accuracy (Pure vector operation)
        overall_acc = (preds == labels).sum().item() / labels.numel()
        
        # Per-class accuracy
        per_class_acc = []
        for i in range(num_classes):
            mask = (labels == i)
            total = mask.sum().item()
            if total > 0:
                # Solo contamos aciertos donde la etiqueta real era 'i'
                correct = (preds[mask] == i).sum().item()
                per_class_acc.append(correct / total)
            else:
                per_class_acc.append(0.0)
                
        return overall_acc, per_class_acc

    @staticmethod
    def compute_full_metrics(preds: torch.Tensor, labels: torch.Tensor, num_classes: int, device: torch.device) -> Dict:
        """
        FULL PATH: For final scientific reports.
        Calculates Precision, Recall, F1 and Support.
        """
        conf_matrix = torch.zeros(num_classes, num_classes, device=device)
        for t, p in zip(labels.view(-1), preds.view(-1)):
            conf_matrix[t.long(), p.long()] += 1
            
        total_samples = conf_matrix.sum().item()
        results = {"per_class": [], "overall_acc": 0.0}
        
        for i in range(num_classes):
            tp = conf_matrix[i, i].item()
            fp = conf_matrix[:, i].sum().item() - tp
            fn = conf_matrix[i, :].sum().item() - tp
            support = int(conf_matrix[i, :].sum().item())
            
            # Advanced metrics
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            
            results["per_class"].append({
                "precision": prec, "recall": rec, "f1": f1, "support": support,
                "accuracy": tp / support if support > 0 else 0.0
            })
            
        results["overall_acc"] = conf_matrix.diag().sum().item() / total_samples
        return results

    @staticmethod
    def print_summary(prefix: str, overall_acc: float, per_class_acc: List[float], class_names: List[str]):
        """Light summary for the Trainer."""
        print(f"\n[{prefix}] Overall Acc: {overall_acc:.4f}")
        print(f"  > {'Class':<15} | {'Acc':<8}")
        print(f"  {'-'*27}")
        for i, acc in enumerate(per_class_acc):
            name = class_names[i] if i < len(class_names) else f"C{i}"
            print(f"    {name:<15} | {acc:.4f}")

    @staticmethod
    def print_full_report(prefix: str, metrics: Dict, class_names: List[str]):
        """Full report for the final experiment."""
        print(f"\n{' REPORT: ' + prefix + ' ':=^75}")
        print(f"{'Class':<20} | {'Prec.':<8} | {'Recall':<8} | {'F1-Score':<8} | {'Support':<8}")
        print("-" * 75)
        for i, m in enumerate(metrics["per_class"]):
            name = class_names[i] if i < len(class_names) else f"C{i}"
            print(f"{name:<20} | {m['precision']:<8.4f} | {m['recall']:<8.4f} | {m['f1']:<8.4f} | {m['support']:<8}")
        print("-" * 75)
        print(f"OVERALL ACCURACY: {metrics['overall_acc']:.4f}")
        print("=" * 75)
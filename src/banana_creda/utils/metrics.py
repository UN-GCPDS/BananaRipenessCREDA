import torch
from typing import List, Dict, Tuple, Any

class MetricTracker:
    """Utility class for computing and reporting classification metrics.

    Provides optimized methods for training-time accuracy calculation and 
    comprehensive reporting for final evaluation.
    """
    @staticmethod
    def compute_accuracy(
        preds: torch.Tensor, 
        labels: torch.Tensor, 
        num_classes: int
    ) -> Tuple[float, List[float]]:
        """Computes global and per-class accuracy using boolean masks.

        This method is optimized for use within training loops.

        Args:
            preds (torch.Tensor): Predicted class labels (integers).
            labels (torch.Tensor): Ground truth labels (integers).
            num_classes (int): Total number of classes.

        Returns:
            Tuple[float, List[float]]: 
                Global accuracy and a list of accuracies for each class.
        """
        # Global accuracy (Pure vector operation)
        overall_acc = (preds == labels).sum().item() / labels.numel()
        
        # Per-class accuracy
        per_class_acc = []
        for i in range(num_classes):
            mask = (labels == i)
            total = mask.sum().item()
            if total > 0:
                # Count correct predictions only where the true label was 'i'
                correct = (preds[mask] == i).sum().item()
                per_class_acc.append(correct / total)
            else:
                per_class_acc.append(0.0)
                
        return overall_acc, per_class_acc

    @staticmethod
    def compute_full_metrics(
        preds: torch.Tensor, 
        labels: torch.Tensor, 
        num_classes: int, 
        device: torch.device
    ) -> Dict[str, Any]:
        """Calculates comprehensive classification metrics including F1-Score.

        Args:
            preds (torch.Tensor): Predicted class labels.
            labels (torch.Tensor): Ground truth labels.
            num_classes (int): Total number of classes.
            device (torch.device): Device where the tensors are located.

        Returns:
            Dict[str, Any]: Dictionary containing per-class metrics and macro averages.
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
            
            # Precision, Recall and F1 calculation
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            
            results["per_class"].append({
                "precision": prec, "recall": rec, "f1": f1, "support": support,
                "accuracy": tp / support if support > 0 else 0.0
            })
            
        results["overall_acc"] = conf_matrix.diag().sum().item() / total_samples
        
        # Macro averages
        results["macro_avg"] = {
            "precision": sum(c["precision"] for c in results["per_class"]) / num_classes,
            "recall": sum(c["recall"] for c in results["per_class"]) / num_classes,
            "f1": sum(c["f1"] for c in results["per_class"]) / num_classes,
            "accuracy": sum(c["accuracy"] for c in results["per_class"]) / num_classes
        }
        
        return results

    @staticmethod
    def print_summary(
        prefix: str, 
        overall_acc: float, 
        per_class_acc: List[float], 
        class_names: List[str]
    ) -> None:
        """Prints a concise summary intended for the training logs.

        Args:
            prefix (str): Label for the evaluation phase (e.g., "Val", "Source").
            overall_acc (float): Global accuracy value.
            per_class_acc (List[float]): List of accuracies for each class.
            class_names (List[str]): Map of class indices to names.
        """
        print(f"\n[{prefix}] Overall Acc: {overall_acc:.4f}")
        print(f"  > {'Class':<15} | {'Acc':<8}")
        print(f"  {'-'*27}")
        for i, acc in enumerate(per_class_acc):
            name = class_names[i] if i < len(class_names) else f"C{i}"
            print(f"    {name:<15} | {acc:.4f}")

    @staticmethod
    def print_full_report(prefix: str, metrics: Dict[str, Any], class_names: List[str]) -> None:
        """Prints a detailed classification report for scientific evaluation.

        Args:
            prefix (str): Title for the report section.
            metrics (Dict[str, Any]): Dictionary of metrics from compute_full_metrics.
            class_names (List[str]): List of human-readable class names.
        """
        print(f"\n{' REPORT: ' + prefix + ' ':=^85}")
        header = f"{'Class':<20} | {'Prec.':<8} | {'Recall':<8} | {'F1-Score':<8} | {'Acc.':<8} | {'Support':<8}"
        print(header)
        print("-" * len(header))
        
        for i, m in enumerate(metrics["per_class"]):
            name = class_names[i] if i < len(class_names) else f"C{i}"
            print(f"{name:<20} | {m['precision']:<8.4f} | {m['recall']:<8.4f} | {m['f1']:<8.4f} | {m['accuracy']:<8.4f} | {m['support']:<8}")
            
        print("-" * len(header))
        
        # Macro Averages row
        if "macro_avg" in metrics:
            m = metrics["macro_avg"]
            print(f"{'Macro Avg':<20} | {m['precision']:<8.4f} | {m['recall']:<8.4f} | {m['f1']:<8.4f} | {m['accuracy']:<8.4f} | {'-':<8}")
            print("-" * len(header))
            
        print(f"OVERALL ACCURACY: {metrics['overall_acc']:.4f}")
        print("=" * len(header))

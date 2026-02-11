import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pathlib import Path
from typing import List, Tuple, Optional
from itertools import cycle

class BananaVisualizer:
    """
    Advanced visualization engine for Computer Vision and Domain Adaptation.
    Generates classification reports, domain alignment, and ROC curves.
    """
    
    def __init__(self, device: torch.device, output_dir: str = "outputs"):
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Parameters to revert ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def _denormalize(self, img_tensor: torch.Tensor) -> np.ndarray:
        """Converts a normalized tensor to RGB image [0, 1]."""
        img = img_tensor.cpu().numpy().transpose(1, 2, 0)
        img = img * self.std + self.mean
        return np.clip(img, 0, 1)

    def _get_inference_data(self, model, loader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
        """Obtains labels, predictions, probabilities, features, and images."""
        model.eval()
        all_feats, all_labels, all_preds, all_probs, all_imgs = [], [], [], [], []
        
        with torch.no_grad():
            for imgs, labels in loader:
                all_imgs.append(imgs.clone())
                imgs = imgs.to(self.device)
                
                feats = model(imgs, mode='feature')
                logits = model(imgs, mode='class')
                
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_feats.append(feats.cpu().numpy())
                all_labels.append(labels.numpy())
                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
        
        return (
            np.concatenate(all_labels),
            np.concatenate(all_preds),
            np.concatenate(all_probs),
            np.concatenate(all_feats),
            torch.cat(all_imgs)
        )

    def plot_confusion_matrix(self, model, loader, class_names: List[str], prefix: str):
        """Generates and saves the normalized confusion matrix."""
        y_true, y_pred, _, _, _ = self._get_inference_data(model, loader)
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        acc = np.trace(cm) / np.sum(cm)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        
        plt.title(f'Confusion Matrix: {prefix}\nGlobal Accuracy: {acc:.2%}', fontsize=12)
        plt.xlabel('Prediction')
        plt.ylabel('True Label')
        
        path = self.output_dir / f"{prefix}_confusion_matrix.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion Matrix saved to: {path}")

    def plot_roc_curve(self, model, loader, class_names: List[str], prefix: str):
        """Generates the multiclass ROC curve (One-vs-Rest)."""
        y_true, _, y_probs, _, _ = self._get_inference_data(model, loader)
        n_classes = len(class_names)
        
        # Binarize labels for multiclass calculation
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        fpr, tpr, roc_auc = {}, {}, {}
        
        # Calculate curves per class
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        #  Micro-average ROC
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Plotting
        plt.figure(figsize=(10, 8))
        plt.plot(fpr["micro"], tpr["micro"],
                 label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.2f})',
                 color='deeppink', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:0.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic - {prefix}')
        plt.legend(loc="lower right")
        
        path = self.output_dir / f"{prefix}_roc_curve.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC Curve saved to: {path}")

    def plot_umap(self, model, source_loader, target_loader, prefix: str):
        """Visualize domain alignment."""
        _, _, _, feat_s, _ = self._get_inference_data(model, source_loader)
        _, _, _, feat_t, _ = self._get_inference_data(model, target_loader)
        
        features = np.concatenate([feat_s, feat_t])
        domains = np.concatenate([np.zeros(len(feat_s)), np.ones(len(feat_t))])
        
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric='cosine', random_state=42)
        embedding = reducer.fit_transform(features)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=domains, cmap='coolwarm', s=10, alpha=0.5)
        plt.legend(handles=scatter.legend_elements()[0], labels=['Source', 'Target'])
        plt.title(f"Domain Alignment - {prefix}")
        
        path = self.output_dir / f"{prefix}_umap_alignment.png"
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"UMAP of Domains saved to: {path}")

    def plot_umap_with_images(self, model, loader, class_names, prefix, min_dist_plots=0.1, image_zoom=0.07):
        """Generates a UMAP with real images superimposed."""
        y_true, _, _, features, images_tensor = self._get_inference_data(model, loader)
        
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric='cosine', random_state=42)
        embedding = reducer.fit_transform(features)
        
        plt.figure(figsize=(12, 10))
        cmap = plt.get_cmap('tab10')
        plt.scatter(embedding[:, 0], embedding[:, 1], c=y_true, cmap=cmap, s=15, alpha=0.2)
        
        shown_positions = np.array([[1000.0, 1000.0]]) 
        ax = plt.gca()
        
        for i in range(len(embedding)):
            curr_pos = embedding[i]
            dist = np.sum((curr_pos - shown_positions) ** 2, axis=1)
            
            if np.min(dist) > min_dist_plots:
                shown_positions = np.r_[shown_positions, [curr_pos]]
                img_rgb = self._denormalize(images_tensor[i])
                imagebox = OffsetImage(img_rgb, zoom=image_zoom)
                ab = AnnotationBbox(imagebox, curr_pos, bboxprops={"edgecolor": cmap(y_true[i]), "lw": 1.5})
                ax.add_artist(ab)

        plt.title(f"Qualitative Latent Space - {prefix}")
        plt.axis("off")
        path = self.output_dir / f"{prefix}_umap_samples.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"UMAP with images saved to: {path}")
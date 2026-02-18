import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap

from pathlib import Path
from typing import List, Tuple, Optional
from itertools import cycle

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.lines import Line2D

import seaborn as sns
from scipy.stats import pearsonr

class BananaVisualizer:
    """
    Advanced visualization engine for Computer Vision and Domain Adaptation.
    """

    def __init__(self, device: torch.device, output_dir: str = "outputs"):
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    # =========================================================
    # Utilities
    # =========================================================

    def _denormalize(self, img_tensor: torch.Tensor) -> np.ndarray:
        img = img_tensor.cpu().numpy().transpose(1, 2, 0)
        img = img * self.std + self.mean
        return np.clip(img, 0, 1)

    def _get_inference_data(
        self,
        model,
        loader,
        return_lime_variant: bool = False,
        domain_id: Optional[int] = None
    ):
        model.eval()

        all_feats, all_labels, all_preds, all_probs = [], [], [], []
        all_imgs, all_lime = [], []

        with torch.no_grad():
            for batch in loader:

                imgs, labels = batch[:2]
                lime_variant = batch[2] if (return_lime_variant and len(batch) > 2) else None

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

                if return_lime_variant:
                    all_lime.append(np.array(lime_variant))

        output = [
            np.concatenate(all_labels),
            np.concatenate(all_preds),
            np.concatenate(all_probs),
            np.concatenate(all_feats),
            torch.cat(all_imgs)
        ]

        if return_lime_variant:
            output.append(np.concatenate(all_lime))

        if domain_id is not None:
            output.append(np.full(len(output[0]), domain_id))

        return tuple(output)

    # =========================================================
    # Confusion Matrix
    # =========================================================

    def plot_confusion_matrix(self, model, loader, class_names: List[str], prefix: str):
        y_true, y_pred, _, _, _ = self._get_inference_data(model, loader)

        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        acc = np.trace(cm) / np.sum(cm)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)

        plt.title(f'Confusion Matrix: {prefix}\nGlobal Accuracy: {acc:.2%}')
        plt.xlabel('Prediction')
        plt.ylabel('True Label')

        path = self.output_dir / f"{prefix}_confusion_matrix.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

    # =========================================================
    # ROC Curve
    # =========================================================

    def plot_roc_curve(self, model, loader, class_names: List[str], prefix: str):
        y_true, _, y_probs, _, _ = self._get_inference_data(model, loader)
        n_classes = len(class_names)

        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        fpr, tpr, roc_auc = {}, {}, {}

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        plt.figure(figsize=(10, 8))
        plt.plot(fpr["micro"], tpr["micro"], linestyle=':', linewidth=4,
                 label=f'micro-average ROC (area = {roc_auc["micro"]:0.2f})')

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'{class_names[i]} (AUC = {roc_auc[i]:0.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()

        path = self.output_dir / f"{prefix}_roc_curve.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

    # =========================================================
    # UMAP with Images
    # =========================================================

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

    def plot_umap_with_images(
        self,
        model,
        source_loader,
        target_loader,
        class_names,
        prefix,
        image_zoom=0.07,
        min_dist_plots=0.15,
        use_lime=False
    ):

        y_s, _, _, feat_s, imgs_s, lime_s, dom_s = self._get_inference_data(
            model,
            source_loader,
            return_lime_variant=use_lime,
            domain_id= 0 if use_lime else None
        )

        y_t, _, _, feat_t, imgs_t, dom_t = self._get_inference_data(
            model,
            target_loader,
            return_lime_variant=False,
            domain_id=1 if use_lime else None
        )

        features = np.concatenate([feat_s, feat_t])
        labels = np.concatenate([y_s, y_t])
        domains = np.concatenate([dom_s, dom_t])

        images_tensor = torch.cat([imgs_s, imgs_t])

        if use_lime:
            lime_variants = np.concatenate([lime_s, np.full(len(y_t), -1)])

        reducer = umap.UMAP(
            n_neighbors=30,
            min_dist=0.3,
            metric='cosine',
            random_state=42
        )

        embedding = reducer.fit_transform(features)

        plt.figure(figsize=(16, 12))
        ax = plt.gca()
        cmap = plt.get_cmap('tab10')

        markers = {0: 'o', 1: '^'}

        for domain in [0, 1]:
            idx = domains == domain

            plt.scatter(
                embedding[idx, 0],
                embedding[idx, 1],
                c=labels[idx],
                cmap=cmap,
                marker=markers[domain],
                s=20,
                alpha=0.25
            )

        # =====================================================
        # Plot representative images
        # =====================================================

        shown_positions = np.array([[1000.0, 1000.0]])

        for i in range(len(embedding)):

            curr_pos = embedding[i]
            dist = np.sum((curr_pos - shown_positions) ** 2, axis=1)

            if np.min(dist) > min_dist_plots:

                shown_positions = np.r_[shown_positions, [curr_pos]]

                img_rgb = self._denormalize(images_tensor[i])
                imagebox = OffsetImage(img_rgb, zoom=image_zoom)

                ab = AnnotationBbox(
                    imagebox,
                    curr_pos,
                    bboxprops=dict(
                        edgecolor=cmap(labels[i]),
                        linewidth=2
                    )
                )

                ax.add_artist(ab)

        # =====================================================
        # LIME Structure + Pearson Correlation
        # =====================================================

        if use_lime:

            unique_lime = np.unique(lime_s)

            for variant in unique_lime:

                idx = (lime_variants == variant) & (domains == 0)

                if np.sum(idx) < 4:
                    continue

                points = embedding[idx]

                try:
                    sns.kdeplot(
                        x=points[:, 0],
                        y=points[:, 1],
                        ax=ax,
                        levels=4,          # Paper-friendly smoothness
                        linewidths=1.5,
                        alpha=0.6,
                        fill=False,         # IMPORTANT → cleaner for papers
                        color=cmap(int(variant))
                    )

                except Exception:
                    continue

        # -------------------------------------------------
        # Pearson Correlation (LIME vs UMAP Axes)
        # -------------------------------------------------

        src_idx = domains == 0

        lime_values = lime_s.astype(float)
        umap_x = embedding[src_idx, 0]
        umap_y = embedding[src_idx, 1]

        try:
            corr_x, p_x = pearsonr(lime_values, umap_x)
            corr_y, p_y = pearsonr(lime_values, umap_y)

            print("\nPearson Correlation (LIME vs UMAP):")
            print(f"UMAP-x → r = {corr_x:.4f} | p = {p_x:.4e}")
            print(f"UMAP-y → r = {corr_y:.4f} | p = {p_y:.4e}")

        except Exception:
            print("Pearson correlation could not be computed.")

        # =====================================================
        # Legends
        # =====================================================

        domain_legend = [
            Line2D([0], [0], marker='o', color='w',
                label='Source', markerfacecolor='gray', markersize=10),
            Line2D([0], [0], marker='^', color='w',
                label='Target', markerfacecolor='gray', markersize=10)
        ]

        class_legend = [
            Line2D([0], [0], marker='s', color='w',
                label=class_names[i],
                markerfacecolor=cmap(i),
                markersize=10)
            for i in range(len(class_names))
        ]

        ax.legend(handles=domain_legend + class_legend, loc='upper right')

        plt.title("Latent Space Structure (Domain • Class" + (" • LIME" if use_lime else "") + ")")
        plt.axis("off")

        path = self.output_dir / f"{prefix}_umap_samples.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved UMAP plot to {path}")



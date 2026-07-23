import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap

from pathlib import Path
from typing import List, Tuple, Optional, Any   
from itertools import cycle

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.lines import Line2D

import seaborn as sns
from scipy.stats import pearsonr

class BananaVisualizer:
    """Advanced visualization engine for Computer Vision and Domain Adaptation.

    Provides tools for performance analysis, latent space exploration (UMAP), 
    and distribution alignment assessment.

    Attributes:
        device (torch.device): Computing device for inference.
        output_dir (Path): Directory where plots are saved.
        mean (np.ndarray): Normalization mean for image denormalization.
        std (np.ndarray): Normalization standard deviation for image denormalization.
    """

    def __init__(self, device: torch.device, output_dir: str = "outputs"):
        """Initializes the BananaVisualizer.

        Args:
            device (torch.device): Device to use for model inference.
            output_dir (str): Path to the directory where results will be stored.
        """
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Standard ImageNet normalization constants
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    # =========================================================
    # Utilities
    # =========================================================

    def _denormalize(self, img_tensor: torch.Tensor) -> np.ndarray:
        """Converts a normalized PyTorch tensor back into a displayable RGB image.

        Args:
            img_tensor (torch.Tensor): Normalized image tensor (C, H, W).

        Returns:
            np.ndarray: Denormalized RGB image (H, W, C) clipped to [0, 1].
        """
        img = img_tensor.cpu().numpy().transpose(1, 2, 0)
        img = img * self.std + self.mean
        return np.clip(img, 0, 1)

    def _get_inference_data(
        self,
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        return_lime_variant: bool = False,
        domain_id: Optional[int] = None,
        save_results: bool = False
    ) -> Tuple[Any, ...]:
        """Runs inference on a dataset and extracts features, labels, and predictions.

        Args:
            model (nn.Module): The model to use for inference.
            loader (DataLoader): The dataset loader.
            return_lime_variant (bool): Whether to extract LIME/lighting variation values.
            domain_id (Optional[int]): If provided, tags every sample with this domain ID.
            save_results (bool): Whether to save classification results to a .npy file.

        Returns:
            Tuple[Any, ...]: A tuple containing (labels, preds, probs, features, images)
                and optionally (lime_variants, domain_ids).
        """
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

        if save_results:
            y_true, y_pred = np.concatenate(all_labels), np.concatenate(all_preds)
            correct = (y_true == y_pred)
            save_path = self.output_dir / "sample_results.npy"
            np.save(save_path, correct.astype(np.int8))
            print(f"Sample results saved to: {save_path}")

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
    # Performance Monitoring
    # =========================================================

    def plot_confusion_matrix(
        self, 
        model: torch.nn.Module, 
        loader: torch.utils.data.DataLoader, 
        class_names: List[str], 
        prefix: str
    ) -> None:
        """Generates and saves a normalized confusion matrix heatmap.

        Args:
            model (nn.Module): The model to evaluate.
            loader (DataLoader): The dataset loader.
            class_names (List[str]): List of class names for axes labeling.
            prefix (str): Filename prefix for the saved plot.
        """
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

    def plot_roc_curve(
        self, 
        model: torch.nn.Module, 
        loader: torch.utils.data.DataLoader, 
        class_names: List[str], 
        prefix: str
    ) -> None:
        """Computes and plots Receiver Operating Characteristic (ROC) curves.

        Args:
            model (nn.Module): The model to evaluate.
            loader (DataLoader): The dataset loader.
            class_names (List[str]): List of class names.
            prefix (str): Filename prefix for the saved plot.
        """
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
    # Latent Space Analysis
    # =========================================================

    def plot_umap(
        self, 
        model: torch.nn.Module, 
        source_loader: torch.utils.data.DataLoader, 
        target_loader: torch.utils.data.DataLoader, 
        prefix: str
    ) -> None:
        """Visualizes domain alignment in the latent space using UMAP.

        Args:
            model (nn.Module): The shared feature extractor.
            source_loader (DataLoader): Source domain data loader.
            target_loader (DataLoader): Target domain data loader.
            prefix (str): Filename prefix for the saved plot.
        """
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
        model: torch.nn.Module,
        source_loader: torch.utils.data.DataLoader,
        target_loader: torch.utils.data.DataLoader,
        class_names: List[str],
        prefix: str,
        image_zoom: float = 0.25,
        min_dist_plots: float = 1.0,
        use_lime: bool = False
    ) -> None:
        """Embeds representative images on the UMAP latent space plot.

        Args:
            model (nn.Module): The model to visualize.
            source_loader (DataLoader): Data from the Source domain.
            target_loader (DataLoader): Data from the Target domain.
            class_names (List[str]): Human-readable class names.
            prefix (str): Filename prefix for saving.
            image_zoom (float): Size multiplier for the overlaid images.
            min_dist_plots (float): Threshold to avoid image overlap in the plot.
            use_lime (bool): If True, analyzes lighting variation correlation (LIME).
        """
        # =====================================================
        # SOURCE: Processing standard data (without LIME/lighting vars)
        # =====================================================
        data_s = self._get_inference_data(
                model,
                source_loader,
                return_lime_variant=False,
                domain_id=0
            )
        
        # Source has 6 elements in the returned tuple as return_lime_variant is False
        y_s, _, _, feat_s, imgs_s, dom_s = data_s

        # =====================================================
        # TARGET: Processing domain-shifted data (optionally with LIME vars)
        # =====================================================
        data_t = self._get_inference_data(
            model,
            target_loader,
            return_lime_variant=use_lime,
            domain_id=1
        )

        if use_lime:
            y_t, _, _, feat_t, imgs_t, lime_t, dom_t = data_t  # Extracts lime_t
        else:
            y_t, _, _, feat_t, imgs_t, dom_t = data_t

        # Aggregation of domain data
        features = np.concatenate([feat_s, feat_t])
        labels = np.concatenate([y_s, y_t])
        domains = np.concatenate([dom_s, dom_t])
        images_tensor = torch.cat([imgs_s, imgs_t])

        # Dimension reduction
        reducer = umap.UMAP(
            n_neighbors=30,
            min_dist=0.3,
            metric='cosine',
            random_state=42
        )
        embedding = reducer.fit_transform(features)

        # Plot setup
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
        # Plot representative image overlays
        # =====================================================
        shown_positions = np.array([[1000.0, 1000.0]])

        for i in range(len(embedding)):
            curr_pos = embedding[i]
            dist = np.sum((curr_pos - shown_positions) ** 2, axis=1)

            # Check if position is sufficiently far from previously plotted images
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
        # LIME Variation Analysis + Pearson Correlation
        # =====================================================
        if use_lime:
            # Analyze correlation between lighting variation and UMAP latent axes
            tgt_idx = domains == 1 
            lime_values = lime_t.astype(float)
            umap_x = embedding[tgt_idx, 0]
            umap_y = embedding[tgt_idx, 1]

            try:
                corr_x, p_x = pearsonr(lime_values, umap_x)
                corr_y, p_y = pearsonr(lime_values, umap_y)

                print("\nPearson Correlation (LIME vs UMAP for Target Domain):")
                print(f"UMAP-x → r = {corr_x:.4f} | p = {p_x:.4e}")
                print(f"UMAP-y → r = {corr_y:.4f} | p = {p_y:.4e}")

            except Exception:
                print("Pearson correlation could not be computed.")

        # =====================================================
        # Visualization Aesthetics and Legend
        # =====================================================
        domain_legend = [
            Line2D([0], [0], marker='o', color='w',
                label='Source (Normal)', markerfacecolor='gray', markersize=12),
            Line2D([0], [0], marker='^', color='w',
                label='Target (LIME Vars)', markerfacecolor='gray', markersize=12)
        ]

        class_legend = [
            Line2D([0], [0], marker='s', color='w',
                label=class_names[i],
                markerfacecolor=cmap(i),
                markersize=12)
            for i in range(len(class_names))
        ]

        ax.legend(handles=domain_legend + class_legend, loc='upper right', fontsize=18)
        plt.title("Latent Space Structure (Domain • Class" + (" • LIME" if use_lime else "") + ")", fontsize=18)
        plt.axis("off")

        path = self.output_dir / f"{prefix}_umap_samples.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved UMAP plot to {path}")



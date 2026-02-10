import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import umap
from sklearn.metrics import confusion_matrix
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pathlib import Path
from typing import List, Tuple, Optional

class BananaVisualizer:
    """
    Engine de visualización para Computer Vision y Domain Adaptation.
    Genera reportes de clasificación, alineación de dominios y análisis de muestras.
    """
    
    def __init__(self, device: torch.device, output_dir: str = "outputs"):
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Parámetros para revertir la normalización de ImageNet
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def _denormalize(self, img_tensor: torch.Tensor) -> np.ndarray:
        """Convierte un tensor normalizado a imagen RGB para visualización."""
        img = img_tensor.cpu().numpy().transpose(1, 2, 0)
        img = img * self.std + self.mean
        return np.clip(img, 0, 1)

    def _get_inference_data(self, model, loader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
        """Obtiene etiquetas reales, predicciones, features e imágenes en una sola pasada."""
        model.eval()
        all_feats, all_labels, all_preds, all_imgs = [], [], [], []
        
        with torch.no_grad():
            for imgs, labels in loader:
                all_imgs.append(imgs.clone())
                
                imgs = imgs.to(self.device)
                feats = model(imgs, mode='feature')
                logits = model(imgs, mode='class')
                preds = torch.argmax(logits, dim=1)
                
                all_feats.append(feats.cpu().numpy())
                all_labels.append(labels.numpy())
                all_preds.append(preds.cpu().numpy())
        
        return (
            np.concatenate(all_labels),
            np.concatenate(all_preds),
            np.concatenate(all_feats),
            torch.cat(all_imgs)
        )

    def plot_confusion_matrix(self, model, loader, class_names: List[str], prefix: str):
        """Genera y guarda la matriz de confusión normalizada."""
        y_true, y_pred, _, _ = self._get_inference_data(model, loader)
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalización por fila (Recall)
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        acc = np.trace(cm) / np.sum(cm)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        
        plt.title(f'Matriz de Confusión: {prefix}\nAccuracy Global: {acc:.2%}', fontsize=12)
        plt.xlabel('Predicción')
        plt.ylabel('Etiqueta Real')
        
        path = self.output_dir / f"{prefix}_confusion_matrix.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Matriz de Confusión guardada en: {path}")

    def plot_umap(self, model, source_loader, target_loader, prefix: str):
        """Visualiza la alineación de dominios (Source vs Target)."""
        _, _, feat_s, _ = self._get_inference_data(model, source_loader)
        _, _, feat_t, _ = self._get_inference_data(model, target_loader)
        
        features = np.concatenate([feat_s, feat_t])
        domains = np.concatenate([np.zeros(len(feat_s)), np.ones(len(feat_t))])
        
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric='cosine', random_state=42)
        embedding = reducer.fit_transform(features)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=domains, cmap='coolwarm', s=10, alpha=0.5)
        plt.legend(handles=scatter.legend_elements()[0], labels=['Source (Sintético)', 'Target (Real)'])
        plt.title(f"Alineación de Dominios - {prefix}")
        
        path = self.output_dir / f"{prefix}_umap_alignment.png"
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"✅ UMAP de Dominios guardado en: {path}")

    def plot_umap_with_images(
        self, 
        model, 
        loader, 
        class_names: List[str], 
        prefix: str, 
        min_dist_plots: float = 0.1,
        image_zoom: float = 0.07
    ):
        """Genera un UMAP proyectando imágenes reales sobre los puntos del espacio latente."""
        y_true, _, features, images_tensor = self._get_inference_data(model, loader)
        
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric='cosine', random_state=42)
        embedding = reducer.fit_transform(features)
        
        plt.figure(figsize=(12, 10))
        cmap = plt.get_cmap('tab10')
        
        # Puntos de fondo
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
                ab = AnnotationBbox(
                    imagebox, curr_pos,
                    bboxprops={"edgecolor": cmap(y_true[i]), "lw": 1.5}
                )
                ax.add_artist(ab)

        plt.title(f"Espacio Latente Cualitativo - {prefix}")
        plt.axis("off")
        
        path = self.output_dir / f"{prefix}_umap_samples.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ UMAP con imágenes guardado en: {path}")
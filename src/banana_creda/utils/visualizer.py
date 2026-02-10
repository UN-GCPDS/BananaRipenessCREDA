import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import umap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pathlib import Path
from typing import List, Tuple, Optional

class BananaVisualizer:
    def __init__(self, device: torch.device, output_dir: str = "outputs", config=None):
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Parámetros de des-normalización (usando los de tu CONFIG)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def _denormalize(self, img_tensor: torch.Tensor) -> np.ndarray:
        """Convierte un tensor de imagen a un array de numpy RGB [0, 1]."""
        img = img_tensor.cpu().numpy().transpose(1, 2, 0)
        img = img * self.std + self.mean
        return np.clip(img, 0, 1)

    def _get_inference_data(self, model, loader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        model.eval()
        all_feats, all_labels, all_imgs = [], [], []
        
        with torch.no_grad():
            for imgs, labels in loader:
                # Guardamos una copia de las imágenes antes de enviarlas al device (para CPU/Plotting)
                all_imgs.append(imgs.clone())
                
                imgs = imgs.to(self.device)
                feats = model(imgs, mode='feature')
                
                all_feats.append(feats.cpu().numpy())
                all_labels.append(labels.numpy())
        
        features = np.concatenate(all_feats)
        y_true = np.concatenate(all_labels)
        # Tensor de imágenes: [N, 3, H, W]
        images_tensor = torch.cat(all_imgs)
        
        return y_true, features, images_tensor

    def plot_umap_with_images(
        self, 
        model, 
        loader, 
        class_names: List[str], 
        prefix: str, 
        min_dist_plots: float = 0.05,
        image_zoom: float = 0.08
    ):
        """
        Genera un UMAP donde algunos puntos son reemplazados por las imágenes reales.
        
        Args:
            min_dist_plots: Distancia mínima en el plano 2D para dibujar otra imagen (evita solapamiento).
            image_zoom: Tamaño de las miniaturas.
        """
        y_true, features, images_tensor = self._get_inference_data(model, loader)
        
        # 1. Proyección UMAP
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric='cosine', random_state=42)
        embedding = reducer.fit_transform(features)
        
        # 2. Configurar Plot
        plt.figure(figsize=(12, 10))
        cmap = plt.cm.get_cmap('tab10', len(class_names))
        
        # Dibujar los puntos base
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=y_true, 
                            cmap=cmap, s=15, alpha=0.3)
        
        # 3. Superposición de Imágenes (AnnotationBbox)
        # Inicializamos con una posición lejana para el primer cálculo
        shown_positions = np.array([[1000.0, 1000.0]]) 
        
        ax = plt.gca()
        for i in range(len(embedding)):
            curr_pos = embedding[i]
            
            # Calcular distancia a las imágenes ya dibujadas
            dist = np.sum((curr_pos - shown_positions) ** 2, axis=1)
            
            if np.min(dist) > min_dist_plots:
                shown_positions = np.r_[shown_positions, [curr_pos]]
                
                # Des-normalizar y preparar imagen
                img_to_show = self._denormalize(images_tensor[i])
                
                imagebox = OffsetImage(img_to_show, zoom=image_zoom)
                ab = AnnotationBbox(
                    imagebox, 
                    curr_pos,
                    bboxprops={"edgecolor": cmap(y_true[i]), "lw": 1.5}
                )
                ax.add_artist(ab)

        plt.title(f"Espacio Latente con Muestras - {prefix}", fontsize=14)
        plt.colorbar(scatter, ticks=range(len(class_names)), label="Clases")
        plt.axis("off")
        
        save_path = self.output_dir / f"{prefix}_umap_images.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Visualización UMAP con imágenes guardada en: {save_path}")
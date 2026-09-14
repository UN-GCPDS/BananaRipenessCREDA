# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
# ---

# %% [markdown]
# <div align="center">
#   <img src="https://raw.githubusercontent.com/UN-GCPDS/BananaRipenessCREDA/main/assets/banner.png" alt="BananaRipenessCREDA Banner" width="850"/>
# </div>
#
# # **Taller Práctico IEEE: De la Teoría al Borde (Edge AI)**
# ## **Clasificación y Cuantización de Madurez de Bananos con MobileNetV3, CREDA y ExecuTorch**
#
# ---
#
# ### **Universidad Estatal Península de Santa Elena (UPSE)**
# #### **Facultad de Sistemas y Telecomunicaciones**
# #### **Rama Estudiantil IEEE UPSE**
#
# * **Docentes a cargo:**
#   * **Prof. Luis Chuquimarca** (`lchuquimarca@upse.edu.ec`)
#   * **Prof. Lucas Iturriago** (`liturriago@unal.edu.co`)
#
# ---
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UN-GCPDS/BananaRipenessCREDA/blob/main/notebooks/Workshop/Taller_IEEE_BananaRipenessCREDA.ipynb)
# ![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
# ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)
# ![ExecuTorch](https://img.shields.io/badge/ExecuTorch-INT8%20XNNPACK-orange)
# ![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces%20%26%20Gradio-FFD21E?logo=huggingface)
#
# ---
#
# ### **Descripción y Objetivos del Taller**
#
# En aplicaciones agrícolas e industriales de Visión Artificial, los modelos entrenados en laboratorios o imágenes sintéticas a menudo sufren una caída drástica de rendimiento al desplegarse en entornos reales debido a variaciones de iluminación, sombras y fondos (**Domain Shift**). Además, el despliegue en dispositivos de bajo consumo (**Edge Devices** como Raspberry Pi o microcontroladores) exige optimizar drásticamente la latencia y la memoria.
#
# En este taller aprenderás a:
# 1. **Configurar el entorno en Google Colab** y vincular el dataset oficial de madurez de bananos (`bananaripeness`) alojado en Kaggle Hub.
# 2. **Entrenar una arquitectura MobileNetV3** bajo el paradigma de **Adaptación de Dominio No Supervisada (UDA)** utilizando el algoritmo **CREDA** (*Class-Regularized Entropy Domain Adaptation*).
# 3. **Evaluar rigurosamente el modelo** mediante métricas científicas (Precision, Recall, F1-Score), Matrices de Confusión, Curvas ROC y visualizaciones latentes con **UMAP**.
# 4. **Convertir y Cuantizar el modelo a INT8 con ExecuTorch (PT2E + XNNPACK)**, analizando la reducción de memoria y la aceleración en CPU.
# 5. **Desplegar la solución en Hugging Face Spaces** con una interfaz interactiva en **Gradio**, dotada de arquitectura Docker y un mecanismo de fallback resiliente.
# 6. **Consumir la API remota del Space** programáticamente utilizando `gradio_client`.

# %% [markdown]
# ---
# # 1. Preparación del Entorno en Google Colab
#
# Primero comprobaremos la disponibilidad de aceleración por GPU (CUDA) y clonaremos el repositorio del proyecto.

# %%
# 1.1. Verificar acelerador de hardware GPU asignado por Colab
!nvidia-smi

# %% [markdown]
# ## 1.2. Clonación e Instalación de Dependencias
#
# Instalaremos el paquete modular `banana_creda` en modo editable junto con sus extensiones de cuantización (`convert`) y explicabilidad (`explain`), además de librerías para despliegue y descarga de datasets.

# %%
import os
import sys

# Clonar el repositorio oficial si no existe localmente
if not os.path.exists("BananaRipenessCREDA"):
    print("[*] Clonando repositorio BananaRipenessCREDA...")
    !git clone https://github.com/UN-GCPDS/BananaRipenessCREDA.git
    %cd BananaRipenessCREDA
else:
    %cd BananaRipenessCREDA

print(f"[*] Directorio de trabajo actual: {os.getcwd()}")

# Instalar dependencias del proyecto
!pip install -e .[convert,explain,dev] -q
!pip install kagglehub gradio gradio_client huggingface_hub pyyaml -q

# %% [markdown]
# ## 1.3. Descarga y Verificación del Dataset
#
# Utilizaremos `kagglehub` para descargar el dataset `lucasiturriago/bananaripeness`.
# Los datos se ubicarán en el directorio estándar de cache de Kaggle:
# `/root/.cache/kagglehub/datasets/lucasiturriago/bananaripeness/versions/1`

# %%
import kagglehub
from pathlib import Path

# Descargar el dataset oficial
print("[*] Descargando dataset desde Kaggle Hub...")
downloaded_path = kagglehub.dataset_download("lucasiturriago/bananaripeness")
DATASET_DIR = Path("/root/.cache/kagglehub/datasets/lucasiturriago/bananaripeness/versions/1")

# En caso de que se ejecute en un entorno donde la ruta varíe:
if not DATASET_DIR.exists() and Path(downloaded_path).exists():
    DATASET_DIR = Path(downloaded_path)

print(f"\n[+] Directorio del dataset: {DATASET_DIR}")
print("[+] Estructura de carpetas disponibles:")
for folder in sorted(DATASET_DIR.iterdir()):
    if folder.is_dir():
        samples_count = len(list(folder.rglob("*.jpg")) + list(folder.rglob("*.png")))
        print(f"  📂 {folder.name:<20} -> {samples_count} imágenes")

# %% [markdown]
# ---
# # 2. Configuración Dinámica del Experimento (MobileNetV3 + CREDA)
#
# Para este taller utilizaremos el backbone **MobileNetV3**, diseñado específicamente para maximizar la precisión mientras mantiene una baja latencia en procesadores móviles y embebidos.
#
# Las clases de madurez corresponden a:
# * **Class A**: Fruto inmaduro / Verde (*Unripe*).
# * **Class B**: Inicio de maduración / Verde-Amarillo (*Barely Ripe*).
# * **Class C**: Maduro óptimo para consumo fresco (*Ripe*).
# * **Class D**: Sobre-maduro / Manchas oscuras (*Overripe*).
#
# Generaremos un archivo de configuración YAML validado por los esquemas Pydantic del paquete `banana_creda`.

# %%
import yaml

# Definición declarativa de la configuración del experimento (CREDA - Medium Variation)
colab_config = {
    "data": {
        "source_data_dir": str(DATASET_DIR / "Original"),
        "target_data_dir": str(DATASET_DIR / "Medium_Variation"),
        "batch_size": 32,
        "img_size": 224,
        "num_workers": 2,
        "imagenet_mean": [0.485, 0.456, 0.406],
        "imagenet_std": [0.229, 0.224, 0.225],
        "use_lime_on_target": True,
        "use_augmentation": True
    },
    "model": {
        "num_classes": 4,
        "pretrained": True,
        "backbone": "mobilenetv3",
        "dropout_rate": 0.2
    },
    "training": {
        "epochs": 20,
        "lr": 0.0001,
        "gamma": 0.94,
        "warmup": True,
        "warmup_epochs": 5,
        "warmup_threshold": 0.9,
        "lambda_creda": 1.0,
        "use_uncertainty": True,
        "sigma": "auto",
        "use_amp": True,
        "device": "cuda",
        "seed": 42
    },
    "experiment": {
        "name": "medium_variation_mobilenetv3_experiment",
        "version": 3,
        "output_dir": "outputs/mobilenetv3/experiment_3",
        "save_results": True
    }
}

CONFIG_FILE = "configs/training/mobilenetv3/workshop_colab.yaml"
os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)

with open(CONFIG_FILE, "w", encoding="utf-8") as f:
    yaml.dump(colab_config, f, sort_keys=False, default_flow_style=False)

print(f"[+] Archivo de configuración generado exitosamente: {CONFIG_FILE}")

# %% [markdown]
# ---
# # 3. Entrenamiento con Adaptación de Dominio (CREDA)
#
# Ejecutaremos el script [`scripts/da_train.py`](file:///scripts/da_train.py). Durante el proceso:
# 1. Se inicializa el modelo pre-entrenado `mobilenet_v3_large`.
# 2. Se activa el cálculo de la divergencia de Rényi condicional de orden 2 con kernel RBF adaptativo para alinear las características del dominio sintético/estudio con el dominio de iluminación variable.
# 3. Los mejores pesos se guardan en `outputs/mobilenetv3/experiment_3/model_final.pth`.

# %%
# Ejecutar entrenamiento de Domain Adaptation (CREDA con MobileNetV3 en Medium_Variation)
!python scripts/da_train.py --config configs/training/mobilenetv3/workshop_colab.yaml

# %% [markdown]
# ---
# # 4. Evaluación Científica del Modelo
#
# Procederemos a evaluar el modelo final sobre el conjunto de prueba del dominio destino (Target Test Set), midiendo exactitud, precisión, recall, F1-score por clase y generando las visualizaciones de diagnóstico.

# %%
# 4.1. Ejecución del script de evaluación
!python scripts/evaluation.py \
    --config configs/training/mobilenetv3/workshop_colab.yaml \
    --model outputs/mobilenetv3/experiment_3/model_final.pth \
    --output_dir outputs/mobilenetv3/experiment_3

# %% [markdown]
# ## 4.2. Visualización Diagnóstica de Resultados

# %% [markdown]
# ### 4.2.1. Métricas Cuantitativas: Matriz de Confusión y Curva ROC
#
# La matriz de confusión permite verificar la tasa de aciertos y posibles confusiones entre clases adyacentes (por ejemplo, entre Fruto Verde e Inicio de Maduración). La curva ROC multicategoría demuestra la capacidad discriminativa del modelo en función de la variación de umbrales de decisión.

# %%
from IPython.display import Image, display
from pathlib import Path

eval_dir = Path("outputs/mobilenetv3/experiment_3/evaluation")
if not eval_dir.exists():
    eval_dir = Path("outputs/mobilenetv3/experiment_3")

# Visualizar Matriz de Confusión
cm_plots = list(eval_dir.glob("*confusion_matrix*.png"))
if cm_plots:
    print(f"[+] Matriz de Confusión ({cm_plots[0].name}):")
    display(Image(filename=str(cm_plots[0]), width=700))

# Visualizar Curva ROC
roc_plots = list(eval_dir.glob("*roc_curve*.png"))
if roc_plots:
    print(f"[+] Curva ROC Multiclase ({roc_plots[0].name}):")
    display(Image(filename=str(roc_plots[0]), width=700))

# %% [markdown]
# ### 4.2.2. Proyección Latente UMAP: Alineación de Dominios (CREDA)
#
# **¿Qué representa esta gráfica?**
# UMAP (*Uniform Manifold Approximation and Projection*) reduce la dimensión de las características profundas extraídas por la penúltima capa de MobileNetV3 (1280 dimensiones) a un espacio visual 2D:
# * **Dominio Origen (Source / Original):** Muestras capturadas bajo condiciones homogéneas de luz.
# * **Dominio Destino (Target / Medium Variation):** Muestras capturadas bajo variaciones lumínicas complejas (sombras, saturación, baja luminosidad).
#
# **Criterio de Éxito en Adaptación de Dominio:**
# Si el algoritmo **CREDA** ha convergido exitosamente, los puntos de ambos dominios se encontrarán **superpuestos armónicamente por cada estado de madurez**, demostrando que la red ha aprendido representaciones semánticas invariantes al cambio de iluminación.

# %%
# 1. Gráfica UMAP de Alineación de Distribuciones (Source vs. Target)
umap_alignment_plots = list(eval_dir.glob("*umap_alignment*.png"))
if umap_alignment_plots:
    print(f"\n[+] UMAP de Alineación de Dominio: {umap_alignment_plots[0].name}")
    display(Image(filename=str(umap_alignment_plots[0]), width=750))
else:
    print("[!] No se encontró la gráfica de alineación UMAP.")

# 2. Gráfica UMAP Cualitativa con Muestras Representativas de Imágenes
umap_images_plots = list(eval_dir.glob("*umap_with_images*.png")) or list(eval_dir.glob("*Latent Space*.png"))
if umap_images_plots:
    print(f"\n[+] UMAP Cualitativo con Imágenes Proyectadas: {umap_images_plots[0].name}")
    display(Image(filename=str(umap_images_plots[0]), width=850))
else:
    print("[INFO] Ejecuta el análisis cualitativo o scripts/explain.py para generar el mapa con imágenes embebidas.")

# %% [markdown]
# ---
# # 5. Conversión y Cuantización a INT8 con ExecuTorch (PT2E + XNNPACK)
#
# ### **¿Por qué ExecuTorch y Cuantización INT8?**
# * **ExecuTorch** es el runtime de nueva generación de PyTorch diseñado específicamente para dispositivos de borde (teléfonos móviles, wearables y SBCs como Raspberry Pi).
# * **Cuantización INT8:** Reduce la precisión de los pesos y activaciones de 32 bits en coma flotante (FP32) a enteros de 8 bits (INT8). Esto permite:
#   * **Reducción de tamaño en ~75%:** Menor consumo de memoria flash y RAM.
#   * **Mayor velocidad:** Uso eficiente de instrucciones SIMD vectoriales de la CPU (ARM NEON / x86 AVX) a través del backend de alto rendimiento **XNNPACK**.
#
# Ejecutaremos [`scripts/convert_and_quantize.py`](file:///scripts/convert_and_quantize.py), el cual realiza:
# 1. Exportación de grafo ATen con `torch.export`.
# 2. Inserción de observadores y calibración estática con 100 muestras reales.
# 3. Conversión a operadores cuantizados y lowering a formato binario `.pte`.
# 4. Comparativa cuantitativa de latencia y métricas.

# %%
# Ejecutar conversión y cuantización PT2E con backend XNNPACK
!python scripts/convert_and_quantize.py \
    --config configs/training/mobilenetv3/workshop_colab.yaml \
    --model outputs/mobilenetv3/experiment_3/model_final.pth

# %% [markdown]
# ---
# # 6. Despliegue en Hugging Face Spaces con Gradio y Docker
#
# Una vez obtenido el modelo optimizado de ExecuTorch (`model_quantized_xnnpack.pte`) y los pesos PyTorch (`model_final.pth`), crearemos un **Hugging Face Space** interactivo.
#
# ### **Arquitectura del Space:**
# * **Frontend:** Gradio Blocks interactivo que permite arrastrar imágenes de bananos y devuelve la clase predicha con su respectiva interpretación comercial.
# * **Backend:** Servidor Docker en Python 3.10 con el runtime compilado de ExecuTorch y backend XNNPACK.
# * **Mecanismo de Resiliencia (Fallback):** Si la arquitectura del host remoto en Hugging Face no soporta el runtime binario de ExecuTorch en C++, la aplicación conmuta automáticamente al modelo PyTorch FP32 para asegurar 100% de disponibilidad sin interrupciones.

# %% [markdown]
# ## 6.1. Autenticación en Hugging Face
#
# Ejecuta la siguiente celda e introduce tu **User Access Token** (con permisos de *write*) obtenido en [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

# %%
from huggingface_hub import notebook_login

# Inicia sesión en tu cuenta de Hugging Face
notebook_login()

# %% [markdown]
# ## 6.2. Parámetros del Space
#
# Configura tu nombre de usuario de Hugging Face y el nombre para el Space.

# %%
HF_USER = "tu_usuario_hf"  # <-- REEMPLAZA CON TU USUARIO DE HUGGING FACE
SPACE_NAME = "banana-ripeness-detector"
SPACE_DIR = Path(SPACE_NAME)
SPACE_DIR.mkdir(parents=True, exist_ok=True)

print(f"Space configurado: https://huggingface.co/spaces/{HF_USER}/{SPACE_NAME}")

# %% [markdown]
# ## 6.3. Generación de `app.py` (Gradio con Runtime de ExecuTorch + Fallback)
#
# Escribimos el código de la aplicación web que gestionará la inferencia en tiempo real.

# %%
app_code = """import os
import sys
import numpy as np
import torch
import gradio as gr
from PIL import Image

# -------------------------------------------------------------
# 1. Carga de Librerías y Detección de Runtime ExecuTorch
# -------------------------------------------------------------
try:
    from executorch.runtime import Runtime
    EXECUTORCH_AVAILABLE = True
    print("[INFO] Runtime nativo de ExecuTorch cargado exitosamente.")
except ImportError:
    EXECUTORCH_AVAILABLE = False
    print("[WARNING] ExecuTorch no disponible. Se utilizará PyTorch como fallback.")

from torchvision import transforms
import torchvision.models as models
import torch.nn as nn

# Rutas de los modelos
PATH_PTE = "model_quantized_xnnpack.pte"
PATH_PTH = "model_final.pth"

# -------------------------------------------------------------
# 2. Definición de Clases e Información de Madurez
# -------------------------------------------------------------
CLASSES = ["Class A", "Class B", "Class C", "Class D"]

DESCRIPCION_CLASES = {
    "Class A": {
        "titulo": "🟢 Clase A: Inmaduro / Totalmente Verde",
        "estado": "Fruto en fase inicial de cosecha o almacenamiento en frío.",
        "consumo": "No apto para consumo fresco inmediato. Alta concentración de almidón no digerible.",
        "aplicacion": "Ideal para transporte de larga distancia o cocción tradicional (plátano verde)."
    },
    "Class B": {
        "titulo": "🟡🟢 Clase B: Inicio de Maduración / Verde-Amarillo",
        "estado": "Inicio de la conversión enzimática de almidones a azúcares simples.",
        "consumo": "Textura firme y sabor ligeramente astringente.",
        "aplicacion": "Punto de llegada a centros de distribución y supermercados."
    },
    "Class C": {
        "titulo": "🟡 Clase C: Maduro Óptimo / Amarillo Brillante",
        "estado": "Punto máximo de calidad organoléptica y balance de azúcares.",
        "consumo": "Óptimo para consumo directo fresco. Máxima digestibilidad.",
        "aplicacion": "Venta directa al consumidor minorista."
    },
    "Class D": {
        "titulo": "🟤 Clase D: Sobre-maduro / Manchas Pardas o Negras",
        "estado": "Fase avanzada de maduración con azúcares concentrados y piel delgada.",
        "consumo": "Sabor muy dulce y textura suave.",
        "aplicacion": "Excelente para repostería, pan de banano, batidos o procesamiento industrial."
    }
}

# Preprocesamiento estándar ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def preprocesar(img: Image.Image) -> torch.Tensor:
    if img.mode != "RGB":
        img = img.convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform(img).unsqueeze(0)

# -------------------------------------------------------------
# 3. Gestor de Inferencia (Runner con Fallback)
# -------------------------------------------------------------
class BananaInferenceEngine:
    def __init__(self, pte_path: str, pth_path: str):
        self.use_executorch = EXECUTORCH_AVAILABLE and os.path.exists(pte_path)
        
        if self.use_executorch:
            try:
                print(f"[*] Cargando modelo cuantizado ExecuTorch: {pte_path}")
                self.runtime = Runtime.get()
                self.program = self.runtime.load_program(pte_path)
                self.method = self.program.load_method("forward")
                print("[+] ExecuTorch inicializado correctamente.")
            except Exception as e:
                print(f"[!] Error al cargar modelo ExecuTorch: {e}. Activando fallback de PyTorch.")
                self.use_executorch = False

        if not self.use_executorch:
            print(f"[*] Cargando modelo fallback PyTorch FP32...")
            # Recrear MobileNetV3 con cabeza de 4 clases
            self.model = models.mobilenet_v3_large(weights=None)
            self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, len(CLASSES))
            
            if os.path.exists(pth_path):
                print(f"[*] Cargando pesos entrenados desde: {pth_path}")
                try:
                    state_dict = torch.load(pth_path, map_location="cpu")
                    # Soporte por si los pesos provienen de BananaModel
                    new_state = {}
                    for k, v in state_dict.items():
                        cleaned_k = k.replace("encoder.", "").replace("head.", "classifier.")
                        new_state[cleaned_k] = v
                    self.model.load_state_dict(new_state, strict=False)
                except Exception as e:
                    print(f"[!] Aviso al cargar pesos: {e}")
            self.model.eval()

    def predict(self, tensor_img: torch.Tensor):
        if self.use_executorch:
            outputs = self.method.execute([tensor_img])
            logits = outputs[0]
            if isinstance(logits, list):
                logits = logits[0]
            if not isinstance(logits, torch.Tensor):
                logits = torch.from_numpy(np.array(logits))
            probs = torch.softmax(logits[0], dim=0)
            return probs, "ExecuTorch INT8 (XNNPACK)"
        else:
            with torch.no_grad():
                logits = self.model(tensor_img)
                probs = torch.softmax(logits[0], dim=0)
            return probs, "PyTorch FP32 (Fallback)"

# Inicializar motor global
engine = BananaInferenceEngine(PATH_PTE, PATH_PTH)

# -------------------------------------------------------------
# 4. Función de Predicción para la Interfaz
# -------------------------------------------------------------
def clasificar_madurez(imagen_pil: Image.Image):
    if imagen_pil is None:
        return None, "", "Por favor carga una imagen de un banano."

    try:
        tensor = preprocesar(imagen_pil)
        probs, backend_name = engine.predict(tensor)

        # Diccionario de probabilidades para gr.Label
        prob_dict = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
        
        # Clase predicha con mayor probabilidad
        top_idx = int(torch.argmax(probs).item())
        clase_top = CLASSES[top_idx]
        info = DESCRIPCION_CLASES[clase_top]

        detalle_md = f\"\"\"
### {info['titulo']}
* **Nivel de Confianza:** {float(probs[top_idx])*100:.2f}%
* **Estado Fisiológico:** {info['estado']}
* **Recomendación de Consumo:** {info['consumo']}
* **Destino Comercial Sugerido:** {info['aplicacion']}
\"\"\"
        info_backend = f"**Motor de Inferencia:** `{backend_name}`"
        return prob_dict, detalle_md, info_backend

    except Exception as e:
        return None, f"Error en inferencia: {str(e)}", "Fallo"

# -------------------------------------------------------------
# 5. Interfaz Gráfica con Gradio Blocks
# -------------------------------------------------------------
with gr.Blocks(theme=gr.themes.Soft(primary_hue="amber", neutral_hue="slate"), title="Detector de Madurez de Bananos - ExecuTorch") as demo:
    gr.Markdown(\"\"\"
    # 🍌 Clasificador de Madurez de Bananos en el Borde (Edge AI)
    ### **Desarrollado con PyTorch, ExecuTorch (INT8) y Algoritmo CREDA**
    *Taller IEEE - Universidad Estatal Península de Santa Elena (UPSE)*
    *Docentes: Luis Chuquimarca (lchuquimarca@upse.edu.ec) & Lucas Iturriago (liturriago@unal.edu.co)*
    \"\"\")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Cargar Imagen de Banano")
            btn_predict = gr.Button("🔍 Analizar Estado de Madurez", variant="primary")
            backend_badge = gr.Markdown("**Motor de Inferencia:** `Detectando...`")

        with gr.Column(scale=1):
            output_label = gr.Label(num_top_classes=4, label="Probabilidades por Clase")
            output_details = gr.Markdown("### Selecciona una imagen y haz clic en 'Analizar'.")

    btn_predict.click(
        fn=clasificar_madurez,
        inputs=[input_image],
        outputs=[output_label, output_details, backend_badge]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
"""

with open(SPACE_DIR / "app.py", "w", encoding="utf-8") as f:
    f.write(app_code)

print(f"[+] Archivo app.py creado en {SPACE_DIR}/app.py")

# %% [markdown]
# ## 6.4. Generación del `Dockerfile`
#
# Configuramos el entorno de ejecución en Hugging Face con Python 3.10-slim y las librerías necesarias.

# %%
dockerfile_code = """FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    libgl1 \\
    libglib2.0-0 \\
    git \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \\
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

RUN pip install --no-cache-dir --upgrade pip && \\
    pip install --no-cache-dir --quiet \\
    executorch \\
    torch==2.11.0 \\
    torchvision \\
    numpy \\
    pillow \\
    gradio \\
    gradio_client

COPY --chown=user app.py ./app.py
COPY --chown=user model_quantized_xnnpack.pte ./model_quantized_xnnpack.pte
COPY --chown=user model_final.pth ./model_final.pth

EXPOSE 7860

CMD ["python", "app.py"]
"""

with open(SPACE_DIR / "Dockerfile", "w", encoding="utf-8") as f:
    f.write(dockerfile_code)

print(f"[+] Archivo Dockerfile creado en {SPACE_DIR}/Dockerfile")

# %% [markdown]
# ## 6.5. Generación de los Metadatos `README.md`
#
# Hugging Face requiere un bloque inicial en formato YAML para definir que el Space operará bajo el SDK de Docker.

# %%
readme_code = f"""---
title: Banana Ripeness Detector - MobileNetV3 ExecuTorch INT8
emoji: 🍌
colorFrom: yellow
colorTo: green
sdk: docker
app_file: app.py
pinned: false
license: mit
---

# Clasificador de Madurez de Bananos (Edge AI)
Aplicación interactiva desarrollada para el **Taller IEEE** en la **Universidad Estatal Península de Santa Elena (UPSE)**.
* **Arquitectura:** MobileNetV3 Large
* **Dominio:** Class-Regularized Entropy Domain Adaptation (CREDA)
* **Aceleración Edge:** Cuantización INT8 con ExecuTorch y backend XNNPACK
* **Docentes:** Luis Chuquimarca (lchuquimarca@upse.edu.ec) & Lucas Iturriago (liturriago@unal.edu.co)
"""

with open(SPACE_DIR / "README.md", "w", encoding="utf-8") as f:
    f.write(readme_code)

print(f"[+] Archivo README.md creado en {SPACE_DIR}/README.md")

# %% [markdown]
# ## 6.6. Copiado de Modelos al Directorio del Space
#
# Copiamos tanto el modelo cuantizado de ExecuTorch (`.pte`) como los pesos originales (`.pth`) dentro de la carpeta del Space para su empaquetado.

# %%
import shutil

# Rutas de los artefactos generados
source_pte = Path("outputs/mobilenetv3/experiment_3/model_quantized_xnnpack.pte")
source_pth = Path("outputs/mobilenetv3/experiment_3/model_final.pth")

# Copiar modelo ExecuTorch INT8
if source_pte.exists():
    shutil.copy(source_pte, SPACE_DIR / "model_quantized_xnnpack.pte")
    print(f"[+] Modelo ExecuTorch copiado ({source_pte.stat().st_size / (1024*1024):.2f} MB)")
else:
    print("[!] AVISO: No se encontró model_quantized_xnnpack.pte. Asegúrate de ejecutar la Sección 5.")

# Copiar modelo PyTorch FP32
if source_pth.exists():
    shutil.copy(source_pth, SPACE_DIR / "model_final.pth")
    print(f"[+] Modelo PyTorch copiado ({source_pth.stat().st_size / (1024*1024):.2f} MB)")
else:
    print("[!] AVISO: No se encontró model_final.pth. Asegúrate de ejecutar la Sección 3.")

print(f"\nContenido del directorio de despliegue '{SPACE_NAME}':")
for file in SPACE_DIR.iterdir():
    print(f"  📄 {file.name} ({file.stat().st_size / 1024:.1f} KB)")

# %% [markdown]
# ## 6.7. Publicación y Despliegue en Hugging Face
#
# Subiremos la carpeta directamente a Hugging Face usando `HfApi.upload_folder`. Esto evita configuraciones complejas de Git LFS y automatiza el inicio de la compilación en el servidor remoto.

# %%
from huggingface_hub import HfApi

api = HfApi()
repo_id = f"{HF_USER}/{SPACE_NAME}"

# 1. Crear el repositorio en Hugging Face si no existe
try:
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    print(f"[+] Espacio '{repo_id}' confirmado en Hugging Face.")
except Exception as e:
    print(f"[!] Error o verificación de repositorio: {e}")

# 2. Subir todos los archivos generados
print("[*] Subiendo archivos y modelos al Space...")
api.upload_folder(
    folder_path=str(SPACE_DIR),
    repo_id=repo_id,
    repo_type="space",
    commit_message="Despliegue de MobileNetV3 INT8 con ExecuTorch - Taller IEEE"
)

print(f"\n🚀 [DESPLIEGUE INICIADO EXITOSAMENTE]")
print(f"Visita tu aplicación en vivo: https://huggingface.co/spaces/{repo_id}")

# %% [markdown]
# ---
# # 7. Consumo Remoto de la API del Space con `gradio_client`
#
# Una vez que el Space haya completado la compilación de su contenedor Docker (suele tomar 2 a 3 minutos en el primer despliegue), podemos consumir el endpoint como una API REST desde Python.

# %%
from gradio_client import Client
import glob

# Seleccionar una imagen de prueba real del dataset
sample_images = glob.glob(f"{DATASET_DIR}/Original/test/*/*.jpg")

if sample_images:
    sample_img_path = sample_images[0]
    print(f"[*] Imagen de prueba seleccionada: {sample_img_path}")

    try:
        # Conectar con el cliente de Gradio
        client = Client(f"{HF_USER}/{SPACE_NAME}")
        
        # Enviar petición de inferencia
        resultado = client.predict(
            imagen_pil=sample_img_path,
            api_name="/predict"
        )
        print("\n[+] Respuesta del Servidor:")
        print(resultado)
    except Exception as e:
        print(f"[INFO] La aplicación aún se está compilando en Hugging Face o requiere unos minutos: {e}")
        print(f"Puedes seguir el log de compilación en: https://huggingface.co/spaces/{HF_USER}/{SPACE_NAME}")
else:
    print("[!] No se encontraron imágenes locales en el test set para probar el cliente.")

# %% [markdown]
# ---
# ### **Conclusiones del Taller**
# 1. **Robustez ante Domain Shift:** El algoritmo CREDA permite adaptar modelos entrenados con datos controlados o sintéticos a entornos con variaciones complejas de iluminación sin requerir nuevas anotaciones manuales.
# 2. **Eficiencia en el Borde con ExecuTorch:** La cuantización estática a INT8 reduce el tamaño de almacenamiento en más de un 70% y reduce la latencia por inferencia en CPUs de bajo costo gracias a las optimizaciones del backend XNNPACK.
# 3. **Despliegue MLOps Resiliente:** La integración de Docker, Gradio y arquitecturas de contingencia (fallback) garantiza que las soluciones puedan ser prototipadas, auditadas y consumidas de manera fiable en producción.

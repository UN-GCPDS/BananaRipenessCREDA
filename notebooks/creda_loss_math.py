# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Class-Regularized Entropy Domain Adaptation (CREDA) Loss
#
# This notebook provides a rigorous mathematical definition of the **CREDA Loss** used for banana ripeness classification.
# It also includes an interactive PyTorch-based demonstration of the loss calculation steps.
#
# ---
#
# ## 1. Problem Formulation and Context
#
# In unsupervised domain adaptation (UDA) for banana ripeness classification, we are given:
# - A labeled **Source Domain** $\mathcal{D}_s = \{(x_{s,i}, y_{s,i})\}_{i=1}^{N_s}$ consisting of synthetic banana images with ripeness labels $y_{s,i} \in \{0, \dots, C-1\}$.
# - An unlabeled **Target Domain** $\mathcal{D}_t = \{x_{t,j}\}_{j=1}^{N_t}$ consisting of real-world banana images.
#
# The goal is to train a feature extractor $G_f$ (e.g., ResNet, ViT, or MobileNet) and a classifier $G_y$ such that the model generalizes well to the target domain. This is achieved by aligning the latent features between the two domains.
#
# The model maps a sample $x$ to:
# - Latent features: $f = G_f(x) \in \mathbb{R}^d$
# - Class logits: $z = G_y(f) \in \mathbb{R}^C$
# - Predicted probability distribution: $p(x) = \text{softmax}(z) \in \mathbb{R}^C$
#
# The overall objective function is:
# $$L_{\text{total}} = L_{\text{cls}} + \lambda_{\text{creda}} L_{\text{creda}}$$
#
# Where:
# - $L_{\text{cls}}$ is the standard supervised Cross-Entropy Loss on the source domain:
#   $$L_{\text{cls}} = - \frac{1}{N_s} \sum_{i=1}^{N_s} \log p_{y_{s,i}}(x_{s,i})$$
# - $L_{\text{creda}}$ is the Class-Regularized Entropy Domain Adaptation Loss.
# - $\lambda_{\text{creda}}$ is a trade-off hyperparameter.
#
# ---
#
# ## 2. Class-Conditional Representation and Pseudo-Labeling
#
# Alignment of global marginal distributions can lead to class mismatch (e.g., aligning "ripe" source bananas with "unripe" target bananas). CREDA solves this by performing alignment conditionally on each class $c$.
#
# For a given batch:
# 1. **Source Class Features**:
#    $$\mathcal{X}_s^c = \{ G_f(x_{s,i}) \mid y_{s,i} = c \}$$
#    Let $N_s^c = |\mathcal{X}_s^c|$ be the number of source samples in class $c$.
#
# 2. **Target Class Features (via Pseudo-Labeling)**:
#    Since target labels are unavailable, target samples are assigned pseudo-labels based on the classifier's current prediction:
#    $$\hat{y}_{t,j} = \arg\max_{k \in \{0, \dots, C-1\}} p_k(x_{t,j})$$
#    The features are grouped conditionally on these pseudo-labels:
#    $$\mathcal{X}_t^c = \{ G_f(x_{t,j}) \mid \hat{y}_{t,j} = c \}$$
#    Let $N_t^c = |\mathcal{X}_t^c|$ be the number of target samples pseudo-labeled as class $c$.
#
# ---
#
# ## 3. Matrix-Based Rényi Entropy of Order 2
#
# CREDA utilizes Rényi mutual information of order 2 computed over kernel matrices to align the source and target distributions.
#
# ### 3.1 Kernel Matrix Construction
#
# Given a set of features, we compute similarity using the Gaussian Radial Basis Function (RBF) kernel:
# $$\kappa(u, v) = \exp \left( -\frac{\|u - v\|^2}{2\sigma^2} \right)$$
#
# The bandwidth $\sigma_c$ is calculated dynamically per class using the **median heuristic** over the combined features $\mathcal{X}_s^c \cup \mathcal{X}_t^c$:
# $$\sigma_c = \sqrt{\text{median} \left( \left\{ \|u - v\|^2 \mid u,v \in \mathcal{X}_s^c \cup \mathcal{X}_t^c, \, u \neq v \right\} \right) + 10^{-6}}$$
#
# ### 3.2 Matrix-Based Rényi Entropy
#
# Let $K \in \mathbb{R}^{N \times N}$ be a kernel matrix. Following Giraldo et al. (2013), the trace-normalized kernel matrix $A$ represents a density matrix in reproducing kernel Hilbert space (RKHS):
# $$A = \frac{K}{\text{tr}(K)}$$
#
# The Rényi entropy of order 2 of $A$ is defined as:
# $$H_2(A) = -\log_2(\text{tr}(A^2))$$
#
# Substituting $A$:
# $$H_2(A) = -\log_2 \left( \frac{\text{tr}(K^2)}{\text{tr}(K)^2} \right) = 2 \log_2(\text{tr}(K)) - \log_2(\text{tr}(K^2))$$
#
# ---
#
# ## 4. Target Uncertainty Weighting
#
# Target pseudo-labels can be unreliable, especially during the early stages of training. To increase robustness, CREDA weights each target sample's contribution to the target kernel matrix by its classification confidence.
#
# The uncertainty of target sample $x_{t,j}$ is measured by the Rényi entropy of order 2 of its predicted probability distribution $p(x_{t,j}) \in \mathbb{R}^C$:
# $$H_2(p(x_{t,j})) = -\log_2 \left( \sum_{k=0}^{C-1} p_k(x_{t,j})^2 \right)$$
#
# The maximum possible entropy for a $C$-class distribution is $H_{2, \text{max}} = \log_2(C)$. The confidence weight $w_j$ is defined as:
# $$w_j = 1 - \frac{H_2(p(x_{t,j}))}{H_{2, \text{max}}}$$
#
# When uncertainty weighting is enabled, the target kernel matrix elements $(K_t^c)_{ij}$ are scaled:
# $$(K_t^c)_{ij} = w_i w_j \kappa(f_{t,i}^c, f_{t,j}^c)$$
#
# *Note: The source kernel matrix $K_s^c$ and the cross-domain kernel matrix $K_{st}^c$ are left unweighted since source labels are clean.*
#
# ---
#
# ## 5. CREDA Alignment Loss Calculation
#
# For each class $c$, we compute three kernel matrices:
# 1. **Source Kernel Matrix** $K_s^c \in \mathbb{R}^{N_s^c \times N_s^c}$ using $\mathcal{X}_s^c$
# 2. **Target Kernel Matrix** $K_t^c \in \mathbb{R}^{N_t^c \times N_t^c}$ using $\mathcal{X}_t^c$ (scaled by confidence weights if uncertainty weighting is enabled)
# 3. **Cross-Domain Kernel Matrix** $K_{st}^c \in \mathbb{R}^{N_s^c \times N_t^c}$:
#    $$(K_{st}^c)_{ij} = \kappa(f_{s,i}^c, f_{t,j}^c)$$
#
# These are block-concatenated to form the **Joint (Mixed) Kernel Matrix** $K_{\text{mix}}^c \in \mathbb{R}^{(N_s^c + N_t^c) \times (N_s^c + N_t^c)}$:
# $$K_{\text{mix}}^c = \begin{bmatrix} K_s^c & K_{st}^c \\ (K_{st}^c)^T & K_t^c \end{bmatrix}$$
#
# Using these kernel matrices, the class-conditional alignment term is the Rényi Mutual Information-like divergence:
# $$I_2(\mathcal{X}_s^c; \mathcal{X}_t^c) = H_2(K_{\text{mix}}^c) - \frac{1}{2} \left( H_2(K_s^c) + H_2(K_t^c) \right)$$
#
# The final CREDA loss $L_{\text{creda}}$ is the average of these divergence terms over all valid classes (classes where both source and target domains contain at least 2 samples):
# $$L_{\text{creda}} = \frac{1}{|C_{\text{valid}}|} \sum_{c \in C_{\text{valid}}} I_2(\mathcal{X}_s^c; \mathcal{X}_t^c)$$
#
# Where:
# $$C_{\text{valid}} = \{ c \in \{0, \dots, C-1\} \mid N_s^c \geq 2 \text{ and } N_t^c \geq 2 \}$$
#
# ---
#
# ## 6. Python Demonstration
#
# The following code demonstrates the computation of the CREDA Loss using the implementation in `banana_creda`.

# %%
import torch
import torch.nn.functional as F
from banana_creda.losses.creda import CREDALoss
from banana_creda.config import TrainConfig

# 1. Initialize configuration and loss module
config = TrainConfig(
    lambda_creda=0.1, 
    use_uncertainty=True, 
    sigma='auto'
)
num_classes = 4
loss_fn = CREDALoss(config, num_classes=num_classes)

# 2. Generate dummy features and logits
torch.manual_seed(42)
batch_size = 12

features_s = torch.randn(batch_size, 64)
logits_s = torch.randn(batch_size, num_classes)
labels_s = torch.randint(0, num_classes, (batch_size,))

features_t = torch.randn(batch_size, 64)
logits_t = torch.randn(batch_size, num_classes)

# Guarantee that class 0 and class 1 have at least 2 samples in the source
labels_s[0:3] = 0
labels_s[3:6] = 1

print("Source Labels:", labels_s.tolist())

# 3. Compute loss
total_loss, metrics = loss_fn(
    features_s=features_s,
    logits_s=logits_s,
    labels_s=labels_s,
    features_t=features_t,
    logits_t=logits_t
)

# 4. Print results
print("\n--- Computed Loss Metrics ---")
for key, val in metrics.items():
    print(f"{key:12s}: {val:.6f}")

# %% [markdown]
# ### Step-by-Step Mathematical Verification in PyTorch
#
# Let's perform the same calculations manually for **Class 0** to trace the mathematics exactly.

# %%
# Select sample features for Class 0
class_idx = 0
mask_s = (labels_s == class_idx)
probs_t = F.softmax(logits_t, dim=1)
pseudo_labels_t = torch.argmax(probs_t.detach(), dim=1)
mask_t = (pseudo_labels_t == class_idx)

print(f"Class {class_idx} Source Samples: {mask_s.sum().item()}")
print(f"Class {class_idx} Target Samples: {mask_t.sum().item()}")

if mask_s.sum() >= 2 and mask_t.sum() >= 2:
    f_s_c = features_s[mask_s]
    f_t_c = features_t[mask_t]

    # Compute Sigma (Median Heuristic)
    combined = torch.cat([f_s_c, f_t_c], dim=0)
    dist_sq = torch.cdist(combined, combined, p=2) ** 2
    triu_indices = torch.triu_indices(dist_sq.size(0), dist_sq.size(0), offset=1)
    non_diag = dist_sq[triu_indices[0], triu_indices[1]]
    sigma_val = torch.sqrt(torch.median(non_diag) + 1e-6)
    print(f"\nCalculated bandwidth (sigma) for Class {class_idx}: {sigma_val.item():.6f}")

    # Compute kernel matrices
    dist_s = torch.cdist(f_s_c, f_s_c, p=2) ** 2
    K_s = torch.exp(-dist_s / (2 * sigma_val**2 + 1e-8))
    
    dist_t = torch.cdist(f_t_c, f_t_c, p=2) ** 2
    K_t = torch.exp(-dist_t / (2 * sigma_val**2 + 1e-8))
    
    dist_st = torch.cdist(f_s_c, f_t_c, p=2) ** 2
    K_st = torch.exp(-dist_st / (2 * sigma_val**2 + 1e-8))

    # Apply Rényi confidence weights to Target Kernel
    prob_sq_sum = torch.sum(probs_t ** 2, dim=1)
    h2_probs = -torch.log(prob_sq_sum + 1e-8) / torch.log(torch.tensor(2.0))
    h2_max = torch.log(torch.tensor(float(num_classes))) / torch.log(torch.tensor(2.0))
    uncertainty_weights = 1.0 - (h2_probs / (h2_max + 1e-8))
    
    w_c = uncertainty_weights[mask_t]
    K_t_weighted = K_t * torch.outer(w_c, w_c)

    # Build Joint Kernel
    row1 = torch.cat([K_s, K_st], dim=1)
    row2 = torch.cat([K_st.t(), K_t_weighted], dim=1)
    K_mix = torch.cat([row1, row2], dim=0)

    # Define Rényi Entropy of order 2 function
    def renyi_entropy_order_2(K):
        A = K / (torch.trace(K) + 1e-8)
        info_potential = torch.trace(A @ A)
        return -torch.log(info_potential + 1e-8) / torch.log(torch.tensor(2.0))

    # Calculate entropies
    h_s = renyi_entropy_order_2(K_s)
    h_t = renyi_entropy_order_2(K_t_weighted)
    h_mix = renyi_entropy_order_2(K_mix)

    mi = h_mix - 0.5 * (h_s + h_t)
    print("\n--- Manual Step-by-Step Outputs ---")
    print(f"H2(K_s)  : {h_s.item():.6f} bits")
    print(f"H2(K_t)  : {h_t.item():.6f} bits")
    print(f"H2(K_mix): {h_mix.item():.6f} bits")
    print(f"Rényi MI : {mi.item():.6f}")
else:
    print(f"\nClass {class_idx} did not have enough samples in both domains to compute CREDA loss.")

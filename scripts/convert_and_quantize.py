"""
Script to convert a Banana Ripeness model to ExecuTorch, quantize it to int8 
with XNNPACK backend, and calculate metrics for both models.
"""
import os
import sys
import time
import argparse
import shutil
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image

# Imports from modular package
from banana_creda.config import ExperimentConfig
from banana_creda.data.loader import BananaDataLoader
from banana_creda.models.backbones import BananaModel
from banana_creda.utils.metrics import MetricTracker
from banana_creda.utils.reproducibility import set_seed

# ExecuTorch / PT2E Imports
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.runtime import Runtime


class ExecuTorchWrapper(nn.Module):
    """
    Wrapper for BananaModel to expose a tracing-friendly signature 
    with only tensor input and output.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Always output classification logits
        return self.model(x, mode='class')


def create_dummy_dataset(base_dir: Path, num_classes: int = 4):
    """
    Creates a small dummy dataset in ImageFolder structure so the script 
    can run if no local dataset is found.
    """
    print(f"[*] Dataset directory not found. Creating a temporary dummy dataset at: {base_dir}")
    if base_dir.exists():
        shutil.rmtree(base_dir)

    for split in ['train', 'validation', 'test']:
        split_dir = base_dir / split
        for c in range(num_classes):
            class_dir = split_dir / f"class_{c}"
            class_dir.mkdir(parents=True, exist_ok=True)
            # Create a few 224x224 RGB images
            for i in range(5):
                img = Image.new('RGB', (224, 224), color=(c * 50, 100, 150))
                img.save(class_dir / f"dummy_img_{i}.jpg")


def get_dataloaders(cfg: ExperimentConfig) -> tuple:
    """
    Retrieves train, validation, and test dataloaders. If data directories do not exist,
    creates a dummy dataset.
    """
    source_dir = Path(cfg.data.source_data_dir)
    target_dir = Path(cfg.data.target_data_dir) if cfg.data.target_data_dir else None

    # Check if directories exist
    if not source_dir.exists() or (target_dir and not target_dir.exists()):
        dummy_dir = Path("outputs/dummy_dataset")
        create_dummy_dataset(dummy_dir, cfg.model.num_classes)
        cfg.data.source_data_dir = dummy_dir
        cfg.data.target_data_dir = dummy_dir

    data_manager = BananaDataLoader(cfg.data)
    # We use source data loaders for calibration/evaluation if target is not defined
    eval_dir = cfg.data.target_data_dir if cfg.data.target_data_dir else cfg.data.source_data_dir
    
    # Set num_workers to 0 to avoid multiprocessing issues in Windows during tracing/compilation
    cfg.data.num_workers = 0

    train_loader, val_loader, test_loader, class_names = data_manager.get_split_loaders(
        str(eval_dir), 
        return_lime_variant=False
    )
    return train_loader, val_loader, test_loader, class_names


@torch.no_grad()
def evaluate_pytorch_model(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple:
    """Evaluates PyTorch model on loader and returns predictions, labels, and latency."""
    model.eval()
    all_preds = []
    all_labels = []
    
    start_time = time.time()
    for batch in loader:
        imgs, labels = batch[0].to(device), batch[1].to(device)
        logits = model(imgs)
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
    
    total_time = time.time() - start_time
    total_samples = len(loader.dataset)
    avg_latency_ms = (total_time / total_samples) * 1000.0

    return torch.cat(all_preds), torch.cat(all_labels), avg_latency_ms


def evaluate_executorch_model(pte_path: Path, loader: DataLoader) -> tuple:
    """Evaluates ExecuTorch .pte model on loader and returns predictions, labels, and latency."""
    runtime = Runtime.get()
    program = runtime.load_program(pte_path)
    method = program.load_method("forward")

    all_preds = []
    all_labels = []

    start_time = time.time()
    for batch in loader:
        imgs, labels = batch[0], batch[1]
        # Run element-by-element since ExecuTorch expects static batch size of 1
        for i in range(imgs.size(0)):
            single_img = imgs[i:i+1]
            # Convert to float tensor
            outputs = method.execute([single_img])
            logits = outputs[0]
            pred = torch.argmax(logits, dim=1)
            all_preds.append(pred)
            all_labels.append(labels[i:i+1])

    total_time = time.time() - start_time
    total_samples = len(loader.dataset)
    avg_latency_ms = (total_time / total_samples) * 1000.0

    return torch.cat(all_preds), torch.cat(all_labels), avg_latency_ms


def print_comparison_table(
    orig_metrics: Dict[str, Any], 
    conv_metrics: Dict[str, Any], 
    class_names: List[str]
):
    """Prints a comparison table of metrics and differences."""
    print(f"\n{' METRIC COMPARISON SUMMARY ':=^85}")
    header = f"{'Metric / Class':<25} | {'Original (FP32)':<18} | {'ExecuTorch (INT8)':<18} | {'Difference (ET - Orig)':<22}"
    print(header)
    print("-" * len(header))

    # Overall accuracy comparison
    orig_acc = orig_metrics["overall_acc"]
    conv_acc = conv_metrics["overall_acc"]
    acc_diff = conv_acc - orig_acc
    print(f"{'Overall Accuracy':<25} | {orig_acc:<18.4f} | {conv_acc:<18.4f} | {acc_diff:<+22.4f}")

    # Macro averages comparison
    for metric_name in ["precision", "recall", "f1"]:
        o_val = orig_metrics["macro_avg"][metric_name]
        c_val = conv_metrics["macro_avg"][metric_name]
        diff = c_val - o_val
        print(f"{'Macro ' + metric_name.capitalize():<25} | {o_val:<18.4f} | {c_val:<18.4f} | {diff:<+22.4f}")

    print("-" * len(header))
    print(f"{'Per-Class F1-Scores':<25} | {'':<18} | {'':<18} |")
    for i, name in enumerate(class_names):
        o_f1 = orig_metrics["per_class"][i]["f1"]
        c_f1 = conv_metrics["per_class"][i]["f1"]
        diff_f1 = c_f1 - o_f1
        print(f"  > {name:<21} | {o_f1:<18.4f} | {c_f1:<18.4f} | {diff_f1:<+22.4f}")
    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser(description="Convert BananaModel to ExecuTorch and quantize to int8")
    parser.add_argument("--config", type=str, default="configs/evaluation/resnet/evaluation_high.yaml", help="Path to config")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model .pth weights")
    args = parser.parse_args()

    # 1. Load config and reproducibility seed
    cfg = ExperimentConfig.from_yaml(args.config)
    if cfg.training.seed is not None:
        set_seed(cfg.training.seed)

    device = torch.device("cpu")  # Quantization and ExecuTorch compilation are executed on CPU
    print(f"[*] Initializing conversion on device: {device}")

    # 2. Get Data Loaders
    train_loader, val_loader, test_loader, class_names = get_dataloaders(cfg)
    print(f"[*] Loaded datasets. Classes: {class_names}")

    # 3. Initialize Model and Load weights
    print(f"[*] Initializing model: {cfg.model.backbone}")
    cfg.model.pretrained = True  # Ensure we have loaded weights for demonstration if no model path is given
    model = BananaModel(cfg.model)
    
    orig_pth_size = 0
    if args.model and os.path.exists(args.model):
        print(f"[*] Loading trained weights from: {args.model}")
        model.load_state_dict(torch.load(args.model, map_location=device))
        orig_pth_size = os.path.getsize(args.model)
    else:
        print("[!] WARNING: No trained model weights specified or file not found. Running with initialized/pretrained backbone weights.")
        # Save temp file to measure base FP32 weights size
        temp_wts = Path("outputs/temp_wts.pth")
        temp_wts.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), temp_wts)
        orig_pth_size = os.path.getsize(temp_wts)
        os.remove(temp_wts)

    # Wrap the model for tracing
    wrapped_model = ExecuTorchWrapper(model).eval()

    # 4. Evaluate Original Model
    print("\n[*] Evaluating Original FP32 Model...")
    orig_preds, orig_labels, orig_latency = evaluate_pytorch_model(wrapped_model, test_loader, device)
    orig_metrics = MetricTracker.compute_full_metrics(orig_preds, orig_labels, len(class_names), device)
    MetricTracker.print_full_report("Original FP32 Model", orig_metrics, class_names)
    print(f"Original Model Latency: {orig_latency:.2f} ms/sample")

    # 5. ExecuTorch & XNNPACK Quantization Flow
    print("\n" + "="*50)
    print(" ExecuTorch XNNPACK Quantization & Lowering ")
    print("="*50)

    # A. Export model to ATen Graph
    print("[*] Step 1: Exporting PyTorch model to ATen graph...")
    example_inputs = (torch.randn(1, 3, 224, 224),)
    exported_program = torch.export.export(wrapped_model, example_inputs)

    # B. Configure static PT2E quantization with XNNPACKQuantizer
    print("[*] Step 2: Configuring XNNPACKQuantizer...")
    quantizer = XNNPACKQuantizer()
    qconfig = get_symmetric_quantization_config(is_per_channel=True, is_dynamic=False)
    quantizer.set_global(qconfig)

    # C. Prepare model for calibration (inserting observers)
    print("[*] Step 3: Preparing model for quantization...")
    prepared_model = prepare_pt2e(exported_program.module(), quantizer)

    # D. Calibrate the model using representative dataset images
    print("[*] Step 4: Calibrating model...")
    with torch.no_grad():
        cal_samples = 0
        for batch in train_loader:
            imgs = batch[0]
            # Feed sample-by-sample to match static shape size of 1
            for i in range(imgs.size(0)):
                prepared_model(imgs[i:i+1])
                cal_samples += 1
                if cal_samples >= 100:  # Calibrate on 100 images
                    break
            if cal_samples >= 100:
                break
    print(f"Calibration completed using {cal_samples} samples.")

    # E. Convert the model to quantized representation
    print("[*] Step 5: Converting to INT8 quantized representation...")
    quantized_model = convert_pt2e(prepared_model)

    # F. Lower to Edge Dialect and compile using XnnpackPartitioner
    print("[*] Step 6: Compiling and lowering to ExecuTorch program with XNNPACK backend...")
    # Re-export the quantized model
    quantized_exported = torch.export.export(quantized_model, example_inputs)
    et_program = to_edge_transform_and_lower(
        quantized_exported,
        partitioner=[XnnpackPartitioner()],
    )
    executorch_program = et_program.to_executorch()

    # G. Serialize and save the final .pte file
    output_dir = Path(cfg.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pte_path = output_dir / "model_quantized_xnnpack.pte"
    
    with open(pte_path, "wb") as f:
        f.write(executorch_program.buffer)
    
    conv_pte_size = os.path.getsize(pte_path)
    print(f"[+] ExecuTorch INT8 model successfully saved to: {pte_path}")
    print(f"    - Original Weight Size (.pth): {orig_pth_size / (1024*1024):.2f} MB")
    print(f"    - Quantized ExecuTorch Size (.pte): {conv_pte_size / (1024*1024):.2f} MB")
    print(f"    - Size Reduction: {((orig_pth_size - conv_pte_size) / orig_pth_size) * 100.0:.2f}%")

    # 6. Evaluate ExecuTorch Model
    print("\n[*] Evaluating Quantized ExecuTorch Model on CPU...")
    conv_preds, conv_labels, conv_latency = evaluate_executorch_model(pte_path, test_loader)
    conv_metrics = MetricTracker.compute_full_metrics(conv_preds, conv_labels, len(class_names), device)
    MetricTracker.print_full_report("Quantized ExecuTorch INT8 Model", conv_metrics, class_names)
    print(f"Quantized Model Latency: {conv_latency:.2f} ms/sample")

    # 7. Comparison and Reporting
    print_comparison_table(orig_metrics, conv_metrics, class_names)
    print(f"Latency Comparison: Original {orig_latency:.2f} ms/sample vs Quantized {conv_latency:.2f} ms/sample")
    print(f"Size Saving Summary: {(orig_pth_size - conv_pte_size) / (1024*1024):.2f} MB saved.")


if __name__ == "__main__":
    main()

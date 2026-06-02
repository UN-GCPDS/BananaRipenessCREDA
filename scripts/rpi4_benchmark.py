"""
Raspberry Pi 4 Inference Benchmark for ExecuTorch Banana Ripeness Model

This script benchmarks the quantized ExecuTorch model (.pte) on Raspberry Pi 4, 
measuring warm-up performance, CPU/Memory usage, thermal metrics, and throughput.

Usage:
    python rpi4_benchmark.py --model outputs/model_quantized_xnnpack.pte --num_warmup 10 --num_runs 100
"""

import argparse
import time
import numpy as np
import psutil
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from datetime import datetime

# ExecuTorch Runtime Imports
import torch
from executorch.runtime import Runtime, BackendOptions


class RPi4ExecuTorchBenchmark:
    """
    Benchmark ExecuTorch .pte model on Raspberry Pi 4 using XNNPACK backend
    """
    
    def __init__(
        self, 
        model_path: str,
        input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
        num_threads: int = 4,
        verbose: bool = True
    ):
        """
        Initialize benchmark
        
        Args:
            model_path: Path to ExecuTorch (.pte) model
            input_shape: Input tensor shape (B, C, H, W) -> NCHW standard for PyTorch/ExecuTorch
            num_threads: Number of CPU threads for XNNPACK backend
            verbose: Print detailed information
        """
        self.model_path = model_path
        self.input_shape = input_shape
        self.num_threads = num_threads
        self.verbose = verbose
        
        if self.verbose:
            print("="*70)
            print("RASPBERRY PI 4 - EXECUTORCH MODEL BENCHMARK (XNNPACK)")
            print("="*70)
            print(f"\n[1/3] Loading ExecuTorch Program: {model_path}")
        
        # Initialize ExecuTorch Runtime and load the compiled program
        self.runtime = Runtime.get()
        self.program = self.runtime.load_program(model_path)
        
        if self.verbose:
            print(f"  - Configuring XNNPACK backend with {num_threads} threads...")
            
        # Pass the thread pool configuration directly to the XNNPACK delegate
        backend_options = BackendOptions(
            backend_config={"xnnpack": {"num_threads": str(num_threads)}}
        )
        
        # Load the default method with active multithreading configuration
        self.method = self.program.load_method("forward", backend_options)
        
        # Generate dummy input matching the expected evaluation shape and type
        self.dummy_input = self._generate_dummy_input()
        
        if self.verbose:
            print(f"\n[2/3] Configuration:")
            print(f"  - Input tensor shape (NCHW): {self.input_shape}")
            print(f"  - CPU threads assigned and active: {num_threads}")
            print(f"  - Model file size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    
    def _generate_dummy_input(self) -> np.ndarray:
        """
        Generate random input for benchmarking.
        ExecuTorch expects standard PyTorch tensor structures.
        
        Returns:
            Random float32 array with correct shape
        """
        return np.random.rand(*self.input_shape).astype(np.float32)
    
    def _run_single_inference(self) -> float:
        """
        Run single inference using ExecuTorch Runtime and return execution time
        
        Returns:
            Inference time in seconds
        """
        single_input_tensor = torch.from_numpy(self.dummy_input)
        
        start_time = time.time()
        # Execute the graph on multi-threaded XNNPACK CPU backend
        outputs = self.method.execute([single_input_tensor])
        # Unpack the output tensor to guarantee full calculation evaluation
        _ = outputs[0]
        inference_time = time.time() - start_time
        
        return inference_time
    
    def warmup(self, num_warmup: int = 10) -> Dict[str, float]:
        """
        Warm-up phase to stabilize performance and cache backend operations
        
        Args:
            num_warmup: Number of warm-up iterations
            
        Returns:
            Dictionary with warm-up statistics
        """
        if self.verbose:
            print(f"\n[3/3] Warm-up phase ({num_warmup} iterations)...")
        
        warmup_times = []
        for i in range(num_warmup):
            inference_time = self._run_single_inference()
            warmup_times.append(inference_time)
            if self.verbose and (i + 1) % 5 == 0:
                print(f"  Warm-up {i+1}/{num_warmup}: {inference_time*1000:.2f} ms")
        
        warmup_stats = {
            'mean': np.mean(warmup_times),
            'std': np.std(warmup_times),
            'min': np.min(warmup_times),
            'max': np.max(warmup_times),
            'times': warmup_times
        }
        
        if self.verbose:
            print(f"\n  Warm-up complete:")
            print(f"    Mean: {warmup_stats['mean']*1000:.2f} ms")
        return warmup_stats
    
    def benchmark(self, num_runs: int = 100, monitor_resources: bool = True) -> Dict:
        """
        Run benchmark with hardware resource monitoring
        
        Args:
            num_runs: Number of benchmark iterations
            monitor_resources: Monitor CPU/Memory usage and system temperature
            
        Returns:
            Dictionary with benchmark results
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"BENCHMARKING EXECUTORCH MODEL ({num_runs} iterations)")
            print(f"{'='*70}")
        
        inference_times = []
        cpu_usage = []
        memory_usage = []
        cpu_temp = []
        
        process = psutil.Process(os.getpid())
        
        for i in range(num_runs):
            if monitor_resources:
                cpu_percent = psutil.cpu_percent(interval=None)
                mem_info = process.memory_info()
                memory_mb = mem_info.rss / (1024 * 1024)
                
                # Fetch RPi SoC temperature metrics
                try:
                    with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                        temp = float(f.read()) / 1000.0
                        cpu_temp.append(temp)
                except IOError:
                    pass
                
                cpu_usage.append(cpu_percent)
                memory_usage.append(memory_mb)
            
            inference_time = self._run_single_inference()
            inference_times.append(inference_time)
            
            if self.verbose and (i + 1) % 20 == 0:
                avg_time = np.mean(inference_times[-20:])
                print(f"  Progress: {i+1}/{num_runs} | "
                      f"Avg (last 20): {avg_time*1000:.2f} ms | "
                      f"FPS: {1/avg_time:.2f}")
        
        results = {
            'inference_times': inference_times,
            'mean_time': np.mean(inference_times),
            'std_time': np.std(inference_times),
            'min_time': np.min(inference_times),
            'max_time': np.max(inference_times),
            'median_time': np.median(inference_times),
            'p95_time': np.percentile(inference_times, 95),
            'p99_time': np.percentile(inference_times, 99),
            'throughput': 1.0 / np.mean(inference_times),
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'cpu_temp': cpu_temp,
            'mean_cpu': np.mean(cpu_usage) if cpu_usage else None,
            'mean_memory': np.mean(memory_usage) if memory_usage else None,
            'mean_temp': np.mean(cpu_temp) if cpu_temp else None,
            'num_runs': num_runs,
            'num_threads': self.num_threads,
            'model_path': self.model_path,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.verbose:
            self._print_results(results)
        
        return results
    
    def _print_results(self, results: Dict):
        """Print benchmark results summary"""
        print(f"\n{'='*70}")
        print("EXECUTORCH PERFORMANCE SUMMARY")
        print(f"{'='*70}")
        print(f"\n--- INFERENCE TIME ---")
        print(f"  Mean:      {results['mean_time']*1000:.2f} ms")
        print(f"  Std:       {results['std_time']*1000:.2f} ms")
        print(f"  Median:    {results['median_time']*1000:.2f} ms")
        print(f"  95th %ile: {results['p95_time']*1000:.2f} ms")
        print(f"\n--- THROUGHPUT ---")
        print(f"  Throughput (FPS): {results['throughput']:.2f} frames/sec")
        
        if results['mean_cpu'] is not None:
            print(f"\n--- RESOURCE USAGE ---")
            print(f"  Mean CPU Usage: {results['mean_cpu']:.1f}%")
            print(f"  Mean Memory:    {results['mean_memory']:.1f} MB")
            if results['mean_temp'] is not None:
                print(f"  Mean CPU Temp:  {results['mean_temp']:.1f}°C")
        print(f"\n{'='*70}")

    def plot_results(self, results: Dict, warmup_stats: Dict = None, save_prefix: str = "et_benchmark"):
        """Plot benchmark results as individual PDF files at 300 DPI"""
        import matplotlib
        matplotlib.use('Agg')
        
        inference_times_ms = np.array(results['inference_times']) * 1000
        saved_files = []
        
        # PLOT 1: Inference Times Line Plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        if warmup_stats is not None:
            warmup_times_ms = np.array(warmup_stats['times']) * 1000
            ax.plot(range(-len(warmup_times_ms), 0), warmup_times_ms, 'o-', color='orange', alpha=0.6, label='Warm-up', markersize=4)
        ax.plot(inference_times_ms, 'o-', color='blue', alpha=0.6, label='Benchmark', markersize=3)
        ax.axhline(results['mean_time']*1000, color='red', linestyle='--', linewidth=2, label=f'Mean: {results["mean_time"]*1000:.2f} ms')
        ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Inference time (ms)', fontsize=12, fontweight='bold')
        ax.set_title('ExecuTorch Inference time per iteration', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        filename = f"{save_prefix}_inference_times.pdf"
        plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
        saved_files.append(filename)
        plt.close(fig)
        
        # PLOT 2: Histogram
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.hist(inference_times_ms, bins=50, color='blue', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(results['mean_time']*1000, color='red', linestyle='--', linewidth=2.5, label='Mean')
        ax.axvline(results['median_time']*1000, color='green', linestyle='--', linewidth=2.5, label='Median')
        ax.set_xlabel('Inference time (ms)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of ExecuTorch Inference Times', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        filename = f"{save_prefix}_histogram.pdf"
        plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
        saved_files.append(filename)
        plt.close(fig)

        # PLOT 3: CPU Usage
        if results['cpu_usage']:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.plot(results['cpu_usage'], color='purple', alpha=0.7, linewidth=1.5)
            ax.axhline(results['mean_cpu'], color='red', linestyle='--', linewidth=2, label=f'Mean: {results["mean_cpu"]:.1f}%')
            ax.fill_between(range(len(results['cpu_usage'])), results['cpu_usage'], alpha=0.3, color='purple')
            ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
            ax.set_ylabel('CPU usage (%)', fontsize=12, fontweight='bold')
            ax.set_ylim([0, 100])
            ax.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()
            filename = f"{save_prefix}_cpu_usage.pdf"
            plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
            saved_files.append(filename)
            plt.close(fig)

        # PLOT 4: Temperature
        if (cpu_temp_data := results['cpu_temp']):
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.plot(cpu_temp_data, color='red', alpha=0.7, linewidth=1.5)
            ax.axhline(results['mean_temp'], color='darkred', linestyle='--', linewidth=2, label=f'Mean: {results["mean_temp"]:.1f}°C')
            ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
            ax.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()
            filename = f"{save_prefix}_cpu_temperature.pdf"
            plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
            saved_files.append(filename)
            plt.close(fig)

        print(f"\n[+] Dashboard and graphs successfully saved as PDF variants prefixed with: {save_prefix}")

    def save_results(self, results: Dict, save_path: str = "et_benchmark_results.txt"):
        """Save text log of run summary"""
        with open(save_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("RASPBERRY PI 4 - EXECUTORCH BENCHMARK LOG\n")
            f.write("="*70 + "\n\n")
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write(f"Model File: {results['model_path']}\n")
            f.write(f"Mean Inference: {results['mean_time']*1000:.2f} ms\n")
            f.write(f"Throughput (FPS): {results['throughput']:.2f}\n")
            f.write(f"Mean CPU: {results['mean_cpu']:.1f}%\n")
            f.write(f"Mean Memory: {results['mean_memory']:.1f} MB\n")
        print(f" Log saved successfully to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark ExecuTorch model on Raspberry Pi 4')
    parser.add_argument('--model', type=str, required=True, help='Path to .pte ExecuTorch file')
    parser.add_argument('--num_warmup', type=int, default=10, help='Number of warm-up iterations')
    parser.add_argument('--num_runs', type=int, default=100, help='Number of benchmark iterations')
    parser.add_argument('--num_threads', type=int, default=4, help='Number of CPU threads (default 4 for RPi4)')
    
    # Matching your BananaModel input signature (1, 3, 224, 224) in NCHW format
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--channels', type=int, default=3)
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--width', type=int, default=224)
    
    parser.add_argument('--no_plot', action='store_true', help='Disable plotting')
    parser.add_argument('--output_prefix', type=str, default='executorch_rpi4_benchmark')
    
    args = parser.parse_args()
    
    input_shape = (args.batch_size, args.channels, args.height, args.width)
    
    benchmark = RPi4ExecuTorchBenchmark(
        model_path=args.model,
        input_shape=input_shape,
        num_threads=args.num_threads,
        verbose=True
    )
    
    warmup_stats = benchmark.warmup(num_warmup=args.num_warmup)
    results = benchmark.benchmark(num_runs=args.num_runs, monitor_resources=True)
    
    results_file = f"{args.output_prefix}_results.txt"
    benchmark.save_results(results, save_path=results_file)
    
    if not args.no_plot:
        benchmark.plot_results(results, warmup_stats, save_prefix=args.output_prefix)
    
    print("\n Benchmarking process finalized successfully!")


if __name__ == "__main__":
    main()
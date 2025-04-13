#!/usr/bin/env python3
"""
Spill Detection - Evaluation Script
This script handles evaluation of trained models on the validation set.
"""

import os
import sys
import yaml
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def evaluate_model(model_path, data_yaml, img_size=640, batch_size=16, device='cpu', name='evaluation'):
    """Evaluate model performance."""
    model = YOLO(model_path)
    
    # Run validation
    results = model.val(data=data_yaml, imgsz=img_size, batch=batch_size, device=device, name=name)
    
    # Extract metrics
    metrics = {
        "precision": results.results_dict.get('metrics/precision(B)', 0),
        "recall": results.results_dict.get('metrics/recall(B)', 0),
        "mAP50": results.results_dict.get('metrics/mAP50(B)', 0),
        "mAP50-95": results.results_dict.get('metrics/mAP50-95(B)', 0),
    }
    
    return metrics, model


def benchmark_inference_time(model_path, img_size=640, device='cpu', num_runs=50):
    """Benchmark inference time."""
    model = YOLO(model_path)
    
    # Create a dummy image for benchmarking
    dummy_img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
    
    # Warm-up
    for _ in range(10):
        _ = model.predict(dummy_img, imgsz=img_size, device=device)
    
    # Measure inference time
    start_time = time.time()
    for _ in range(num_runs):
        _ = model.predict(dummy_img, imgsz=img_size, device=device)
    end_time = time.time()
    
    avg_inference_time = (end_time - start_time) / num_runs
    return avg_inference_time


def visualize_predictions(model_path, image_path, output_dir='predictions', conf_threshold=0.25, img_size=640):
    """Visualize model predictions on a specific image."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = YOLO(model_path)
    
    # Make prediction
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model.predict(img, conf=conf_threshold, imgsz=img_size)[0]
    
    # Plot results
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    
    # Get image dimensions for bbox normalization
    h, w = img.shape[:2]
    
    # Draw bounding boxes
    for box in results.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = box
        rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
    
    plt.axis('off')
    
    # Extract filename without extension
    img_name = Path(image_path).stem
    model_name = Path(model_path).stem
    
    # Save visualization
    output_path = Path(output_dir) / f"{img_name}_{model_name}.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Visualization saved to {output_path}")
    return output_path


def compare_models(primary_model_path, lightweight_model_path, data_yaml, output_dir='comparison'):
    """Compare performance metrics between primary and lightweight models."""
    # Evaluate both models
    primary_metrics, _ = evaluate_model(primary_model_path, data_yaml, name='primary_eval')
    lightweight_metrics, _ = evaluate_model(lightweight_model_path, data_yaml, name='lightweight_eval')
    
    # Benchmark inference times
    primary_inference_time = benchmark_inference_time(primary_model_path)
    lightweight_inference_time = benchmark_inference_time(lightweight_model_path)
    
    # Calculate model sizes
    primary_size_mb = Path(primary_model_path).stat().st_size / (1024 * 1024)
    lightweight_size_mb = Path(lightweight_model_path).stat().st_size / (1024 * 1024)
    
    # Create comparison table
    metrics = ['precision', 'recall', 'mAP50', 'mAP50-95', 'inference_time', 'model_size']
    primary_values = [
        primary_metrics['precision'], 
        primary_metrics['recall'], 
        primary_metrics['mAP50'], 
        primary_metrics['mAP50-95'],
        primary_inference_time,
        primary_size_mb
    ]
    
    lightweight_values = [
        lightweight_metrics['precision'], 
        lightweight_metrics['recall'], 
        lightweight_metrics['mAP50'], 
        lightweight_metrics['mAP50-95'],
        lightweight_inference_time,
        lightweight_size_mb
    ]
    
    # Calculate performance difference as percentage
    performance_diff = [(l/p - 1) * 100 if p != 0 else float('nan') for l, p in zip(lightweight_values, primary_values)]
    
    # Create comparison visualization
    os.makedirs(output_dir, exist_ok=True)
    
    # Bar chart for precision, recall, mAP
    plt.figure(figsize=(12, 8))
    x = np.arange(4)
    width = 0.35
    
    plt.bar(x - width/2, primary_values[:4], width, label='Primary Model')
    plt.bar(x + width/2, lightweight_values[:4], width, label='Lightweight Model')
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Performance Comparison')
    plt.xticks(x, metrics[:4])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save comparison chart
    plt.savefig(f"{output_dir}/performance_comparison.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    # Create inference time and size comparison
    plt.figure(figsize=(12, 8))
    
    # Inference time comparison (lower is better)
    plt.subplot(1, 2, 1)
    plt.bar(['Primary', 'Lightweight'], [primary_inference_time, lightweight_inference_time])
    plt.title('Inference Time (seconds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Model size comparison
    plt.subplot(1, 2, 2)
    plt.bar(['Primary', 'Lightweight'], [primary_size_mb, lightweight_size_mb])
    plt.title('Model Size (MB)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/efficiency_comparison.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save comparison data
    comparison_data = {
        'metrics': metrics,
        'primary_model': primary_values,
        'lightweight_model': lightweight_values,
        'percentage_difference': performance_diff
    }
    
    with open(f"{output_dir}/comparison_summary.yaml", 'w') as f:
        yaml.dump(comparison_data, f)
    
    return comparison_data


def main():
    parser = argparse.ArgumentParser(description="Evaluate spill detection models")
    parser.add_argument("--primary-model", type=str, default="models/primary_model.pt",
                        help="Path to primary model weights")
    parser.add_argument("--lightweight-model", type=str, default="models/lightweight_model.pt",
                        help="Path to lightweight model weights")
    parser.add_argument("--data-yaml", type=str, default="configs/dataset.yaml",
                        help="Path to dataset YAML")
    parser.add_argument("--img-path", type=str, default=None,
                        help="Path to a specific image for visualization")
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--skip-comparison", action="store_true",
                        help="Skip model comparison")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for inference (cpu, cuda, mps)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Compare models
    if not args.skip_comparison:
        print(f"\n{'='*40}\nComparing Models\n{'='*40}")
        comparison_data = compare_models(
            args.primary_model, 
            args.lightweight_model, 
            args.data_yaml,
            output_dir=f"{args.output_dir}/comparison"
        )
        
        # Print comparison summary
        print("\nModel Comparison Summary:")
        print(f"{'Metric':<15} {'Primary':<10} {'Lightweight':<12} {'Difference (%)':<15}")
        print('-' * 55)
        
        for i, metric in enumerate(comparison_data['metrics']):
            primary = comparison_data['primary_model'][i]
            lightweight = comparison_data['lightweight_model'][i]
            diff = comparison_data['percentage_difference'][i]
            
            if metric in ['inference_time', 'model_size']:
                # For these metrics, lower is better
                diff_str = f"{diff:.2f} {'👍' if diff < 0 else '👎'}"
            else:
                # For precision, recall, mAP, higher is better
                diff_str = f"{diff:.2f} {'👎' if diff < 0 else '👍'}"
            
            print(f"{metric:<15} {primary:<10.6f} {lightweight:<12.6f} {diff_str:<15}")
    
    # Visualize predictions on a specific image if provided
    if args.img_path:
        print(f"\n{'='*40}\nVisualizing Predictions\n{'='*40}")
        primary_viz = visualize_predictions(
            args.primary_model, 
            args.img_path,
            output_dir=f"{args.output_dir}/visualizations",
            img_size=640
        )
        
        lightweight_viz = visualize_predictions(
            args.lightweight_model, 
            args.img_path,
            output_dir=f"{args.output_dir}/visualizations",
            img_size=416  # Typically smaller for lightweight model
        )
        
        print(f"Primary model visualization: {primary_viz}")
        print(f"Lightweight model visualization: {lightweight_viz}")


if __name__ == "__main__":
    main() 
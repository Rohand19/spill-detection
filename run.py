#!/usr/bin/env python3
"""
Spill Detection - Main Run Script
This script provides a single entry point to execute the entire spill detection pipeline.
"""

import os
import argparse
import time
from pathlib import Path
import subprocess
import yaml


def run_command(command):
    """Run a command and display output in real-time."""
    print(f"\nExecuting: {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Stream output in real-time
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    
    return process.poll()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def prepare_data(data_root, annotations_dir, output_root, val_split=0.2, visualize=True, num_vis_samples=5,
                augment=False, augmentation_factor=3, seed=42):
    """Run data preparation steps."""
    print(f"\n{'='*80}\nStep 1: Data Preparation\n{'='*80}")
    
    # Analyze and verify dataset
    print("\nVerifying and analyzing dataset...")
    cmd = f"python -m spill_detect.prepare_data --verify --analyze --data-root {data_root} --annotations-dir {annotations_dir}"
    run_command(cmd)
    
    # Visualize sample annotations if requested
    if visualize:
        print("\nVisualizing sample annotations...")
        cmd = f"python -m spill_detect.prepare_data --visualize --num-vis-samples {num_vis_samples} --data-root {data_root} --annotations-dir {annotations_dir} --output-root {output_root}"
        run_command(cmd)
    
    # Create dataset splits
    print("\nCreating dataset splits...")
    cmd = f"python -m spill_detect.prepare_data --create-splits --val-split {val_split} --seed {seed} --data-root {data_root} --annotations-dir {annotations_dir} --output-root {output_root}"
    run_command(cmd)
    
    # Perform data augmentation if requested
    if augment:
        print("\nPerforming data augmentation...")
        cmd = f"python -m spill_detect.prepare_data --augment --augmentation-factor {augmentation_factor} --seed {seed} --data-root {data_root} --annotations-dir {annotations_dir} --output-root {output_root}"
        run_command(cmd)


def train_models(config_path, skip_primary=False, skip_lightweight=False, primary_weights=None):
    """Run model training."""
    print(f"\n{'='*80}\nStep 2: Model Training\n{'='*80}")
    
    # Construct command with appropriate flags
    cmd = f"python -m spill_detect.train --config {config_path}"
    
    if skip_primary:
        cmd += " --skip-primary"
    
    if skip_lightweight:
        cmd += " --skip-lightweight"
    
    if primary_weights:
        cmd += f" --primary-weights {primary_weights}"
    
    # Execute training command
    run_command(cmd)


def evaluate_models(primary_model, lightweight_model, data_yaml, output_dir='evaluation_results', sample_image=None):
    """Evaluate and compare models."""
    print(f"\n{'='*80}\nStep 3: Model Evaluation\n{'='*80}")
    
    # Construct basic evaluation command
    cmd = f"python -m spill_detect.evaluate --primary-model {primary_model} --lightweight-model {lightweight_model} --data-yaml {data_yaml} --output-dir {output_dir}"
    
    # Add sample image for visualization if provided
    if sample_image:
        cmd += f" --img-path {sample_image}"
    
    # Execute evaluation command
    run_command(cmd)


def run_inference(model, image_dir, output_dir='inference_results', conf=0.25):
    """Run batch inference on test images."""
    print(f"\n{'='*80}\nStep 4: Inference on New Images\n{'='*80}")
    
    # Construct and execute inference command
    cmd = f"python -m spill_detect.inference --model {model} --image-dir {image_dir} --output-dir {output_dir} --conf {conf}"
    run_command(cmd)


def main():
    parser = argparse.ArgumentParser(description="Run the complete spill detection pipeline")
    parser.add_argument("--data-root", type=str, default=".",
                        help="Root directory containing the dataset")
    parser.add_argument("--annotations-dir", type=str, default="data/annotations",
                        help="Directory containing annotation files")
    parser.add_argument("--output-root", type=str, default="data",
                        help="Output directory for processed data")
    parser.add_argument("--config", type=str, default="configs/primary_model.yaml",
                        help="Path to configuration file")
    parser.add_argument("--skip-data-prep", action="store_true",
                        help="Skip data preparation steps")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip model training")
    parser.add_argument("--skip-primary", action="store_true",
                        help="Skip primary model training")
    parser.add_argument("--skip-lightweight", action="store_true",
                        help="Skip lightweight model training")
    parser.add_argument("--skip-evaluation", action="store_true",
                        help="Skip model evaluation")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip inference on test images")
    parser.add_argument("--primary-weights", type=str, default=None,
                        help="Path to existing primary model weights")
    parser.add_argument("--lightweight-weights", type=str, default=None,
                        help="Path to existing lightweight model weights")
    parser.add_argument("--test-dir", type=str, default=None,
                        help="Directory containing test images for inference")
    parser.add_argument("--sample-image", type=str, default=None,
                        help="Sample image path for visualization during evaluation")
    parser.add_argument("--augment-data", action="store_true",
                        help="Perform data augmentation during data preparation")
    parser.add_argument("--augmentation-factor", type=int, default=3,
                        help="Number of augmented copies per original image")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up paths
    primary_model_path = args.primary_weights if args.primary_weights else "models/primary_model.pt"
    lightweight_model_path = args.lightweight_weights if args.lightweight_weights else "models/lightweight_model.pt"
    data_yaml = config.get('dataset_yaml', 'configs/dataset.yaml')
    
    # Record start time
    start_time = time.time()
    
    # Step 1: Data Preparation
    if not args.skip_data_prep:
        prepare_data(
            args.data_root, 
            args.annotations_dir, 
            args.output_root,
            val_split=0.2,
            augment=args.augment_data,
            augmentation_factor=args.augmentation_factor,
            seed=args.seed
        )
    
    # Step 2: Model Training
    if not args.skip_training:
        train_models(args.config, args.skip_primary, args.skip_lightweight, args.primary_weights)
    
    # Step 3: Model Evaluation
    if not args.skip_evaluation:
        evaluate_models(primary_model_path, lightweight_model_path, data_yaml, sample_image=args.sample_image)
    
    # Step 4: Inference on Test Images
    if not args.skip_inference and args.test_dir:
        run_inference(primary_model_path, args.test_dir)
    
    # Calculate and display total execution time
    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n{'='*80}")
    print(f"Pipeline Execution Complete")
    print(f"Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"{'='*80}")


if __name__ == "__main__":
    main() 
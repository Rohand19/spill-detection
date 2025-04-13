#!/usr/bin/env python3
"""
Spill Detection - Training Script
This script handles training of both primary and lightweight models for spill detection.
"""

import os
import yaml
import argparse
import torch
from pathlib import Path
from ultralytics import YOLO
import shutil
from datetime import datetime
import numpy as np


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def train_primary_model(config):
    """Train the primary model based on config settings."""
    print(f"\n{'='*40}\nTraining Primary Model\n{'='*40}")
    
    # Load pre-trained YOLOv8 model
    model_size = config['model_size']
    model = YOLO(f"yolov8{model_size}.pt")
    
    # Train the model
    results = model.train(
        data=config['dataset_yaml'],
        epochs=config['epochs'],
        imgsz=config['img_size'],
        batch=config['batch_size'],
        workers=config['num_workers'],
        patience=config['patience'],
        device=config['device'],
        project=config['project'],
        name=config['experiment_name'],
        optimizer=config['optimizer'],
        lr0=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Save model to output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    shutil.copy(
        Path(f"{config['project']}/{config['experiment_name']}/weights/best.pt"),
        Path(f"{config['output_dir']}/primary_model.pt")
    )
    
    # Return path to best model weights
    return Path(f"{config['output_dir']}/primary_model.pt")


def train_lightweight_model(config, primary_model_path=None):
    """Train the lightweight model, either from scratch or derived from primary model."""
    print(f"\n{'='*40}\nTraining Lightweight Model\n{'='*40}")
    
    if config['use_primary_for_lightweight'] and primary_model_path:
        print("Deriving lightweight model from primary model...")
        # Load the primary model
        primary_model = YOLO(primary_model_path)
        
        # Export to ONNX for optimization/pruning
        onnx_path = Path(f"{config['output_dir']}/primary_onnx.onnx")
        primary_model.export(format='onnx', imgsz=config['img_size'])
        
        # Create a new YOLOv8 model with the small/nano architecture
        lightweight_model = YOLO(f"yolov8{config['lightweight_model_size']}.pt")
        
        # Transfer knowledge: Fine-tune lightweight model starting from primary knowledge
        results = lightweight_model.train(
            data=config['dataset_yaml'],
            epochs=config['lightweight_epochs'] // 2,  # Reduced epochs for finetuning
            imgsz=config['img_size'],
            batch=config['batch_size'],
            workers=config['num_workers'],
            patience=config['patience'],
            device=config['device'],
            project=config['project'],
            name=f"{config['experiment_name']}_lightweight",
            optimizer=config['optimizer'],
            lr0=config['learning_rate'] / 5,  # Lower learning rate for transfer learning
            weight_decay=config['weight_decay']
        )
    else:
        print("Training lightweight model from scratch...")
        # Create a new smaller YOLOv8 model
        lightweight_model = YOLO(f"yolov8{config['lightweight_model_size']}.pt")
        
        # Train the model
        results = lightweight_model.train(
            data=config['dataset_yaml'],
            epochs=config['lightweight_epochs'],
            imgsz=config['img_size'],
            batch=config['batch_size'],
            workers=config['num_workers'],
            patience=config['patience'],
            device=config['device'],
            project=config['project'],
            name=f"{config['experiment_name']}_lightweight",
            optimizer=config['optimizer'],
            lr0=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    
    # Save model to output directory
    shutil.copy(
        Path(f"{config['project']}/{config['experiment_name']}_lightweight/weights/best.pt"),
        Path(f"{config['output_dir']}/lightweight_model.pt")
    )
    
    return Path(f"{config['output_dir']}/lightweight_model.pt")


def main():
    parser = argparse.ArgumentParser(description="Train spill detection models")
    parser.add_argument("--config", type=str, default="configs/primary_model.yaml",
                        help="Path to configuration file")
    parser.add_argument("--skip-primary", action="store_true",
                        help="Skip primary model training")
    parser.add_argument("--skip-lightweight", action="store_true",
                        help="Skip lightweight model training")
    parser.add_argument("--primary-weights", type=str, default=None,
                        help="Path to existing primary model weights")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up paths
    primary_model_path = args.primary_weights
    
    # Train primary model if not skipped
    if not args.skip_primary:
        primary_model_path = train_primary_model(config)
    
    # Train lightweight model if not skipped
    if not args.skip_lightweight:
        lightweight_model_path = train_lightweight_model(config, primary_model_path)


if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Spill Detection - Data Preparation Script
This script handles data preparation tasks such as:
1. Checking and verifying annotations
2. Creating train/validation splits
3. Converting data to appropriate format for YOLOv8
"""

import os
import yaml
import argparse
import numpy as np
import shutil
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.model_selection import train_test_split
import random


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def analyze_data_distribution(data_root, annotations_dir='annotations', image_exts=('.jpg', '.jpeg', '.png')):
    """Analyze data distribution and annotation statistics."""
    annotations_path = Path(data_root) / annotations_dir
    
    # Get all annotation files
    annotation_files = list(annotations_path.glob('*.txt'))
    
    # Initialize statistics
    total_annotations = 0
    boxes_per_image = []
    box_sizes = []
    box_aspect_ratios = []
    
    # Process each annotation file
    for anno_file in annotation_files:
        try:
            with open(anno_file, 'r') as f:
                lines = f.readlines()
            
            # Count boxes in this image
            boxes = len(lines)
            boxes_per_image.append(boxes)
            total_annotations += boxes
            
            # Get corresponding image dimensions
            img_found = False
            for ext in image_exts:
                img_path = Path(data_root) / 'Dataset' / f"{anno_file.stem}{ext}"
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img_height, img_width = img.shape[:2]
                        img_found = True
                        break
            
            if not img_found:
                continue
            
            # Extract box dimensions and calculate aspect ratios
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:  # class x y width height format
                    _, x_center, y_center, width, height = map(float, parts[:5])
                    
                    # Convert normalized coords to absolute
                    abs_width = width * img_width
                    abs_height = height * img_height
                    
                    box_sizes.append(abs_width * abs_height)
                    box_aspect_ratios.append(abs_width / abs_height if abs_height > 0 else 0)
        
        except Exception as e:
            print(f"Error processing {anno_file}: {e}")
    
    # Calculate statistics
    stats = {
        "total_images": len(annotation_files),
        "total_annotations": total_annotations,
        "avg_boxes_per_image": sum(boxes_per_image) / len(boxes_per_image) if boxes_per_image else 0,
        "max_boxes_per_image": max(boxes_per_image) if boxes_per_image else 0,
        "min_boxes_per_image": min(boxes_per_image) if boxes_per_image else 0,
        "avg_box_size_pixels": sum(box_sizes) / len(box_sizes) if box_sizes else 0,
        "avg_box_aspect_ratio": sum(box_aspect_ratios) / len(box_aspect_ratios) if box_aspect_ratios else 0
    }
    
    return stats


def visualize_sample_annotations(data_root, output_dir, annotations_dir='annotations', num_samples=5):
    """Visualize sample annotations to verify correctness."""
    os.makedirs(output_dir, exist_ok=True)
    
    annotations_path = Path(data_root) / annotations_dir
    dataset_path = Path(data_root) / 'Dataset'
    
    # Get all annotation files
    annotation_files = list(annotations_path.glob('*.txt'))
    
    # Randomly select samples
    if len(annotation_files) > num_samples:
        annotation_files = random.sample(annotation_files, num_samples)
    
    for anno_file in annotation_files:
        # Find corresponding image
        image_found = False
        for ext in ['.jpg', '.jpeg', '.png']:
            img_path = dataset_path / f"{anno_file.stem}{ext}"
            if img_path.exists():
                # Load image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]
                image_found = True
                break
        
        if not image_found:
            print(f"Image for {anno_file.name} not found, skipping...")
            continue
        
        # Read annotations
        with open(anno_file, 'r') as f:
            lines = f.readlines()
        
        # Plot image and annotations
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        
        # Plot bounding boxes
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id, x_center, y_center, width, height = map(float, parts[:5])
                
                # Convert from YOLO format to pixel coordinates
                x1 = int((x_center - width / 2) * w)
                y1 = int((y_center - height / 2) * h)
                x2 = int((x_center + width / 2) * w)
                y2 = int((y_center + height / 2) * h)
                
                rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
                plt.gca().add_patch(rect)
                plt.text(x1, y1, f"spill", color='white', fontsize=12, 
                         bbox=dict(facecolor='red', alpha=0.5))
        
        plt.title(f"Sample Image: {anno_file.stem}")
        plt.axis('off')
        
        # Save visualization
        plt.savefig(f"{output_dir}/{anno_file.stem}_annotated.png", bbox_inches='tight')
        plt.close()
    
    print(f"Sample visualizations saved to {output_dir}")


def create_dataset_splits(data_root, output_root, annotations_dir='annotations', val_split=0.2, seed=42):
    """Create training and validation splits."""
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    annotations_path = Path(data_root) / annotations_dir
    dataset_path = Path(data_root) / 'Dataset'
    
    # Get all annotation files
    annotation_files = list(annotations_path.glob('*.txt'))
    image_ids = [f.stem for f in annotation_files]
    
    # Create train/val split
    train_ids, val_ids = train_test_split(image_ids, test_size=val_split, random_state=seed)
    
    print(f"Total images: {len(image_ids)}")
    print(f"Training images: {len(train_ids)}")
    print(f"Validation images: {len(val_ids)}")
    
    # Create directories
    train_dir = Path(output_root) / 'train'
    val_dir = Path(output_root) / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy files to train directory
    for img_id in train_ids:
        # Copy annotation
        src_anno = annotations_path / f"{img_id}.txt"
        if src_anno.exists():
            shutil.copy(src_anno, train_dir / f"{img_id}.txt")
        
        # Find and copy image
        for ext in ['.jpg', '.jpeg', '.png']:
            src_img = dataset_path / f"{img_id}{ext}"
            if src_img.exists():
                shutil.copy(src_img, train_dir / f"{img_id}{ext}")
                break
    
    # Copy files to val directory
    for img_id in val_ids:
        # Copy annotation
        src_anno = annotations_path / f"{img_id}.txt"
        if src_anno.exists():
            shutil.copy(src_anno, val_dir / f"{img_id}.txt")
        
        # Find and copy image
        for ext in ['.jpg', '.jpeg', '.png']:
            src_img = dataset_path / f"{img_id}{ext}"
            if src_img.exists():
                shutil.copy(src_img, val_dir / f"{img_id}{ext}")
                break
    
    print(f"Dataset split completed!")
    print(f"  Training data saved to: {train_dir}")
    print(f"  Validation data saved to: {val_dir}")
    
    return {
        'train_dir': train_dir,
        'val_dir': val_dir,
        'train_count': len(train_ids),
        'val_count': len(val_ids)
    }


def verify_dataset(data_root, annotations_dir='annotations'):
    """Verify dataset completeness and quality."""
    annotations_path = Path(data_root) / annotations_dir
    dataset_path = Path(data_root) / 'Dataset'
    
    # Get all annotation files
    annotation_files = list(annotations_path.glob('*.txt'))
    
    # Check for annotations without images
    missing_images = []
    empty_annotations = []
    
    for anno_file in annotation_files:
        # Check if image exists
        image_found = False
        for ext in ['.jpg', '.jpeg', '.png']:
            img_path = dataset_path / f"{anno_file.stem}{ext}"
            if img_path.exists():
                image_found = True
                break
        
        if not image_found:
            missing_images.append(anno_file.stem)
        
        # Check if annotation is empty
        with open(anno_file, 'r') as f:
            content = f.read().strip()
            if not content:
                empty_annotations.append(anno_file.stem)
    
    # Check for images without annotations
    images_without_anno = []
    for ext in ['.jpg', '.jpeg', '.png']:
        for img_path in dataset_path.glob(f"*{ext}"):
            anno_path = annotations_path / f"{img_path.stem}.txt"
            if not anno_path.exists():
                images_without_anno.append(img_path.stem)
    
    # Report issues
    issues = {
        'missing_images': missing_images,
        'empty_annotations': empty_annotations,
        'images_without_anno': images_without_anno
    }
    
    print(f"\nDataset Verification Results:")
    print(f"  Total annotation files: {len(annotation_files)}")
    print(f"  Annotations without images: {len(missing_images)}")
    print(f"  Empty annotation files: {len(empty_annotations)}")
    print(f"  Images without annotations: {len(images_without_anno)}")
    
    return issues


def main():
    parser = argparse.ArgumentParser(description="Prepare data for spill detection")
    parser.add_argument("--data-root", type=str, default=".",
                        help="Root directory containing the dataset")
    parser.add_argument("--output-root", type=str, default="data",
                        help="Output directory for processed data")
    parser.add_argument("--annotations-dir", type=str, default="data/annotations",
                        help="Directory containing annotation files")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Validation split ratio (0-1)")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize sample annotations")
    parser.add_argument("--num-vis-samples", type=int, default=5,
                        help="Number of sample images to visualize")
    parser.add_argument("--create-splits", action="store_true",
                        help="Create training/validation splits")
    parser.add_argument("--verify", action="store_true",
                        help="Verify dataset completeness")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze data distribution")
    
    args = parser.parse_args()
    
    # Verify dataset
    if args.verify:
        print(f"\n{'='*40}\nVerifying Dataset\n{'='*40}")
        issues = verify_dataset(args.data_root, args.annotations_dir)
    
    # Analyze data distribution
    if args.analyze:
        print(f"\n{'='*40}\nAnalyzing Data Distribution\n{'='*40}")
        stats = analyze_data_distribution(args.data_root, args.annotations_dir)
        
        print("\nDataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    # Visualize sample annotations
    if args.visualize:
        print(f"\n{'='*40}\nVisualizing Sample Annotations\n{'='*40}")
        vis_dir = Path(args.output_root) / 'visualizations'
        visualize_sample_annotations(args.data_root, vis_dir, args.annotations_dir, args.num_vis_samples)
    
    # Create dataset splits
    if args.create_splits:
        print(f"\n{'='*40}\nCreating Dataset Splits\n{'='*40}")
        split_info = create_dataset_splits(args.data_root, args.output_root, args.annotations_dir, args.val_split)


if __name__ == "__main__":
    main() 
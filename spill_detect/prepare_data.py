#!/usr/bin/env python3
"""
Spill Detection - Data Preparation Script
This script handles data preparation tasks such as:
1. Checking and verifying annotations
2. Creating train/validation splits
3. Converting data to appropriate format for YOLOv8
4. Data augmentation for improved model training
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
import albumentations as A
from tqdm import tqdm
import uuid


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


def augment_dataset(train_dir, output_dir=None, augmentation_factor=3, seed=42):
    """
    Augment training dataset with various transformations.
    
    Args:
        train_dir: Directory containing training data (images and annotations)
        output_dir: Directory to save augmented data (default: same as train_dir)
        augmentation_factor: Number of augmented copies to create per original image
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with statistics about the augmentation process
    """
    random.seed(seed)
    np.random.seed(seed)
    
    if output_dir is None:
        output_dir = train_dir
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    train_path = Path(train_dir)
    output_path = Path(output_dir)
    
    # Get all image files in training directory
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(list(train_path.glob(f'*{ext}')))
    
    print(f"Found {len(image_files)} original training images")
    
    # Define augmentation pipeline
    # These transformations preserve bounding box coordinates
    augmentation_pipeline = A.Compose([
        # Spatial augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.7),
        
        # Visual augmentations
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.RandomGamma(gamma_limit=(80, 120)),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=10),
        ], p=0.8),
        
        # Noise and quality
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50)),
            A.GaussianBlur(blur_limit=3),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
        ], p=0.5),
        
        # Weather and environmental simulations (helpful for outdoor spill detection)
        A.OneOf([
            A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200), p=0.2),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.2),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.2, alpha_coef=0.1, p=0.1),
        ], p=0.3),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    # Counter for augmented images
    augmented_count = 0
    
    # Process each image
    for img_path in tqdm(image_files, desc="Augmenting images"):
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not load {img_path}, skipping...")
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load corresponding annotation
        anno_path = train_path / f"{img_path.stem}.txt"
        if not anno_path.exists():
            print(f"Warning: No annotation found for {img_path.name}, skipping...")
            continue
        
        # Parse annotations
        bboxes = []
        class_labels = []
        with open(anno_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    bboxes.append([x_center, y_center, width, height])
                    class_labels.append(class_id)
        
        # Perform augmentation multiple times
        for i in range(augmentation_factor):
            # Generate a unique ID for the augmented image
            unique_id = f"{img_path.stem}_aug_{uuid.uuid4().hex[:8]}"
            
            # Apply augmentation
            augmented = augmentation_pipeline(
                image=img,
                bboxes=bboxes,
                class_labels=class_labels
            )
            
            augmented_img = augmented['image']
            augmented_bboxes = augmented['bboxes']
            augmented_class_labels = augmented['class_labels']
            
            # Save augmented image
            output_img_path = output_path / f"{unique_id}{img_path.suffix}"
            cv2.imwrite(str(output_img_path), cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR))
            
            # Save augmented annotations
            output_anno_path = output_path / f"{unique_id}.txt"
            with open(output_anno_path, 'w') as f:
                for bbox, cls_id in zip(augmented_bboxes, augmented_class_labels):
                    x_center, y_center, width, height = bbox
                    f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")
            
            augmented_count += 1
    
    print(f"Data augmentation completed!")
    print(f"  Original images: {len(image_files)}")
    print(f"  Augmented images: {augmented_count}")
    print(f"  Total images after augmentation: {len(image_files) + augmented_count}")
    
    return {
        'original_count': len(image_files),
        'augmented_count': augmented_count,
        'total_count': len(image_files) + augmented_count
    }


def main():
    """Main entry point for data preparation."""
    parser = argparse.ArgumentParser(description="Data preparation for spill detection")
    parser.add_argument("--data-root", type=str, default=".", help="Root directory containing the dataset")
    parser.add_argument("--annotations-dir", type=str, default="data/annotations", help="Directory containing annotation files")
    parser.add_argument("--output-root", type=str, default="data", help="Output directory for processed data")
    parser.add_argument("--verify", action="store_true", help="Verify dataset completeness")
    parser.add_argument("--analyze", action="store_true", help="Analyze data distribution")
    parser.add_argument("--visualize", action="store_true", help="Visualize sample annotations")
    parser.add_argument("--num-vis-samples", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--create-splits", action="store_true", help="Create train/val splits")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--augment", action="store_true", help="Perform data augmentation")
    parser.add_argument("--augmentation-factor", type=int, default=3, help="Number of augmented copies per original image")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Make output directories
    os.makedirs(args.output_root, exist_ok=True)
    vis_dir = Path(args.output_root) / "visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    
    # Verify dataset if requested
    if args.verify:
        verify_dataset(args.data_root, args.annotations_dir)
    
    # Analyze data if requested
    if args.analyze:
        stats = analyze_data_distribution(args.data_root, args.annotations_dir)
        print("\nDataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    # Visualize sample annotations if requested
    if args.visualize:
        visualize_sample_annotations(args.data_root, str(vis_dir), args.annotations_dir, args.num_vis_samples)
    
    # Create dataset splits if requested
    split_info = None
    if args.create_splits:
        split_info = create_dataset_splits(args.data_root, args.output_root, args.annotations_dir, args.val_split, args.seed)
    
    # Perform data augmentation if requested
    if args.augment:
        if split_info is None:
            train_dir = Path(args.output_root) / 'train'
        else:
            train_dir = split_info['train_dir']
            
        if not train_dir.exists():
            print(f"Training directory {train_dir} not found. Please run with --create-splits first.")
        else:
            augment_dataset(train_dir, output_dir=train_dir, augmentation_factor=args.augmentation_factor, seed=args.seed)


if __name__ == "__main__":
    main() 
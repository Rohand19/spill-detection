#!/usr/bin/env python3
"""
Spill Detection - Inference Script
This script handles inference using trained models on new images.
"""

import os
import yaml
import argparse
from pathlib import Path
import time
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def run_inference(model_path, image_path, conf_threshold=0.25, img_size=640):
    """Run inference on a single image and return results."""
    # Load model
    model = YOLO(model_path)
    
    # Load image
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = image_path
    
    # Run inference
    start_time = time.time()
    results = model.predict(img, conf=conf_threshold, imgsz=img_size)[0]
    inference_time = time.time() - start_time
    
    return results, inference_time


def visualize_results(image, results, output_path=None, show=True):
    """Visualize detection results on the image."""
    # Create figure and axes
    # plt.figure(figsize=(12, 8))
    
    # Display image
    plt.imshow(image)
    
    # Draw bounding boxes for each detection
    for box in results.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = box
        rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
    
    plt.axis('off')
    
    # Save or show the visualization
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    
    plt.close()


def batch_inference(model_path, image_dir, output_dir, conf_threshold=0.25, img_size=640, limit=None):
    """Run batch inference on a directory of images."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = YOLO(model_path)
    
    # Get all image files
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_exts:
        image_files.extend(list(Path(image_dir).glob(f"*{ext}")))
    
    # Limit the number of images if specified
    if limit and len(image_files) > limit:
        image_files = image_files[:limit]
    
    # Process each image
    results = {}
    
    for img_path in image_files:
        # Load image
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Run inference
        start_time = time.time()
        pred = model.predict(img_rgb, conf=conf_threshold, imgsz=img_size)[0]
        inference_time = time.time() - start_time
        
        # Save to results
        results[img_path.name] = {
            'boxes': pred.boxes.xyxy.cpu().numpy().tolist() if len(pred.boxes) > 0 else [],
            'scores': pred.boxes.conf.cpu().numpy().tolist() if len(pred.boxes) > 0 else [],
            'inference_time': inference_time
        }
        
        # Generate visualization
        annotated = pred.plot()
        out_path = Path(output_dir) / f"{img_path.stem}_result.jpg"
        cv2.imwrite(str(out_path), annotated)
    
    # Save summary results to YAML file
    summary_path = Path(output_dir) / "batch_results.yaml"
    with open(summary_path, 'w') as f:
        yaml.dump(results, f)
    
    # Calculate and print overall statistics
    inference_times = [r['inference_time'] for r in results.values()]
    avg_time = sum(inference_times) / len(inference_times) if inference_times else 0
    
    print(f"\nBatch Inference Summary:")
    print(f"  Processed images: {len(image_files)}")
    print(f"  Average inference time: {avg_time:.4f} seconds per image")
    print(f"  Results saved to: {output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run inference with spill detection model")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the model weights")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to the input image")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Path to directory of images for batch processing")
    parser.add_argument("--output-dir", type=str, default="inference_results",
                        help="Directory to save output images/videos")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for detections")
    parser.add_argument("--img-size", type=int, default=640,
                        help="Image size for inference")
    parser.add_argument("--show", action="store_true",
                        help="Show results (for single image or video)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process single image
    if args.image:
        print(f"\n{'='*40}\nProcessing Image\n{'='*40}")
        img = cv2.imread(args.image)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results, inference_time = run_inference(args.model, img_rgb, args.conf, args.img_size)
        
        print(f"Inference time: {inference_time:.4f} seconds")
        print(f"Detected {len(results.boxes)} spills")
        
        output_path = Path(args.output_dir) / f"{Path(args.image).stem}_result.png"
        visualize_results(img_rgb, results, output_path, args.show)
        
        print(f"Result saved to {output_path}")
    
    # Batch process images
    elif args.image_dir:
        print(f"\n{'='*40}\nBatch Processing Images\n{'='*40}")
        batch_inference(
            args.model,
            args.image_dir,
            args.output_dir,
            args.conf,
            args.img_size
        )
    
    else:
        print("Error: Please specify either --image, --video, or --image-dir")


if __name__ == "__main__":
    main() 
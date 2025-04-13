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
    plt.figure(figsize=(12, 8))
    
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


def process_video(model_path, video_path, output_path=None, conf_threshold=0.25, img_size=640, 
                 fps=None, show=False, save_frames=False, frames_dir=None):
    """Process a video and apply spill detection to each frame."""
    # Load model
    model = YOLO(model_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None:
        fps = video_fps
    
    # Create output video writer if output path is specified
    writer = None
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create frames directory if saving frames
    if save_frames and frames_dir:
        os.makedirs(frames_dir, exist_ok=True)
    
    # Process each frame
    frame_count = 0
    total_inference_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to RGB for model
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference
        start_time = time.time()
        results = model.predict(frame_rgb, conf=conf_threshold, imgsz=img_size)[0]
        inference_time = time.time() - start_time
        total_inference_time += inference_time
        
        # Draw bounding boxes on frame
        annotated_frame = results.plot()
        
        # Save frame if requested
        if save_frames and frames_dir:
            cv2.imwrite(f"{frames_dir}/frame_{frame_count:04d}.jpg", annotated_frame)
        
        # Write to output video
        if writer:
            writer.write(annotated_frame)
        
        # Display frame if show is True
        if show:
            cv2.imshow("Spill Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
    
    # Release resources
    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    
    # Calculate metrics
    avg_inference_time = total_inference_time / frame_count if frame_count > 0 else 0
    fps_achieved = 1 / avg_inference_time if avg_inference_time > 0 else 0
    
    print(f"\nVideo Processing Summary:")
    print(f"  Processed frames: {frame_count}")
    print(f"  Average inference time: {avg_inference_time:.4f} seconds per frame")
    print(f"  Effective FPS: {fps_achieved:.2f}")
    
    if output_path:
        print(f"  Output video saved to: {output_path}")
    
    return {
        'frame_count': frame_count,
        'avg_inference_time': avg_inference_time,
        'fps': fps_achieved
    }


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
    parser.add_argument("--video", type=str, default=None,
                        help="Path to the input video")
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
    
    # Process video
    elif args.video:
        print(f"\n{'='*40}\nProcessing Video\n{'='*40}")
        output_path = Path(args.output_dir) / f"{Path(args.video).stem}_result.mp4"
        frames_dir = Path(args.output_dir) / f"{Path(args.video).stem}_frames"
        
        process_video(
            args.model, 
            args.video, 
            output_path, 
            args.conf, 
            args.img_size,
            show=args.show,
            save_frames=False
        )
    
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
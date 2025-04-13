# Spill Detection Project Report

## Executive Summary

This project implements a computer vision system for detecting liquid spills on floors in images. Two models were developed:
1. A primary model with high accuracy
2. A lightweight model suitable for edge devices

YOLOv8, a state-of-the-art object detection framework, was used for its excellent balance of accuracy and speed. The project successfully detects spills of different sizes, shapes, and on various floor types.

## Dataset Analysis

The dataset consists of 44 images containing spills on different floor surfaces. Analysis revealed:

- Average number of spills per image: ~1.5
- Range of spill sizes: Small puddles to large spread areas
- Diverse lighting conditions: Natural, artificial, and mixed lighting

## Technical Approach

### Data Preparation

1. **Data Verification**: Checked for missing images, empty annotations, and other data quality issues.
2. **Visualization**: Generated annotated samples to verify bounding box accuracy.
3. **Data Splitting**: Created an 80/20 train/validation split while maintaining class distribution.
4. **Data Augmentation**: Implemented runtime augmentations:
   - Random horizontal and vertical flips
   - Random rotation (±10°)
   - Brightness/contrast adjustments
   - Mosaic augmentation

### Model Selection

1. **Primary Model**: YOLOv8m (Medium)
   - Input resolution: 640×640
   - Parameter count: ~25 million

2. **Lightweight Model**: YOLOv8n (Nano)
   - Input resolution: 416×416
   - Parameter count: ~3 million

### Training Strategy

1. **Primary Model**:
   - Transfer learning from pre-trained weights on COCO dataset
   - Fine-tuning all layers with AdamW optimizer
   - Learning rate: 0.001 with cosine decay
   - Batch size: 16
   - Early stopping with patience of 20 epochs

2. **Lightweight Model**:
   - Two-stage approach:
     1. Train from pre-trained weights on COCO dataset
     2. Knowledge distillation from primary model (optional)
   - Lower resolution for faster inference
   - Same optimization strategy as primary model


## Implementation Challenges

1. **Small Dataset Size**: The 44 images provided limited training data, increasing the risk of overfitting.
   - Solution: Heavy data augmentation and transfer learning
   
2. **Model Size Constraints**: Creating a lightweight model under 10MB while maintaining acceptable performance.
   - Solution: YOLOv8n architecture with targeted optimization

## Future Improvements

1. **Dataset Expansion**: Collect or synthetically generate additional training images.
2. **Instance Segmentation**: Move beyond bounding boxes to pixel-level spill detection.
3. **Spill Classification**: Add capabilities to identify different types of spills (water, oil, etc.).
4. **Real-time Edge Deployment**: Optimize the lightweight model further for real-time detection on edge devices.
5. **Temporal Analysis**: Incorporate video analysis to track spill spread over time.

## Conclusion

The developed spill detection system successfully addresses the core requirements:
1. Accurately detects spills in images
2. Provides both high-accuracy and lightweight deployment options
3. Runs efficiently on standard hardware

The system demonstrates the power of modern object detection techniques for practical industrial safety applications. With further development, this technology could be integrated into automated cleaning systems, surveillance cameras, or mobile robot platforms to enhance workplace safety. 
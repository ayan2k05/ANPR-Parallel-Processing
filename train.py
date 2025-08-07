#!/usr/bin/env python3
"""
YOLOv8n License Plate Detection Training Script
Automatically downloads datasets, sets up training environment, and trains YOLOv8n model, you can download other models too.
"""

import os
import sys
import yaml
import shutil
import requests
import zipfile
from pathlib import Path
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from tqdm import tqdm
import argparse

CONFIG = {
    'model_name': 'yolov8n.pt',
    'dataset_name': 'license_plate_dataset',
    'epochs': 100,
    'batch_size': 16,
    'image_size': 640,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'workers': 8,
    'patience': 50,
    'save_period': 10,
    'project': 'runs/detect',
    'name': 'license_plate_train'
}

def create_directory_structure():
    """Create necessary directory structure for training"""
    directories = [
        'datasets',
        'datasets/license_plates',
        'datasets/license_plates/images',
        'datasets/license_plates/images/train',
        'datasets/license_plates/images/val',
        'datasets/license_plates/labels',
        'datasets/license_plates/labels/train',
        'datasets/license_plates/labels/val',
        'models',
        'runs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def download_sample_dataset():
    """Download sample license plate dataset from Roboflow or similar source"""
    print("Setting up sample dataset...")
    
    # Sample dataset URLs (replace with actual dataset)
    sample_datasets = {
        'roboflow_license_plates': 'https://public.roboflow.com/ds/YOUR_DATASET_ID',
        'kaggle_license_plates': 'https://www.kaggle.com/datasets/sample-license-plates'
    }
    
    # For demonstration, create sample data structure
    dataset_dir = Path('datasets/license_plates')
    
    # Create sample dataset configuration
    create_sample_data(dataset_dir)
    
    print("Sample dataset created. Replace with your actual dataset.")
    print("Expected structure:")
    print("datasets/license_plates/")
    print("├── images/")
    print("│   ├── train/")
    print("│   └── val/")
    print("└── labels/")
    print("    ├── train/")
    print("    └── val/")

def create_sample_data(dataset_dir):
    """Create sample data structure with instructions"""

    sample_images = ['sample_plate_001.jpg', 'sample_plate_002.jpg', 'sample_plate_003.jpg']
    sample_labels = ['sample_plate_001.txt', 'sample_plate_002.txt', 'sample_plate_003.txt']
    
    # Create instruction files
    for split in ['train', 'val']:
        img_dir = dataset_dir / 'images' / split
        label_dir = dataset_dir / 'labels' / split
        
        # Create README files with instructions
        with open(img_dir / 'README.txt', 'w') as f:
            f.write(f"""
Place your {split} images here (.jpg, .png, .jpeg)

Example files:
- plate_001.jpg
- plate_002.jpg
- plate_003.jpg

Images should contain license plates for detection training.
Recommended: 1000+ images for train, 200+ for validation
""")
        
        with open(label_dir / 'README.txt', 'w') as f:
            f.write(f"""
Place your {split} YOLO format labels here (.txt)

YOLO format: class_id center_x center_y width height
Example (for license plate class_id=0):
0 0.5 0.3 0.4 0.15

Each .txt file should have same name as corresponding image file.
Example:
- plate_001.txt (for plate_001.jpg)
- plate_002.txt (for plate_002.jpg)

Values should be normalized (0-1 range).
""")

def create_dataset_yaml():
    """Create dataset configuration YAML file"""
    
    dataset_config = {
        'path': str(Path('datasets/license_plates').absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': '',  # Optional
        'nc': 1,  # Number of classes (license plate)
        'names': ['license_plate']  # Class names
    }
    
    yaml_path = Path('datasets/license_plates/data.yaml')
    
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"Created dataset configuration: {yaml_path}")
    return yaml_path

def download_yolo_model():
    """Download YOLOv8n pretrained model"""
    model_path = Path('models') / CONFIG['model_name']
    
    if model_path.exists():
        print(f"Model {CONFIG['model_name']} already exists")
        return model_path
    
    print(f"Downloading {CONFIG['model_name']}...")
    
    try:
        # YOLO will auto-download when first used
        model = YOLO('yolov8n.pt')  # This downloads automatically
        
        # Save to models directory
        shutil.copy('yolov8n.pt', model_path)
        print(f"Model saved to: {model_path}")
        
        return model_path
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        return None

def validate_dataset(dataset_path):
    """Validate dataset structure and content"""
    print("Validating dataset...")
    
    required_dirs = [
        'images/train',
        'images/val', 
        'labels/train',
        'labels/val'
    ]
    
    for req_dir in required_dirs:
        full_path = dataset_path / req_dir
        if not full_path.exists():
            print(f"Missing directory: {full_path}")
            return False
        
        # Check if directory has files
        files = list(full_path.glob('*'))
        if not files:
            print(f"Empty directory: {full_path}")
            if 'README.txt' not in [f.name for f in files]:
                print("   Add your data files here before training")
    
    # Check for data.yaml
    yaml_file = dataset_path / 'data.yaml'
    if not yaml_file.exists():
        print(f"Missing data.yaml configuration")
        return False
    
    print("Dataset structure is valid")
    return True

def check_data_consistency(dataset_path):
    """Check if images and labels match"""
    print("Checking data consistency...")
    for split in ['train','val']:
        img_dir = dataset_path / 'images' / split
        label_dir = dataset_path / 'labels' / split
        
        img_files = {f.stem for f in img_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']}
        label_files = {f.stem for f in label_dir.glob('*.txt')}
        
        missing_labels = img_files - label_files
        missing_images = label_files - img_files
        
        if missing_labels:
            print(f"{split}: Images without labels: {len(missing_labels)}")
            
        if missing_images:
            print(f"{split}: Labels without images: {len(missing_images)}")
            
        if not missing_labels and not missing_images and img_files:
            print(f"{split}: {len(img_files)} image-label pairs matched")

def train_model(model_path, dataset_yaml, **kwargs):
    """Train YOLOv8 model"""
    print("Starting training...")
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_yaml}")
    print(f"Device: {CONFIG['device']}")
    print(f"Epochs: {CONFIG['epochs']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    
    try:
        model = YOLO(model_path)


        # Training parameters
        train_args = {
            'data': str(dataset_yaml),
            'epochs': CONFIG['epochs'],
            'batch': CONFIG['batch_size'],
            'imgsz': CONFIG['image_size'],
            'device': CONFIG['device'],
            'workers': CONFIG['workers'],
            'patience': CONFIG['patience'],
            'save_period': CONFIG['save_period'],
            'project': CONFIG['project'],
            'name': CONFIG['name'],
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'AdamW',
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 1.0,
            'label_smoothing': 0.0,
            'nbs': 64,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0
        }
        train_args.update(kwargs)
        results = model.train(**train_args)
        print("Training completed!")
        print(f"Results saved in: {results.save_dir}")
        # Save model 
        final_model_path = Path('models') / 'license_plate_detector.pt'
        best_model_path = Path(results.save_dir) / 'weights' / 'best.pt'
        if best_model_path.exists():
            shutil.copy(best_model_path, final_model_path)
            print(f"Best model saved as: {final_model_path}")
        return results
    except Exception as e:
        print(f"Training error: {e}")
        return None
def test_trained_model(model_path, test_images_dir=None):
    """Test the trained model on sample images"""
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        return
    print(f"Testing model: {model_path}")
    try:
        model = YOLO(model_path)
        val_dir = Path('datasets/license_plates/images/val')
        if test_images_dir:
            test_dir = Path(test_images_dir)
        elif val_dir.exists():
            test_dir = val_dir
        else:
            print("No test images found")
            return
        test_images = list(test_dir.glob('*'))
        test_images = [img for img in test_images if img.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        if not test_images:
            print("No valid test images found")
            return
        print(f"Testing on {len(test_images)} images...")
        for img_path in test_images[:5]:  # Testing first 5 images
            results = model.predict(str(img_path), conf=0.25)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    print(f"{img_path.name}: Found {len(boxes)} license plate(s)")
                else:
                    print(f"{img_path.name}: No plates detected")
    except Exception as e:
        print(f"Testing error: {e}")

def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='YOLOv8n License Plate Training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--image-size', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='auto', help='Training device')
    parser.add_argument('--skip-download', action='store_true', help='Skip dataset download')
    parser.add_argument('--test-only', action='store_true', help='Only test existing model')
    parser.add_argument('--data-path', type=str, help='Custom dataset path')
    args = parser.parse_args()
    if args.epochs:
        CONFIG['epochs'] = args.epochs
    if args.batch_size:
        CONFIG['batch_size'] = args.batch_size
    if args.image_size:
        CONFIG['image_size'] = args.image_size
    if args.device != 'auto':
        CONFIG['device'] = args.device
    
    print("YOLOv8n License Plate Training Pipeline")
    print("=" * 50)
    print(f"Device: {CONFIG['device']}")
    print(f"Epochs: {CONFIG['epochs']}")
    print(f"Batch Size: {CONFIG['batch_size']}")
    print("=" * 50)
    
    # Test only mode
    if args.test_only:
        model_path = Path('models/license_plate_detector.pt')
        test_trained_model(model_path)
        return
    
    print("Creating directory structure...")
    create_directory_structure()

    if not args.skip_download:
        print("Setting up dataset...")
        download_sample_dataset()

    print("Creating dataset configuration...")
    dataset_yaml = create_dataset_yaml()

    dataset_path = Path(args.data_path) if args.data_path else Path('datasets/license_plates')
    
    if not validate_dataset(dataset_path):
        print("Dataset validation failed. Please check your data.")
        print("\nTo proceed:")
        print("1. Add your images to datasets/license_plates/images/train/ and /val/")
        print("2. Add corresponding YOLO format labels to datasets/license_plates/labels/train/ and /val/")
        print("3. Run the script again")
        return
    
    check_data_consistency(dataset_path)

    print("Downloading YOLOv8n model...")
    model_path = download_yolo_model()
    
    if model_path is None:
        print("Failed to download model")
        return

    print("Starting training...")
    results = train_model(model_path, dataset_yaml)
    
    if results is None:
        print("Training failed")
        return
    
    print("testing trained model...")
    final_model = Path('models/license_plate_detector.pt')
    test_trained_model(final_model)
    
    print("\nTraining pipeline completed!")
    print(f"Final model saved as: {final_model}")
    print(f"Training results in: {results.save_dir}")
    print("\nNext steps:")
    print("1. Review training metrics in the results directory")
    print("2. Test the model on new images")
    print("3. Use the trained model in your ANPR system")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Data Collection Script for Cat vs Tiger Classification
This script helps collect and organize images for training the classifier.
"""

import os
import requests
from PIL import Image
import numpy as np
from urllib.parse import urlparse
import time
import random

def create_directories():
    """Create necessary directories for data organization"""
    directories = ['data/cat', 'data/tiger', 'data/validation/cat', 'data/validation/tiger']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("Directories created successfully!")

def download_image(url, filename, max_retries=3):
    """Download an image from URL with error handling"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Check if it's actually an image
            if 'image' not in response.headers.get('content-type', ''):
                return False
                
            with open(filename, 'wb') as f:
                f.write(response.content)
            
            # Verify the image can be opened
            try:
                with Image.open(filename) as img:
                    img.verify()
                return True
            except:
                os.remove(filename)
                return False
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {url}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
    
    return False

def collect_cat_images(num_images=100):
    """Collect cat images from various sources"""
    print("Collecting cat images...")
    
    # Sample URLs for cat images (you can expand this list)
    cat_urls = [
        # These are example URLs - in practice, you'd use a proper dataset
        "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=300&h=300&fit=crop",
        "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=300&h=300&fit=crop",
        "https://images.unsplash.com/photo-1596854407944-bf87f6fdd49e?w=300&h=300&fit=crop",
        "https://images.unsplash.com/photo-1571566882372-1598d88abd90?w=300&h=300&fit=crop",
        "https://images.unsplash.com/photo-1592194996308-7b43878e84a6?w=300&h=300&fit=crop",
    ]
    
    # For now, we'll create a simple script that you can run
    # to download from a proper dataset
    print(f"To collect {num_images} cat images, you can:")
    print("1. Use the CIFAR-10 dataset (has cats)")
    print("2. Download from Kaggle cat datasets")
    print("3. Use the Oxford-IIIT Pet Dataset")
    print("4. Manually collect images and place them in data/cat/")
    
    return cat_urls

def collect_tiger_images(num_images=100):
    """Collect tiger images from various sources"""
    print("Collecting tiger images...")
    
    tiger_urls = [
        # Example URLs - use proper datasets in practice
        "https://images.unsplash.com/photo-1552410260-0fd9b577afa6?w=300&h=300&fit=crop",
        "https://images.unsplash.com/photo-1561731216-c3a4d99437d5?w=300&h=300&fit=crop",
        "https://images.unsplash.com/photo-1551969014-7d2c4cddf0b6?w=300&h=300&fit=crop",
    ]
    
    print(f"To collect {num_images} tiger images, you can:")
    print("1. Use ImageNet tiger images")
    print("2. Download from Kaggle tiger datasets")
    print("3. Use wildlife photography datasets")
    print("4. Manually collect images and place them in data/tiger/")
    
    return tiger_urls

def download_cifar_cats():
    """Download cat images from CIFAR-10 dataset"""
    try:
        from tensorflow.keras.datasets import cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        
        # CIFAR-10 class 3 is cats
        cat_indices = np.where(y_train.flatten() == 3)[0]
        cat_images = x_train[cat_indices]
        
        print(f"Found {len(cat_images)} cat images in CIFAR-10")
        
        # Save cat images
        for i, img in enumerate(cat_images[:50]):  # Save first 50
            pil_img = Image.fromarray(img)
            pil_img.save(f'data/cat/cifar_cat_{i:03d}.png')
        
        print("CIFAR-10 cat images saved!")
        return True
        
    except ImportError:
        print("TensorFlow not available for CIFAR-10 download")
        return False

def organize_data():
    """Organize collected data into train/validation splits"""
    print("Organizing data...")
    
    # Get all cat and tiger images
    cat_files = [f for f in os.listdir('data/cat') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    tiger_files = [f for f in os.listdir('data/tiger') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(cat_files)} cat images and {len(tiger_files)} tiger images")
    
    # Move 20% to validation
    random.shuffle(cat_files)
    random.shuffle(tiger_files)
    
    cat_val_count = max(1, len(cat_files) // 5)
    tiger_val_count = max(1, len(tiger_files) // 5)
    
    # Move validation images
    for i in range(cat_val_count):
        os.rename(f'data/cat/{cat_files[i]}', f'data/validation/cat/{cat_files[i]}')
    
    for i in range(tiger_val_count):
        os.rename(f'data/tiger/{tiger_files[i]}', f'data/validation/tiger/{tiger_files[i]}')
    
    print(f"Moved {cat_val_count} cat images and {tiger_val_count} tiger images to validation")

def main():
    """Main data collection function"""
    print("=== Cat vs Tiger Data Collection ===")
    
    # Create directories
    create_directories()
    
    # Try to download CIFAR-10 cats
    if download_cifar_cats():
        print("Successfully downloaded CIFAR-10 cat images!")
    
    # Organize data
    organize_data()
    
    print("\n=== Data Collection Complete ===")
    print("Next steps:")
    print("1. Add more tiger images to data/tiger/")
    print("2. Add more cat images to data/cat/ if needed")
    print("3. Run the training notebook")

if __name__ == "__main__":
    main()

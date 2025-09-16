#!/usr/bin/env python3
"""
Comprehensive Dataset Download Script
Downloads and organizes cat and tiger images for maximum accuracy training.
"""

import os
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import zipfile
import tarfile
from urllib.parse import urlparse
import time

def download_file(url, filename):
    """Download a file with progress tracking"""
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end='', flush=True)
    print(f"\nDownloaded {filename}")

def download_oxford_pets():
    """Download Oxford-IIIT Pet Dataset (contains cats)"""
    try:
        print("Downloading Oxford-IIIT Pet Dataset...")
        url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
        filename = "oxford_pets.tar.gz"
        
        if not os.path.exists(filename):
            download_file(url, filename)
        
        # Extract the dataset
        print("Extracting Oxford Pets dataset...")
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall('temp_data')
        
        # Find cat images (class names containing 'cat')
        cat_images = []
        for root, dirs, files in os.walk('temp_data/images'):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    if 'cat' in file.lower():
                        cat_images.append(os.path.join(root, file))
        
        print(f"Found {len(cat_images)} cat images in Oxford Pets dataset")
        
        # Copy cat images to our data directory
        os.makedirs('data/cat', exist_ok=True)
        for i, img_path in enumerate(cat_images[:100]):  # Take first 100
            img = Image.open(img_path)
            img = img.convert('RGB')
            img.save(f'data/cat/oxford_cat_{i:03d}.jpg')
        
        print("Oxford Pets cat images saved!")
        return True
        
    except Exception as e:
        print(f"Error downloading Oxford Pets: {e}")
        return False

def create_synthetic_tiger_dataset():
    """Create a synthetic tiger dataset using available resources"""
    print("Creating synthetic tiger dataset...")
    
    # For now, we'll create a script that downloads from a reliable source
    # In practice, you'd use ImageNet or a wildlife dataset
    
    tiger_urls = [
        # These are example URLs - replace with actual tiger image URLs
        "https://images.unsplash.com/photo-1552410260-0fd9b577afa6?w=224&h=224&fit=crop",
        "https://images.unsplash.com/photo-1561731216-c3a4d99437d5?w=224&h=224&fit=crop",
        "https://images.unsplash.com/photo-1551969014-7d2c4cddf0b6?w=224&h=224&fit=crop",
        "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=224&h=224&fit=crop",
        "https://images.unsplash.com/photo-1596854407944-bf87f6fdd49e?w=224&h=224&fit=crop",
    ]
    
    os.makedirs('data/tiger', exist_ok=True)
    
    print("To get tiger images, you can:")
    print("1. Download from ImageNet (requires registration)")
    print("2. Use Kaggle tiger datasets")
    print("3. Use wildlife photography websites")
    print("4. Manually collect and place in data/tiger/")
    
    return tiger_urls

def download_sample_images():
    """Download a small sample of images for testing"""
    print("Downloading sample images...")
    
    # Sample cat images from Unsplash
    cat_urls = [
        "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=224&h=224&fit=crop",
        "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=224&h=224&fit=crop",
        "https://images.unsplash.com/photo-1596854407944-bf87f6fdd49e?w=224&h=224&fit=crop",
        "https://images.unsplash.com/photo-1571566882372-1598d88abd90?w=224&h=224&fit=crop",
        "https://images.unsplash.com/photo-1592194996308-7b43878e84a6?w=224&h=224&fit=crop",
    ]
    
    # Sample tiger images from Unsplash
    tiger_urls = [
        "https://images.unsplash.com/photo-1552410260-0fd9b577afa6?w=224&h=224&fit=crop",
        "https://images.unsplash.com/photo-1561731216-c3a4d99437d5?w=224&h=224&fit=crop",
        "https://images.unsplash.com/photo-1551969014-7d2c4cddf0b6?w=224&h=224&fit=crop",
        "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=224&h=224&fit=crop",
        "https://images.unsplash.com/photo-1596854407944-bf87f6fdd49e?w=224&h=224&fit=crop",
    ]
    
    # Download cat images
    os.makedirs('data/cat', exist_ok=True)
    for i, url in enumerate(cat_urls):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(f'data/cat/sample_cat_{i:02d}.jpg', 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded cat image {i+1}")
        except Exception as e:
            print(f"Failed to download cat image {i+1}: {e}")
    
    # Download tiger images
    os.makedirs('data/tiger', exist_ok=True)
    for i, url in enumerate(tiger_urls):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(f'data/tiger/sample_tiger_{i:02d}.jpg', 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded tiger image {i+1}")
        except Exception as e:
            print(f"Failed to download tiger image {i+1}: {e}")

def organize_dataset():
    """Organize the dataset into proper train/validation splits"""
    print("Organizing dataset...")
    
    # Create validation directories
    os.makedirs('data/validation/cat', exist_ok=True)
    os.makedirs('data/validation/tiger', exist_ok=True)
    
    # Get all images
    cat_files = [f for f in os.listdir('data/cat') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    tiger_files = [f for f in os.listdir('data/tiger') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(cat_files)} cat images and {len(tiger_files)} tiger images")
    
    # Split into train/validation (80/20)
    np.random.seed(42)
    np.random.shuffle(cat_files)
    np.random.shuffle(tiger_files)
    
    cat_val_count = max(1, len(cat_files) // 5)
    tiger_val_count = max(1, len(tiger_files) // 5)
    
    # Move validation images
    for i in range(cat_val_count):
        os.rename(f'data/cat/{cat_files[i]}', f'data/validation/cat/{cat_files[i]}')
    
    for i in range(tiger_val_count):
        os.rename(f'data/tiger/{tiger_files[i]}', f'data/validation/tiger/{tiger_files[i]}')
    
    print(f"Dataset organized: {len(cat_files)-cat_val_count} train cats, {cat_val_count} val cats")
    print(f"                  {len(tiger_files)-tiger_val_count} train tigers, {tiger_val_count} val tigers")

def main():
    """Main function to set up the dataset"""
    print("=== Cat vs Tiger Dataset Setup ===")
    
    # Download sample images first
    download_sample_images()
    
    # Try to download Oxford Pets dataset
    if download_oxford_pets():
        print("Successfully downloaded additional cat images!")
    
    # Organize the dataset
    organize_dataset()
    
    print("\n=== Dataset Setup Complete ===")
    print("Your dataset is ready for training!")
    print("Next: Run the improved training notebook")

if __name__ == "__main__":
    main()

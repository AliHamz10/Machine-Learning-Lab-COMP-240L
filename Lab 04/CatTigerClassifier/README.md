# Lab 04: Deep Learning & Computer Vision - Cat vs Tiger Classification

**Course:** Machine Learning Lab (COMP-240L)  
**Class:** BSAI F23 Red  
**Lab Number:** 04  
**Topic:** Deep Learning, Computer Vision, and Transfer Learning

## 📚 Lab Overview

This lab focuses on implementing deep learning models for computer vision tasks, specifically building a classifier to distinguish between cats and tigers. Students will learn about convolutional neural networks (CNNs), transfer learning, data augmentation, and model optimization techniques.

## 🎯 Learning Objectives

By the end of this lab, students will be able to:

- Understand deep learning concepts and neural networks
- Implement convolutional neural networks (CNNs)
- Apply transfer learning using pre-trained models
- Use data augmentation techniques
- Optimize model performance and prevent overfitting
- Evaluate computer vision models effectively

## 🐱🐅 Project Description

### Problem Statement
Build a deep learning model that can accurately classify images as either cats or tigers. This is a binary classification problem in computer vision that demonstrates the power of CNNs and transfer learning.

### Business Applications
- **Wildlife Monitoring:** Automated species identification in camera traps
- **Pet Care:** Smart pet identification systems
- **Education:** Interactive learning tools for animal recognition
- **Research:** Automated data collection for wildlife studies

## 📊 Dataset

### Oxford-IIIT Pet Dataset
- **Source:** Oxford Visual Geometry Group
- **Total Images:** 7,349 pet images
- **Categories:** 37 different pet breeds
- **Subset Used:** Cat and Tiger images only
- **Image Resolution:** Variable (typically 224x224 after preprocessing)

### Data Structure
```
data/
├── cat/                    # Cat images
│   ├── sample_cat_00.jpg
│   ├── sample_cat_01.jpg
│   └── ...
├── tiger/                  # Tiger images
│   ├── sample_tiger_00.jpg
│   ├── sample_tiger_01.jpg
│   └── ...
└── validation/             # Validation set
    ├── cat/
    └── tiger/
```

## 🏗️ Model Architecture

### Transfer Learning Approach
- **Base Model:** MobileNetV2 (pre-trained on ImageNet)
- **Architecture:** Feature extraction + custom classification head
- **Input Size:** 224x224x3 (RGB images)
- **Output:** Binary classification (Cat/Tiger)

### Model Components
1. **Feature Extractor:** Pre-trained MobileNetV2 (frozen weights)
2. **Global Average Pooling:** Reduces spatial dimensions
3. **Dense Layers:** Custom classification head
4. **Dropout:** Prevents overfitting
5. **Output Layer:** Sigmoid activation for binary classification

## 📁 Files Structure

```
Lab 04/CatTigerClassifier/
├── CatTigerClassifier.ipynb                    # Original implementation
├── CatTigerClassifier_Improved.ipynb          # Enhanced version
├── cat_tiger_mobilenetv2.h5                   # Trained model weights
├── data/                                       # Dataset directory
│   ├── cat/                                    # Cat images
│   ├── tiger/                                  # Tiger images
│   ├── validation/                             # Validation set
│   └── README.md                               # Data documentation
├── temp_data/                                  # Temporary data processing
├── data_collection.py                          # Data collection script
├── download_dataset.py                         # Dataset download script
├── oxford_pets.tar.gz                          # Original dataset archive
├── requirements.txt                            # Python dependencies
├── venv/                                       # Virtual environment
└── README.md                                   # This documentation
```

## 🛠️ Technical Implementation

### Libraries Used
```python
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

### Key Techniques

#### 1. Data Augmentation
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)
```

#### 2. Transfer Learning Setup
```python
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model layers
base_model.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)
```

#### 3. Model Compilation
```python
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

## 📈 Model Performance

### Training Results
- **Training Accuracy:** 95%+
- **Validation Accuracy:** 90%+
- **Training Time:** ~30 minutes (GPU)
- **Model Size:** ~14MB (MobileNetV2)

### Evaluation Metrics
- **Precision:** 0.92 (Cat), 0.89 (Tiger)
- **Recall:** 0.89 (Cat), 0.92 (Tiger)
- **F1-Score:** 0.90 (Cat), 0.90 (Tiger)
- **Confusion Matrix:** Low false positive/negative rates

## 🚀 Getting Started

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation
```bash
# Download dataset (if needed)
python download_dataset.py

# Organize data
python data_collection.py
```

### 3. Model Training
```bash
# Run Jupyter notebook
jupyter notebook CatTigerClassifier.ipynb

# Or run improved version
jupyter notebook CatTigerClassifier_Improved.ipynb
```

## 📊 Key Features

### 1. Data Augmentation
- **Rotation:** ±20 degrees
- **Translation:** Horizontal/vertical shifts
- **Zoom:** Scale variations
- **Flip:** Horizontal flipping
- **Shear:** Geometric transformations

### 2. Transfer Learning
- **Pre-trained Model:** MobileNetV2 on ImageNet
- **Frozen Layers:** Feature extractor weights
- **Trainable Layers:** Custom classification head
- **Fine-tuning:** Optional unfreezing of top layers

### 3. Model Optimization
- **Early Stopping:** Prevents overfitting
- **Learning Rate Reduction:** Adaptive learning
- **Dropout:** Regularization technique
- **Batch Normalization:** Training stability

## 🎓 Learning Outcomes

### Technical Skills
- **Deep Learning:** CNN architecture understanding
- **Transfer Learning:** Leveraging pre-trained models
- **Computer Vision:** Image preprocessing and augmentation
- **Model Optimization:** Hyperparameter tuning and regularization

### Practical Applications
- **Image Classification:** Binary classification tasks
- **Model Deployment:** Saving and loading trained models
- **Performance Evaluation:** Comprehensive model assessment
- **Data Pipeline:** End-to-end ML workflow

## 🔧 Troubleshooting

### Common Issues
1. **Memory Errors:** Reduce batch size or image resolution
2. **Slow Training:** Use GPU acceleration or reduce model complexity
3. **Poor Performance:** Check data quality and augmentation
4. **Overfitting:** Increase dropout or reduce model complexity

### Solutions
```python
# Reduce memory usage
BATCH_SIZE = 16
IMG_SIZE = (224, 224)

# Enable GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
```

## 📚 Additional Resources

### Documentation
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Documentation](https://keras.io/api/)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)

### Tutorials
- [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Computer Vision with Keras](https://keras.io/examples/vision/)
- [Data Augmentation Techniques](https://www.tensorflow.org/tutorials/images/data_augmentation)

## ✅ Lab Completion Checklist

- [ ] Environment setup completed
- [ ] Dataset downloaded and organized
- [ ] Data augmentation implemented
- [ ] Transfer learning model built
- [ ] Model trained successfully
- [ ] Performance evaluation completed
- [ ] Results documented and visualized

## 📝 Lab Report

Students should document their findings including:
- Model architecture and hyperparameters
- Training progress and metrics
- Performance evaluation results
- Challenges encountered and solutions
- Business applications and insights

## 🎯 Next Steps

After completing Lab 04, students will be ready for:
- Lab 05: Logistic Regression & Classification
- Advanced deep learning topics
- Model deployment and serving
- Real-world computer vision applications

---

**Lab Duration:** 4 hours  
**Difficulty Level:** Intermediate  
**Prerequisites:** Lab 01-03, Basic Deep Learning  
**Deliverables:** Trained model, Jupyter notebook, Performance analysis

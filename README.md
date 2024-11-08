
```markdown
# Skin Lesion Diagnosis and Classification System

## Overview
This project implements a deep learning-based system for skin lesion diagnosis and classification using the ISIC dataset. The system performs both detailed subtype classification of skin lesions (8 classes) and binary classification (cancerous vs. non-cancerous).

## Features
- Multi-class classification of skin lesions into 8 categories:
  - AKIEC (Actinic Keratosis & Intraepithelial Carcinoma)
  - BCC (Basal Cell Carcinoma)
  - BKL (Benign Keratosis)
  - DF (Dermatofibroma)
  - MEL (Melanoma)
  - NV (Melanocytic Nevus)
  - VASC (Vascular Lesion)
  - Normal Skin
- Binary classification of lesions (Cancerous vs. Non-Cancerous)
- Utilizes EfficientNetB1 architecture
- Implements data augmentation techniques
- Handles class imbalance

## Performance Metrics

### Detailed Classification (8 classes)
- Overall Accuracy: 72%
- Class-wise Performance:
  ```
              Precision    Recall  F1-Score
  AKIEC         0.76       0.63     0.69
  BCC           0.83       0.67     0.74
  BKL           0.64       0.70     0.67
  DF            0.47       0.64     0.54
  MEL           0.67       0.53     0.59
  NV            0.58       0.70     0.64
  Normal        1.00       1.00     1.00
  VASC          0.78       1.00     0.88
  ```

### Binary Classification
- Overall Accuracy: 98%
- Performance Metrics:
  ```
                 Precision    Recall  F1-Score
  Cancerous         1.00       0.98     0.99
  Non-Cancerous     0.92       1.00     0.96
  ```



## Dataset Structure
```
Dataset/
├── Skin_Lesion_Datasets/
│   ├── All/
│   │   └── our_normal_skin/
│   └── GroundTruth.csv
```

## Model Architecture
- Base Model: EfficientNetB1
- Additional layers:
  - Global Average Pooling
  - Dropout layers
  - Dense layers with regularization
- Optimization: Adam optimizer
- Loss: Categorical Crossentropy

## Data Preprocessing
- Image resizing
- Normalization
- Data augmentation:
  - Rotation
  - Horizontal/Vertical flip
  - Zoom
  - Brightness adjustment

## Model Training
- Learning Rate: Adaptive (with ReduceLROnPlateau)
- Early Stopping
- Model Checkpointing
- Class weight balancing

## Limitations
- Limited dataset size for some classes (e.g., DF with only 11 samples)
- Lower performance on certain subtypes (e.g., Dermatofibroma)
- Requires high-quality images for accurate prediction

## Acknowledgments
- ISIC Archive for the dataset



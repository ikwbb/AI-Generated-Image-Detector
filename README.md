
# AI-Generated-Image-Detector

**A detector ensembled with Swin-Transformer and CLIP**

This project implements an ensemble model for detecting AI-generated images, combining a fine-tuned Swin-Transformer and a CLIP-based feature classifier. The Swin-Transformer is fine-tuned for image classification, while CLIP extracts robust features that are classified using a custom neural network. The final prediction is an ensemble of both models' outputs.


## Table of Contents
1. [Requirements](#requirements)
2. [Setup](#setup)
3. [Reproduction](#reproduction)
   - [Fine-Tuning Swin-Transformer](#fine-tuning-swin-transformer)
   - [CLIP Feature Classification](#clip-feature-classification)
4. [Model Ensembling](#model-ensembling)
5. [Testing](#testing)
6. [License](#license)


## Requirements

To run this project, install the following Python packages:

```bash
pip install torch torchvision timm
pip install git+https://github.com/openai/CLIP.git
```

Additional dependencies (automatically installed with the above):
- `numpy`
- `scikit-learn`
- `pillow`
- `tqdm`

Ensure you have a CUDA-enabled GPU for optimal performance, though the code supports CPU execution as well.



## Setup

1. **Dataset Preparation**: Modify the `dataset_path` variable in the code to point to your dataset directory:
   ```python
   dataset_path = "./AIGC-Detection-Dataset"
   ```
   The dataset should have the following structure:
   ```
   AIGC-Detection-Dataset/
   ├── train/
   │   ├── 0_real/
   │   └── 1_fake/
   └── val/
       ├── 0_real/
       └── 1_fake/
   ```

2. **Pretrained Models**: The code uses pretrained weights for Swin-Transformer (`swinv2_small_window16_256`) and CLIP (`ViT-L/14@336px`), which are downloaded automatically via `timm` and `clip`.



## Reproduction

### Fine-Tuning Swin-Transformer

The Swin-Transformer is fine-tuned on the dataset with specific layers unfrozen for training. Key configurations include:
- **Model**: `swinv2_small_window16_256`
- **Batch Size**: 64
- **Epochs**: 100
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingWarmRestarts
- **Loss**: CrossEntropyLoss with label smoothing (0.05)

#### Steps
1. Run the fine-tuning script (provided in the code).
2. Models are saved per epoch in `Fine_Tuned_Swin_Models/`.
3. The final model is selected based on the **largest validation loss** among epochs 50-100 with validation accuracy > 0.99. This is saved as `swin_model.pth`.

#### Why Largest Validation Loss?
Validation accuracy reflects intra-domain performance, which may not generalize across domains. A model with slightly lower intra-domain accuracy (and higher loss) might generalize better in cross-domain scenarios.



### CLIP Feature Classification

CLIP (`ViT-L/14@336px`) extracts image features, which are then classified using a custom neural network.

#### Data Augmentation
- **Input**: Training images
- **Augmentations**:
  - Padding to 336x336
  - Center crop to 336x336
  - Horizontal flip
  - TenCrop (four corners, center, and their flipped versions)
- **Output**: Augmented images saved in `train_augmented/`

#### Feature Extraction
- Features are extracted using CLIP’s `encode_image` method.
- Features are scaled using `StandardScaler` (saved as `trained_scaler.pkl`).

#### Model Training
- **Architecture**: `ComplexClassifier` (input_dim=768, hidden_dim=512, output_dim=1)
- **Optimizer**: SGD (lr=0.9, momentum=0.99, weight_decay=1e-4)
- **Loss**: BCEWithLogitsLoss
- **Epochs**: 100
- **Selection**: Model with the highest validation accuracy is saved as `model.pth`.

#### Why Highest Accuracy?
Since CLIP is not fine-tuned for this task, validation accuracy is assumed to correlate strongly with test set performance.



## Model Ensembling

The final prediction combines outputs from both models:
- **CLIP Probability**: Weight = 0.489
- **Swin Probability**: Weight = 0.511
- **Threshold**: Combined score > 0.5 indicates an AI-generated image.

See the [testing section](#testing) for implementation details.



## Testing

### Test Code Template
The provided test function evaluates the ensemble model:
```python
def test(model, swin_model, test_dataset_path):
    # Load models, dataset, and compute predictions
    # Returns accuracy
```

### Demo Implementation
- **`data_loader`**: Custom dataset class to preprocess images and extract CLIP features.
- **`ComplexClassifier`**: Loads `model.pth`.
- **Swin-Transformer**: Loads `swin_model.pth`.
- **Usage**:
  ```python
  test_dataset_path = "./AIGC-Detection-Dataset/val"
  accuracy = test(model, swin_model, test_dataset_path)
  print(f"Test Accuracy: {accuracy:.4f}")
  ```

### Notes
- Customize `data_loader` and `model` based on your dataset structure.
- Ensure `trained_scaler.pkl`, `model.pth`, and `swin_model.pth` are in the working directory.



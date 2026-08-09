# CAM-Guided Multi-Task Learning for Industrial Surface Defect Detection

This project focuses on detecting defects in industrial surface images using deep learning.

The model uses DenseNet-121 as the backbone and performs both defect classification and segmentation. Class Activation Maps (CAM) are also used to visualize the regions related to the model's predictions.

## Dataset

The project uses the Kolektor Surface Defect Dataset 2 (KSDD2).

- Total images: 3,335
- Defective images: 356
- Non-defective images: 2,979
- Original image size: approximately 230 × 630 pixels
- Input size used for the model: 256 × 256 pixels

The dataset is not included in this repository.

## Model

- DenseNet-121 with ImageNet pretraining
- U-Net style decoder for segmentation
- Classification and segmentation heads
- Class Activation Maps (CAM)
- CrossEntropyLoss for classification
- BCEWithLogitsLoss + Dice Loss for segmentation
- Adam optimizer
- Learning rate: 0.0001
- Batch size: 8
- Epochs: 10

## Results

### Classification

| Metric | Score |
|---|---:|
| Accuracy | 95.72% |
| Precision | 100% |
| Recall | 60.91% |
| F1 Score | 75.71% |
| AUROC | 0.9698 |

### Segmentation

| Metric | Score |
|---|---:|
| Mean Dice Score | 0.9432 |
| Mean IoU | 0.9302 |

## Files

- `train_cam_multitask.ipynb` – Notebook used for training the model
- `evaluate_model.ipynb` – Notebook used for evaluating the model
- `requirements.txt` – Required Python packages
- `Documentation/` – Project report and presentation

## Technologies

- Python
- PyTorch
- Torchvision
- OpenCV
- Albumentations
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## How to Run

1. Download the KSDD2 dataset.

2. Download the trained model weights:

[Download trained model weights](https://drive.google.com/file/d/17xrwEPIo7foRC8fccb0OBXmYYn19XSHZ/view?usp=sharing)

3. Place `cam_multitask_densenet_ksdd.pth` in the project folder.

4. Install the required packages:

```bash
pip install -r requirements.txt
```

5. Open `train_cam_multitask.ipynb` to train the model.
6. Open `evaluate_model.ipynb` to evaluate the model.

## Documentation

The Documentation folder contains the project report and presentation.


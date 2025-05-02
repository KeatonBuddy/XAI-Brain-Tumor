# Explainable Tumor Detection with Deep Learning

## Overview

This project investigates state‑of‑the‑art convolutional and transformer architectures for **brain tumor classification** on MRI scans, with an emphasis on **explainability**.  
We fine‑tune three pretrained models—EfficientNet‑B0, ResNet‑50, and Vision Transformer (ViT‑B16)—to distinguish between:  
- Glioma  
- Meningioma  
- Pituitary tumor  
- Healthy brain  

To build trust in model predictions, we generate **Grad‑CAM** heatmaps for the CNNs, revealing which regions of the scan drive each decision.

## Key Features

- **Model Fine‑Tuning**  
  - Replace top layers of ImageNet‑pretrained backbones to output four classes  
  - Unfreeze entire network for end‑to‑end fine‑tuning  
  - Dynamic learning‑rate scheduling & checkpointing  

- **Performance Evaluation**  
  - Metrics: accuracy, precision, recall, F1‑score  
  - Training vs. held‑out testing analysis  
  - Comparative study: CNN vs. transformer  

- **Explainable AI**  
  - Grad‑CAM / SmoothGradCAM++ visualizations for ResNet & EfficientNet  
  - Overlay activation heatmaps on raw MRIs  
  - Side‑by‑side failure‑case comparisons  

- **Reproducible Environment**  
  - `environment.yml` for Conda setup (TensorFlow & PyTorch stacks)  
  - `requirements.txt` for pip users  

## Data

We use the publicly available [Brain Tumor MRI Dataset](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection), organized into:
```
data/
├── Training/ # 4 subfolders: glioma, meningioma, pituitary, healthy
└── Testing/ # Same 4 subfolders for held‑out evaluation
```

> **Note:** Before running the notebooks, download the dataset into `data/` following the above structure.

## Repository Structure
```
`├── data/
│ ├── Training/
│ └── Testing/
├── efficientNet_v7.ipynb # Keras / TensorFlow workflow
├── resnet-torch.ipynb # PyTorch / torchvision workflow
├── vit.ipynb # Vision Transformer (timm or tf.keras) workflow
├── gradcam_comparison.ipynb # Grad‑CAM generation & failure‑case analysis
├── environment.yml # Conda dependencies
├── requirements.txt # pip dependencies
└── README.md # This file
```
## Quickstart

1. **Clone the repo**  
   ```
   git clone https://github.com/your-org/explainable-tumor-detection.git
   cd explainable-tumor-detection
   ```
2. **Set up environment**
   - Conda:
   ```
   conda env create -f environment.yml
   conda activate tumor-xai
   ```
   - pip:
    ```
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3. **Place Data**
  Download and unzip the Brain Tumor MRI dataset into ./data/.
4. *Launch notebooks*
  ```
  jupyter lab
  ```
  - Run efficientNet_v7.ipynb first to train EfficientNet‑B0.

  - Next, run resnet-torch.ipynb for ResNet‑50.

  - Then vit.ipynb to fine‑tune the Vision Transformer.

  - Finally, gradcam_comparison.ipynb to generate and visualize Grad‑CAM maps.
## Results Summary
| Model           | Test Accuracy | Precision | Recall   | F1‑Score |
| --------------- | ------------- | --------- | -------- | -------- |
| EfficientNet‑B0 | 91.4 %        | 0.91      | 0.91     | 0.91     |
| ResNet‑50       | 87.6 %        | 0.88      | 0.88     | 0.88     |
| ViT‑B16         | **93.4 %**    | **0.93**  | **0.93** | **0.93** |

*Transformer‑based ViT achieves the highest accuracy, while all models deliver clinically relevant performance.*


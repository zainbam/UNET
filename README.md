# Medical Image Segmentation: U-Net and Hybrid LSTM-U-Net

This repository contains two advanced models for medical image segmentation:  
1. **U-Net**: A widely used convolutional neural network architecture for image segmentation.  
2. **Hybrid LSTM-U-Net**: A novel architecture combining Convolutional LSTM layers with U-Net for iterative refinement, enhancing segmentation performance through feedback mechanisms.

Both architectures are tailored for medical image segmentation, particularly focusing on brain tumor segmentation using the BRATS dataset.

## Project Overview

This project was developed for **Aga Khan Hospital** as part of a collaboration to improve brain tumor detection. The models implemented here focus on semantic segmentation, with special attention to improving accuracy in medical image analysis.

### 1. U-Net Model
The U-Net architecture is a standard for image segmentation tasks and is well-suited for medical applications like detecting brain tumors, retinal vessels, and skin lesions. This version has been optimized for the BRATS dataset and other medical datasets.

### 2. Hybrid LSTM-U-Net Model
Our novel approach extends the U-Net architecture by incorporating **Convolutional LSTM layers**. This hybrid model leverages feedback loops to iteratively refine segmentation outputs, mimicking feedback mechanisms in the human brain. This method improves segmentation results for complex medical images, including brain tumor segmentation, where subtle features are critical for accurate analysis.

## Key Features
- **Semantic Segmentation**: Focused on medical imaging, particularly brain tumor segmentation.
- **Two Models**: U-Net and Hybrid LSTM-U-Net.
- **Datasets**: Trained on the BRATS dataset.
- **Evaluation Metrics**: Dice coefficient, Jaccard index, recall, and precision (available in code).
- **Feedback Mechanism**: In the LSTM-U-Net model, results are iteratively refined for improved accuracy.

## Architecture Breakdown
### U-Net
The U-Net architecture comprises:
- **Encoder**: Extracts features from the input image using convolutional layers.
- **Decoder**: Reconstructs segmented outputs using upsampling and concatenation with corresponding encoder layers.

### Hybrid LSTM-U-Net
The Hybrid LSTM-U-Net modifies the U-Net with:
- **Convolutional LSTM Blocks**: Used in both the encoder and decoder stages to capture temporal dependencies in medical images.
- **Feedback Loop**: Outputs from earlier iterations are fed back as inputs for subsequent refinements.

## Results
Both models have been tested on medical image datasets, achieving promising results. The **Hybrid LSTM-U-Net** model significantly improved segmentation accuracy due to the feedback mechanism, especially in complex cases like brain tumors.

## Usage
Clone the repository and follow the instructions in the provided Jupyter notebooks to run the models on your data. You can use Google Colab for training with GPU support.

```bash
git clone https://github.com/zainbam/UNET.git

# TinyVGG: FashionMNIST Image Classification with PyTorch

## Overview
TinyVGG is a PyTorch-based project that implements a simple yet effective convolutional neural network (CNN) inspired by the VGG architecture. The model is trained to classify images from the FashionMNIST dataset, which consists of 28x28 grayscale images of clothing items across 10 categories.

## Features
- Custom TinyVGG CNN architecture implemented from scratch
- Training and evaluation loops with accuracy and loss tracking
- Model saving and loading
- Simple, modular code structure for easy experimentation

## Project Structure
```
Tiny_VGG/
├── accuracy_fn.py   # Accuracy calculation function
├── eval.py          # Model evaluation logic
├── main.py          # Main training and evaluation script
├── test.py          # Testing loop
├── tiny_VGG.py      # TinyVGG model definition
├── train.py         # Training loop
```

## Model Architecture
The TinyVGG model consists of:
- Two convolutional blocks (Conv2d + ReLU + Conv2d + ReLU + MaxPool2d)
- A fully connected classifier layer

## Setup
1. **Clone the repository** and navigate to the project directory.
2. **Install dependencies** (see `requirements.txt`). For example:
   ```bash
   pip install torch torchvision pandas matplotlib torchmetrics
   ```
3. **Run the main script**:
   ```bash
   cd Tiny_VGG
   python main.py
   ```

## Usage
- The script will automatically download the FashionMNIST dataset.
- The model will be trained for 3 epochs and saved as `model.pth`.
- After training, evaluation metrics (loss and accuracy) will be printed.

## Requirements
- Python 3.7+
- PyTorch
- torchvision
- pandas
- matplotlib
- torchmetrics

## Acknowledgements
- [PyTorch](https://pytorch.org/)
- [FashionMNIST Dataset](https://github.com/zalandoresearch/fashion-mnist) 

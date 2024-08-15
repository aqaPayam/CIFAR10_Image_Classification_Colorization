# CIFAR10 Image Classification and Colorization

This repository contains the implementation of two tasks using PyTorch: image classification and image colorization, both applied to the CIFAR-10 dataset. This is part of HW2 for the Deep Learning course instructed by Dr. Soleymani.

## Project Overview

The goal of this project is to apply deep learning techniques to solve two key tasks using the CIFAR-10 dataset:
1. **Image Classification:** Classifying images into one of the 10 classes in the CIFAR-10 dataset.
2. **Image Colorization:** Colorizing grayscale images from the CIFAR-10 dataset using deep learning techniques.

## Task 1: CIFAR10 Image Classification

The image classification task uses a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset into one of 10 categories (airplanes, cars, birds, cats, etc.). The project explores how CNNs can effectively learn image features and perform high-accuracy classification.

## Task 2: Image Colorization

The colorization task focuses on taking grayscale images and converting them back into color using deep learning techniques. This task aims to predict the missing color channels of grayscale images by learning from the color patterns present in the CIFAR-10 dataset.

## Dataset

The dataset used is the CIFAR-10 dataset, which consists of 60,000 32x32 color images divided into 10 classes. The dataset is automatically downloaded via the `torchvision` library.

- **CIFAR-10 Classes:** Airplanes, Cars, Birds, Cats, Deer, Dogs, Frogs, Horses, Ships, Trucks.
- The dataset is split into training and validation sets for both tasks.

## Getting Started

1. Clone the repository:
    ```bash
    git clone https://github.com/AqaPayam/CIFAR10_Image_Classification_Colorization.git
    ```

2. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Jupyter notebook:
    ```bash
    jupyter notebook CIFAR10_Image_Classification_And_Colorization.ipynb
    ```

## Requirements

- Python 3.x
- PyTorch
- Torchvision
- Numpy
- Pandas
- Matplotlib
- Seaborn
- tqdm
- scikit-learn
- torchsummary

## Running the Model

1. **Classification Task:** Execute the cells related to image classification in the Jupyter notebook to train the CNN on the CIFAR-10 dataset and evaluate its performance on the test set.
2. **Colorization Task:** Execute the cells related to image colorization to train a model that converts grayscale images back to color. The notebook contains the necessary steps for data preprocessing, model training, and evaluation.

## Results

- **Classification:** The model aims to achieve high accuracy in identifying the correct class for each image.
- **Colorization:** The model attempts to restore the original colors of grayscale images as closely as possible by learning the distribution of colors in the CIFAR-10 dataset.

## Acknowledgments

This project is part of the Deep Learning course at [Institution Name] by Dr. Soleymani.

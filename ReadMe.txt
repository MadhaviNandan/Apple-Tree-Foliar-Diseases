# Apple Tree Foliar Disease Recognition and Classification

## Overview

This project focuses on the recognition and classification of apple tree foliar diseases using image processing and machine learning techniques. The goal is to assist farmers and agricultural experts in identifying different types of leaf diseases early, allowing for timely intervention and better crop management. The system leverages a deep learning model to analyze leaf images, detect diseases, and classify them into predefined categories.

## Features

- **Image Preprocessing**: Uses techniques like resizing, normalization, and data augmentation to prepare leaf images for training.
- **Disease Detection**: A convolutional neural network (CNN) trained to identify and classify diseases from images of apple tree leaves.
- **Multi-class Classification**: Classifies the leaf into categories such as healthy or one of several common foliar diseases.
- **User Interface**: A simple interface where users can upload leaf images and receive classification results.
- **Performance Metrics**: Evaluates the model's accuracy, precision, recall, and F1-score on test data.

## Dataset

The dataset used for this project consists of labeled images of apple tree leaves, including both healthy leaves and leaves affected by diseases such as apple scab, cedar apple rust, and more. The dataset is pre-split into training, validation, and test sets.

- **Training Data**: Images used to train the model.
- **Validation Data**: Used to fine-tune hyperparameters.
- **Test Data**: Used to evaluate the final model.

## Technologies Used

- **Python**
- **TensorFlow/Keras** for building and training the neural network.
- **OpenCV** for image preprocessing and augmentation.
- **Flask** (optional) for building the web-based user interface.
- **Matplotlib** and **Seaborn** for visualizing data and performance metrics.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/apple-tree-disease-classification.git
   cd apple-tree-disease-classification
   ```

2. Create a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate   # On Windows: env\Scripts\activate
   ```


4. (Optional) Download or load your dataset in the appropriate directory.

## Usage

1. **Training the Model**:
   - To train the model, run the following command:
     ```bash
     python train_model.py
     ```

2. **Testing the Model**:
   - To evaluate the trained model on test data, use:
     ```bash
     python test_model.py
     ```

3. **Running the Application**:
   - To launch the web interface for disease recognition, run:
     ```bash
     python app.py
     ```

4. **Image Upload**:
   - Once the application is running, you can upload a leaf image to get the disease classification results.

## Model Architecture

The neural network used is a Convolutional Neural Network (CNN) with the following architecture:
- **Input Layer**: Accepts 3-channel (RGB) images of size 256x256.
- **Convolutional Layers**: Several layers to extract features from the images.
- **Pooling Layers**: Reduces dimensionality and focuses on important features.
- **Fully Connected Layer**: For classification into disease categories.
- **Output Layer**: Uses softmax activation for multi-class classification.


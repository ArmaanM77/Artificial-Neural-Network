# Neural Network from Scratch

## Overview

This project implements an artificial neural network (ANN) from scratch using only [NumPy](https://numpy.org/) and [Pandas](https://pandas.pydata.org/). The purpose of this project is to demonstrate a deep understanding of the fundamental principles behind neural networks, including the forward and backward propagation algorithms, weight updates, and activation functions.

## Project Description

In this project, I built a fully functional neural network without using any high-level deep learning libraries. The network includes:

- **Layer Abstraction:** Custom classes to represent layers with weight and bias initialization, feedforward, and backpropagation methods.
- **Activation Functions:** Implementation of several activation functions, including:
  - **Standard Functions:** Sigmoid, Tanh, ReLU, Leaky ReLU, and Softmax.
  - **Custom Activation Function:** *SinLog* — a hybrid function that applies `sin(x)` if \( x < 0 \) and `log(x+1)` otherwise.
- **Loss Functions:** Both mean squared error and cross-entropy variants with their respective derivatives.
- **Data Processing:** Reading and pre-processing of the MNIST dataset using IDX file formats.
- **Accuracy Evaluation:** Testing the neural network predictions on the MNIST dataset and comparing performance across different activation functions.

This project also includes a performance comparison of the custom SinLog activation function against widely used activations, such as Sigmoid, Leaky ReLU, and Softmax.

## Installation

Make sure you have Python 3 installed along with the required libraries. You can install the necessary dependencies using pip:

```bash
pip install numpy pandas matplotlib

Usage
Data Loading:
The MNIST dataset is loaded directly from IDX files into a Pandas DataFrame. The images are normalized to ensure numerical stability during training.

Training:
Instantiate the neural network (e.g., using the custom SinLog activation) and start training:

python
Copy
# Example for training the SinLog Neural Network
sinlognn = SinLog_NeuralNetwork(epochs=100, batch_size=64, learning_rate=0.1)
sinlognn.train(df)
print("SinLog Neural Network Training Completed")
Prediction and Evaluation:
After training, predictions are made on test data, and accuracy is computed by comparing the network's predictions to the true labels:

python
Copy
sinlog_prediction = sinlognn.predict(df)
# Accuracy calculation loop is provided in the code.
Comparison:
The network's performance with the custom SinLog activation function is compared with that using Sigmoid, Leaky ReLU, and Softmax.

Project Structure
Neural Network Implementation:
The core neural network code is implemented with classes for layers and the network as a whole. It includes the complete forward and backward propagation logic.

Custom Activation (SinLog):
The SinLog activation function applies a sine operation for negative inputs and a logarithmic operation for non-negative inputs, along with an appropriately defined derivative for backpropagation.

Data Handling:
Functions to load and process the MNIST dataset from IDX files are provided. The data is normalized and then fed into the network.

Evaluation Script:
There is a section in the code that evaluates the network's predictions and compares the accuracies of the different activation functions.

Acknowledgements
This project is inspired by fundamental deep learning techniques as discussed in textbooks like Deep Learning by Goodfellow, Bengio, and Courville. It represents one of my first projects in machine learning and neural network implementation.

Contributing
Since this is my first submission on GitHub, I am open to feedback and contributions to improve the quality and performance of the code. Please feel free to fork the repository and submit pull requests.

Contact
Armaan Mahajan
Email: armaanmahajanbg@gmail.com

Thank you for checking out my project! 
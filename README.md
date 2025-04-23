🦁 Animal Detection using Convolutional Neural Networks (CNN)
📌 Overview
This project focuses on detecting and classifying animals from images using a Convolutional Neural Network (CNN). It aims to help learners understand the fundamentals of image classification through deep learning, using a small curated dataset of animal images.

📂 Dataset from Kaggle: https://www.kaggle.com/datasets/ikjotsingh221/animal-dataset
The dataset is a custom collection of animal images, featuring four classes:
The entire dataset contains 2 folders:

Testing
-------
Bears : 95 images
Crows : 72 images
Elephants : 85 images
Rats : 74 images

Training
--------
Bears : 395 images
Crows : 231 images
Elephants : 367 images
Rats : 245 images

Each image is labeled according to the animal it represents.

🧠 Model Architecture
The CNN model is built from scratch using TensorFlow / Keras. The architecture includes:

Convolutional Layers (with ReLU activation)

Max Pooling Layers (to reduce spatial dimensions)

Dropout Layers (for preventing overfitting)

Flatten Layer (to convert feature maps to a 1D vector)

Fully Connected (Dense) Layers

Softmax Activation in Output Layer (for multi-class classification)

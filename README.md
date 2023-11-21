
# Emotional Recognition Model with Convolutional Neural Network (CNN)
A Convolutional Neural Network (CNN) is a powerful type of artificial neural network designed for processing and identifying patterns in two-dimensional data, particularly images. CNNs have demonstrated exceptional effectiveness in various computer vision tasks, with one notable application being facial emotion recognition.
![image](https://github.com/Gorchon/FacialEmotionRecognitionCVN/assets/116988804/0c606c52-f203-47f7-9389-20a580eedcc4)

## Overview
In our project, we utilized a CNN to detect and recognize emotions in individuals using approximately 30,000 labeled images. This model serves as a crucial component in understanding and responding to the emotional state of users interacting with our application.

## Training the Model
Requirements
To replicate our work, make sure you have the following dependencies installed:
```python
pip install -r requirements.txt
```
## Training Process
We trained the emotion recognition model using the following key steps:

Data Collection: Acquired a diverse dataset of approximately 30,000 labeled images, encompassing various facial expressions representing different emotions.

Data Preprocessing: Prepared the dataset by resizing images to a consistent format, normalizing pixel values, and augmenting the data to enhance model generalization.

Model Architecture: Designed a CNN architecture using Keras, a high-level neural networks API, built on top of TensorFlow. The model consisted of convolutional layers for feature extraction, pooling layers for spatial reduction, and dense layers for classification.

Training the Model: Fed the preprocessed data into the CNN, optimizing the model's parameters to minimize the classification error. This process involved backpropagation and updating the weights of the network.

Model Evaluation: Assessed the model's performance on a separate test dataset to ensure generalization and identify areas for potential improvement.

## Integration with OpenAI
With the trained emotion recognition model in place, we seamlessly integrated it into our application's backend. Upon detecting the user's emotional state, we leverage OpenAI to provide supportive and context-aware responses, aimed at enhancing the user's emotional well-being.

This combination of a powerful emotion recognition model and the natural language processing capabilities of OpenAI creates a holistic and responsive system that can significantly contribute to user satisfaction and emotional support.

By understanding and addressing user emotions, our application goes beyond conventional interactions, providing a personalized and empathetic experience for each individual.



# Fire Detection Model using Deep Learning

This project implements a deep learning model for detecting fire and smoke in images. The model is built using TensorFlow/Keras and deployed using Streamlit for creating an interactive web application.

## Overview

The fire detection model is trained on a dataset containing images of fire, smoke, and non-fire scenes. The model architecture consists of a convolutional neural network (CNN) designed to classify images into three categories: fire, smoke, and non-fire.

The trained model achieves high accuracy in classifying images, making it suitable for real-world applications such as fire detection in surveillance systems, firefighting drones, and environmental monitoring.

## Features

- Image classification: The model can classify images into three categories: fire, smoke, and non-fire.
- Streamlit app: The model is deployed as an interactive web application using Streamlit, allowing users to upload images and view the classification results in real-time.

## Installation

To run the fire detection model locally, follow these steps:

1. Clone this repository: 20Aarkay/fire-detection-model-using-deep-learning
2. Navigate to the project directory:

cd fire-detection-model

markdown
Copy code

3. Install the required dependencies:

pip install -r requirements.txt

yaml
Copy code

## Usage

To use the fire detection model, follow these steps:

1. Run the Streamlit app:

fire_detection_app.py

markdown
Copy code

2. Open your web browser and navigate to the URL provided by Streamlit.

3. Upload an image containing fire, smoke, or non-fire scenes.

4. View the classification results generated by the model.

## Contributing

Contributions to the fire detection model project are welcome! Here's how you can contribute:

- Fork the repository.
- Create a new branch (`git checkout -b feature/improvement`).
- Make your changes.
- Commit your changes (`git commit -am 'Add new feature'`).
- Push to the branch (`git push origin feature/improvement`).
- Create a new Pull Request.

## Dataset

The fire detection model is trained on a curated dataset containing thousands of images sourced from various sources, including public datasets, surveillance footage, and online resources. The dataset is annotated with labels indicating the presence of fire, smoke, or non-fire scenes, allowing the model to learn from diverse examples and generalize well to unseen data.

## Model Architecture

The fire detection model architecture consists of a convolutional neural network (CNN) with multiple layers of convolution, pooling, and fully connected layers. The model utilizes transfer learning by initializing with pre-trained weights from popular CNN architectures (e.g., ResNet, MobileNet) and fine-tuning the model on the fire detection dataset to adapt to specific features and patterns related to fire and smoke.

## Evaluation Metrics

The performance of the fire detection model is evaluated using standard metrics such as accuracy, precision, recall, and F1-score. The model is validated on a separate test dataset to assess its generalization ability and robustness to unseen data. The evaluation results demonstrate the effectiveness of the model in accurately detecting fire and smoke incidents while minimizing false alarms.

## Contributing

Contributions to the fire detection model project are welcome! Here's how you can contribute:

- Fork the repository.
- Create a new branch (`git checkout -b feature/improvement`).
- Make your changes.
- Commit your changes (`git commit -am 'Add new feature'`).
- Push to the branch (`git push origin feature/improvement`).
- Create a new Pull Request.

## Acknowledgments

- This project was inspired by the need for effective fire detection solutions in various industries.
- Special thanks to the contributors and developers of TensorFlow, Keras, and Streamlit for their amazing tools and libraries.

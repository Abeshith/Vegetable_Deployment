# Vegetable Image Classification App

**Click here for the app:** [Vegetable Image Classification App](https://vegetabledeployment-abe.streamlit.app/)

This Streamlit application classifies vegetable images using a pre-trained model. Users can upload an image of a vegetable, and the model will predict its class and provide detailed features of the vegetable.

## Features

- **Home**: Displays a welcome message and provides brief instructions on how to use the app.
- **Upload Image**: Allows users to upload a vegetable image. The app processes the image, predicts its class using the model, and displays the predicted class along with detailed features of the vegetable.
- **About**: Offers information about the app and its functionality.

## Code Explanation

### Page Configuration
Sets the title of the web page to "Vegetable Image Classification."

### Model Loading
Loads the pre-trained model from a saved file, which is used for predicting the class of the uploaded vegetable images.

### Class Labels and Features
Defines a mapping of numeric labels to vegetable names and provides detailed descriptions of each vegetable. These descriptions include nutritional benefits and common uses of the vegetables.

### Navigation Menu
Creates a sidebar with navigation options for "Home," "Upload Image," and "About."

### Home Section
Displays a welcome message and instructions on how to use the app.

### Upload Image Section
Enables users to upload a JPG image of a vegetable. Upon uploading:
- The image is processed and resized to the required dimensions.
- The model predicts the class of the vegetable.
- The predicted class is displayed along with a detailed description of the vegetable.

### About Section
Provides information about the app's purpose and how it works.

## Trained Model Link

The trained model used in this application is available for download at the following link:

[Trained Model - VGG19](https://github.com/Abeshith/ImageClassification-VGG19)

## Usage

1. **Home**: Visit the home page for an introduction and usage instructions.
2. **Upload Image**: Upload a vegetable image to classify it. The app will display the predicted class and detailed features of the vegetable.
3. **About**: Read more about the app and its functionality.

## Requirements

- Streamlit
- TensorFlow
- Pillow
- NumPy
- streamlit_option_menu

Ensure all required packages are installed before running the app.

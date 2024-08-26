import streamlit as st
from streamlit_option_menu import option_menu
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import requests

# Set page configuration
st.set_page_config(page_title="Vegetable Image Classification")

# Load the model
# Function to download the model from Google Drive
def download_model_from_drive(drive_url, model_path):
    response = requests.get(drive_url)
    with open(model_path, 'wb') as f:
        f.write(response.content)

# Google Drive direct download URL for the model
drive_url = "https://drive.google.com/uc?export=download&id=1V5ynGZ9ZdwLn92bt-mWpc0F1UtlG2c8R"

# Local path where the model will be saved
model_path = 'model.h5'

# Download and save the model locally if it doesn't exist
if not os.path.exists(model_path):
    st.write("Downloading the model...")
    download_model_from_drive(drive_url, model_path)
    st.write("Model downloaded successfully!")

# Load the model
try:
    model = load_model(model_path)
    st.write("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Define class labels
class_labels = {
    0: 'Bean',
    1: 'Bitter gourd',
    2: 'Bottle gourd',
    3: 'Brinjal',
    4: 'Broccoli',
    5: 'Cabbage',
    6: 'Capsicum',
    7: 'Carrot',
    8: 'Cauliflower',
    9: 'Cucumber',
    10: 'Papaya',
    11: 'Potato',
    12: 'Pumpkin',
    13: 'Radish',
    14: 'Tomato'
}

# Define vegetable features
vegetable_features = {
    # Same vegetable features dictionary
}

# Create a navigation menu
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "Upload Image", "About"],
        icons=["house", "cloud-upload", "info-circle"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
    )

# Display content based on selected menu option
if selected == "Home":
    st.title("Welcome to Vegetable Classification")
    st.write("Upload an image of a vegetable and the model will predict its class.")

elif selected == "Upload Image":
    st.title("Upload an Image")
    st.write("Upload your vegetable image here.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert the file to an image
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image.', use_column_width=False, width=200)  # Fixed size
        st.write("")
        st.write("Classifying...")

        # Preprocess the image
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Predict the class
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)
        class_name = class_labels[predicted_class[0]]

        st.write(f"**Prediction:** {class_name}")

        # Display features of the predicted vegetable
        st.write(f"**Features of {class_name}:** {vegetable_features[class_name]}")

elif selected == "About":
    st.title("About This App")
    st.write("This app classifies vegetables using a pre-trained model.")
    st.write("Upload an image of a vegetable, and the model will predict its class along with detailed features of the vegetable.")

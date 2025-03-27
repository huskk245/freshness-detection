import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Set page configuration
st.set_page_config(page_title="Freshness Detection System", page_icon="🍎", layout="centered")

# Title and description
st.title("🍎 Freshness Detection System")
st.write("Upload an image to check its freshness.")

# Cache the model loading to improve performance
@st.cache_resource
def load_model():
    model_path = "final_freshness_resnet_model.keras"
    
    # Debugging: Print current working directory and list files
    st.write(f"Current working directory: {os.getcwd()}")
    st.write(f"Files in directory: {os.listdir('.')}")
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Load the model
model = load_model()

# Class names for prediction
CLASS_NAMES = ["Fresh Apple", "Fresh Banana", "Fresh Orange", "Fresh Tomato", 
               "Rotten Apple", "Rotten Banana", "Rotten Orange", "Rotten Tomato"]

# Image preprocessing function
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match model input
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], accept_multiple_files=False)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make prediction
    if model is not None:
        try:
            prediction = model.predict(processed_image)
            predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
            confidence = np.max(prediction[0]) * 100

            # Display prediction result
            st.subheader("Prediction Result:")
            st.write(f"**Category:** {predicted_class}")
            st.write(f"**Confidence Score:** {confidence:.2f}%")

            # Warning if the item is rotten
            if "Rotten" in predicted_class:
                st.error(f"⚠️ The food item is {predicted_class}!")
            else:
                st.success(f"✅ The food item is {predicted_class}!")
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

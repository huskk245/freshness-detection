import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Set environment variable for protobuf compatibility
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Function to load the model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('final_freshness_resnet_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Prediction function
def predict_freshness(img):
    # Load the model
    model = load_model()
    if model is None:
        return None, None

    # Preprocess the image (exactly like in your training script)
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence_score = predictions[0][predicted_class] * 100

    # Get class names from the training generator
    class_names = list(train_generator.class_indices.keys())

    return class_names[predicted_class], round(confidence_score, 2)

# Recreate the data generator to get class indices
train_dir = r"C:\Users\Welcome\ai\archive (1)\dataset\Train"
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Streamlit app main function
def main():
    st.set_page_config(page_title="Freshness Detection System", page_icon=":apple:")
    
    st.title("🍎 Freshness Detection System")
    st.write("Upload an image to check its freshness.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Predict freshness
        predicted_label, confidence = predict_freshness(image)

        # Display results
        if predicted_label is not None:
            st.subheader("Prediction Result:")
            st.write(f"**Category:** {predicted_label}")
            st.write(f"**Confidence Score:** {confidence}%")

            # Highlight based on prediction
            if "fresh" in predicted_label.lower():
                st.success(f"✅ The food item is {predicted_label}!")
            else:
                st.warning(f"⚠️ The food item is {predicted_label}!")

if __name__ == '__main__':
    main()
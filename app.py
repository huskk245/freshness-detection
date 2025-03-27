import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ✅ Set page config (MUST be the first Streamlit command)
st.set_page_config(page_title="Freshness Detection System", page_icon="🍏")

# ✅ Environment variable fix for protobuf compatibility
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# ✅ Define correct class names
CLASS_NAMES = [
    "Fresh Apples", "Fresh Banana", "Fresh Cucumber", "Fresh Okra", "Fresh Oranges",
    "Fresh Potato", "Fresh Tomato", "Rotten Apples", "Rotten Banana", "Rotten Cucumber",
    "Rotten Okra", "Rotten Oranges", "Rotten Potato", "Rotten Tomato"
]

# ✅ Load the model only once (Optimized)
@st.cache_resource
def load_model():
    model_path = "final_freshness_resnet_model.keras"  # Make sure this file is uploaded
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# ✅ Prediction function
def predict_freshness(img):
    model = load_model()
    if model is None:
        return None, None

    # Preprocess image
    img_resized = img.resize((224, 224))
    img_array = np.expand_dims(np.array(img_resized), axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

    # Make prediction
    predictions = model.predict(img_array)

    # Debugging output
    print("Model Output Shape:", predictions.shape)
    print("Predictions:", predictions)

    # Validate class count
    if predictions.shape[1] != len(CLASS_NAMES):
        st.error(f"❌ Model predicts {predictions.shape[1]} classes, but {len(CLASS_NAMES)} class labels are defined.")
        return None, None

    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence_score = predictions[0][predicted_class] * 100

    return CLASS_NAMES[predicted_class], round(confidence_score, 2)

# ✅ Streamlit app main function
def main():
    st.title("🍎 Freshness Detection System")
    st.write("Upload an image to check its freshness.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Predict freshness
        predicted_label, confidence = predict_freshness(img)

        # Display results
        if predicted_label is not None:
            st.subheader("Prediction Result:")
            st.write(f"**Category:** {predicted_label}")
            st.write(f"**Confidence Score:** {confidence}%")

            # Highlight result
            if "Fresh" in predicted_label:
                st.success(f"✅ The food item is {predicted_label}!")
            else:
                st.warning(f"⚠️ The food item is {predicted_label}!")

if __name__ == "__main__":
    main()

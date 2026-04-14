import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
import os

# -------------------------------
# Load Model (SAFE)
# -------------------------------
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(
            "cat_dog_model.keras",
            compile=False   # 🔥 Fix for GetItem error
        )
        return model
    except Exception as e:
        st.error("❌ Model loading failed")
        st.text(str(e))
        return None

model = load_model()

# -------------------------------
# App Config
# -------------------------------
st.set_page_config(page_title="Cat vs Dog AI", layout="centered")

st.title("🐱🐶 Cat vs Dog Classifier")
st.write("Upload an image and let AI predict!")

# Debug (optional)
# st.write("Current directory:", os.getcwd())

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("Settings")
show_confidence = st.sidebar.checkbox("Show Confidence", True)

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "png", "jpeg"]
)

# -------------------------------
# Prediction Logic
# -------------------------------
if uploaded_file is not None and model is not None:
    try:
        # Load image
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Preprocess
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)  # 🔥 IMPORTANT

        # Predict
        prediction = model.predict(img_array)[0][0]

        # Classification
        if prediction > 0.5:
            label = "🐶 Dog"
            confidence = float(prediction)
        else:
            label = "🐱 Cat"
            confidence = 1 - float(prediction)

        # -------------------------------
        # Output
        # -------------------------------
        st.markdown("## Prediction Result")
        st.success(f"Result: {label}")

        if show_confidence:
            st.info(f"Confidence: {confidence * 100:.2f}%")

        st.progress(confidence)

    except Exception as e:
        st.error("❌ Error processing image")
        st.text(str(e))

elif model is None:
    st.warning("⚠️ Model not loaded. Check your model file path.")

else:
    st.warning("📤 Please upload an image to start.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Built using TensorFlow + Streamlit 🚀")
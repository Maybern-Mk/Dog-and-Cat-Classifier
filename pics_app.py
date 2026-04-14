import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# PAGE CONFIG
st.set_page_config(
    page_title="AI Vision App",
    page_icon="🤖",
    layout="centered"
)

# CUSTOM CSS (STARTUP STYLE)
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
h1 {
    text-align: center;
}
.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# LOAD MODEL (FAST)
@st.cache_resource
def load_my_model():
    return load_model("cnn_cifar10_best_model.h5")

model = load_my_model()

# CLASS LABELS
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

# HEADER
st.title("🚀 AI Image Classifier")
st.markdown("### Classify images using Deep Learning (CNN)")

# UPLOAD SECTION
uploaded_file = st.file_uploader(
    "📤 Upload your image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    # Show Image
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Uploaded Image", width="stretch")

    # Preprocessing
    image = image.convert("RGB")
    img = image.resize((32, 32))

    img_array = np.array(img) / 255.0

    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = np.expand_dims(img_array, axis=0)

    # Prediction Button
    if st.button("🔍 Predict"):

        prediction = model.predict(img_array)[0]

        # Top Prediction
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

        # RESULT CARD
        st.markdown("## 🎯 Prediction Result")
        st.success(f"**{predicted_class}** ({confidence:.2f})")

        # 📊 TOP 3 PREDICTIONS
        st.markdown("### 📊 Top Predictions")

        top_indices = prediction.argsort()[-3:][::-1]

        for i in top_indices:
            st.write(f"{class_names[i]} : {prediction[i]:.2f}")
            st.progress(float(prediction[i]))

# 📌 SIDEBAR
st.sidebar.title("ℹ️ About App")
st.sidebar.write("""
This AI model is trained on CIFAR-10 dataset.

It can classify images into:
- ✈️ Airplane
- 🚗 Automobile
- 🐦 Bird
- 🐱 Cat
- 🐶 Dog
...and more
""")

# 🎯 FOOTER
st.markdown("---")
st.caption("🚀 Built with Deep Learning + Streamlit | Portfolio Project")
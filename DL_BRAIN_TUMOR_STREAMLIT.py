# DL_BRAIN_TUMOR_STREAMLIT.py

import os
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="centered"
)

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown("""
    <h1 style='text-align: center; color: #003366; font-size: 2.5em;'>
        🧠 Brain Tumor MRI Classifier
    </h1>
    <hr style='border: 2px solid #003366;'>
    <p style='font-size: 1.2em; color: #444; text-align: center;'>
        Upload a brain MRI image and instantly get a deep learning–powered tumor type prediction.
    </p>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Load Model (PATH FIXED – NO ERROR)
# -------------------------------------------------
@st.cache_resource
def load_brain_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "best_tl_model.h5")
    return load_model(model_path)

model = load_brain_model()

# -------------------------------------------------
# Class Names (MUST MATCH TRAINING ORDER)
# -------------------------------------------------
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# -------------------------------------------------
# File Uploader
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Choose an MRI image (.jpg, .png, .jpeg)",
    type=['jpg', 'png', 'jpeg']
)

# -------------------------------------------------
# Prediction Logic
# -------------------------------------------------
if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_expanded = np.expand_dims(img_array, axis=0)

    st.image(
        img,
        caption="Uploaded MRI Image",
        width=300,
        use_container_width=False
    )

    st.info("Processing image...")

    # Model prediction
    pred_probs = model.predict(img_expanded)[0]
    pred_index = np.argmax(pred_probs)
    pred_class = class_names[pred_index]
    confidence = float(pred_probs[pred_index])

    # -------------------------------------------------
    # Output
    # -------------------------------------------------
    st.success(f"**Prediction:** {pred_class.upper()}")
    st.write(f"**Model Confidence:** {confidence:.2%}")

    st.progress(confidence)

    # -------------------------------------------------
    # Probability Chart
    # -------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.barh(class_names, pred_probs, color="#3CB371")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title("Class Probabilities")

    for i, v in enumerate(pred_probs):
        ax.text(v + 0.01, i, f"{v:.2%}", va="center", fontweight="bold")

    st.pyplot(fig)

    # -------------------------------------------------
    # Tumor Info
    # -------------------------------------------------
    tumor_info = {
        "glioma": "Gliomas originate from glial cells in the brain.",
        "meningioma": "Meningiomas arise from membranes covering the brain and spinal cord.",
        "no_tumor": "No visible tumor detected in this MRI scan.",
        "pituitary": "Pituitary tumors affect the pituitary gland and hormone regulation."
    }

    st.markdown(f"**About:** {tumor_info.get(pred_class, '')}")

else:
    st.warning("Please upload an MRI image to get prediction results.")


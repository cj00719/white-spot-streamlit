
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

st.set_page_config(page_title="White Spot Classifier", layout="centered")

model = tf.keras.models.load_model("white_spot_cnn_model.tflite")
labels = ['Healthy', 'Infected']

st.title("🐟 White Spot Classifier Dashboard")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
results_log = []

def predict(image):
    img = image.resize((128, 128))
    img_array = np.asarray(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0]
    index = np.argmax(prediction)
    confidence = prediction[index]
    return labels[index], confidence

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    label, confidence = predict(image)
    results_log.append((label, confidence))
    st.success(f"Prediction: {label} ({confidence*100:.2f}%)")

    if 'history' not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append((label, confidence))

if st.button("Show Dashboard"):
    st.subheader("📊 Classification Summary")
    history = st.session_state.get("history", [])
    total = len(history)
    healthy = sum(1 for r in history if r[0] == "Healthy")
    infected = sum(1 for r in history if r[0] == "Infected")

    st.write(f"Total Predictions: {total}")
    st.write(f"Healthy: {healthy}")
    st.write(f"Infected: {infected}")

    if history:
        st.write("Recent Results:")
        for i, (lbl, conf) in enumerate(reversed(history[-10:])):
            st.write(f"{i+1}. {lbl} ({conf*100:.2f}%)")

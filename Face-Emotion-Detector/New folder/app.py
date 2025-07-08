import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip().split(" ", 1)[1] for line in f.readlines()]

# Preprocess image
def preprocess_image(img):
    img = ImageOps.fit(img.convert("RGB"), (224, 224), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# Predict function
def predict_image(img):
    input_data = preprocess_image(img)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    index = np.argmax(output)
    return labels[index], output[0][index], output[0]

# Streamlit UI
st.title("🧠 Emotion Predictor using Teachable Machine (.tflite)")
st.write("Upload an image and get emotion prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    label, confidence, all_probs = predict_image(Image.open(uploaded_file))

    st.success(f"**Prediction:** {label}")
    st.info(f"**Confidence:** {confidence * 100:.2f}%")

    st.write("### 🔍 All Class Probabilities:")
    for i, prob in enumerate(all_probs):
        st.write(f"{labels[i]}: {prob * 100:.2f}%")

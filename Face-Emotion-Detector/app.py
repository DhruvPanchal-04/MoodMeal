import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from openai import OpenAI

# ✅ Replace with your GPT-4 API key
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip().split(" ", 1)[1] for line in f.readlines()]

# Image preprocessing
def preprocess_image(img):
    img = ImageOps.fit(img.convert("RGB"), (224, 224), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# Predict mood
def predict_image(img):
    input_data = preprocess_image(img)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    index = np.argmax(output)
    return labels[index], output[0][index], output[0]

# GPT-4 meal suggestion (short version)
def get_meal_for_mood(mood):
    prompt = f"My mood is '{mood}'. Suggest just one simple food or meal to match this mood. Be short and crisp."

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"❌ OpenAI Error: {str(e)}")
        return "⚠️ Error fetching meal suggestion. Please check your API key or GPT model access."

# Streamlit UI
st.set_page_config(page_title="Mood & Meal Predictor", page_icon="🍽️")
st.title("🧠 Mood Detector + Simple Meal Suggestion (GPT-4)")
st.markdown("Upload your image to detect your mood and get a quick food suggestion using GPT-4.")

uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="📸 Uploaded Image", use_column_width=True)

    label, confidence, all_probs = predict_image(Image.open(uploaded_file))

    st.success(f"**🧠 Detected Mood:** {label}")
    st.info(f"**🎯 Confidence:** {confidence * 100:.2f}%")

    st.write("### 📊 Class Probabilities")
    for i, prob in enumerate(all_probs):
        st.write(f"- {labels[i]}: {prob * 100:.2f}%")

    st.write("### 🍽️ GPT-4 Powered Meal Suggestion")
    meal = get_meal_for_mood(label)
    st.markdown(f"💡 **{meal}**")

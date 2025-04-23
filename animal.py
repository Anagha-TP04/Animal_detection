import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# Page Configuration
st.set_page_config(page_title="Animal Detector", page_icon="🐾", layout="wide")

# Load Trained Model
def load_animal_model():
    try:
        model = load_model("Animal_classifier.keras")  # Change to your model path
        return model
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None

# Image Preprocessing
def preprocess_image(image):
    img = image.resize((150, 150))
    img_array = np.array(img)

    if len(img_array.shape) == 3 and img_array.shape[2] > 1:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=[0, -1])

    return img_array

# Prediction Function
def predict_animal(image):
    model = load_animal_model()
    if model is None:
        return None

    img_array = preprocess_image(image)
    prediction = model.predict(img_array)

    class_idx = int(np.argmax(prediction[0]))
    confidence = float(np.max(prediction[0]))
    class_names = ['bears','crows','elephants','rats']

    return {
        'class': class_names[class_idx],
        'confidence': confidence * 100
    }

# Main App
def main():
    st.title("🐾 Animal Detection")

    input_method = st.radio("Select Input Method", ["Upload Image", "Camera Capture"])

    image = None
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Choose an animal image", type=["jpg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
    else:
        camera_input = st.camera_input("Take a picture")
        if camera_input:
            image = Image.open(camera_input).convert('RGB')

    if image:
        st.image(image, width=300, caption="Animal Image")

        if st.button("Detect Animal"):
            with st.spinner("Analyzing..."):
                result = predict_animal(image)

                if result:
                    st.success(f" Prediction: **{result['class']}** with {result['confidence']:.2f}% confidence.")
                    st.info("Model is trained on basic animal classes. Accuracy may vary.")

if __name__ == "__main__":
    main()

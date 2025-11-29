import os
import numpy as np
import tensorflow as tf
from PIL import Image
from huggingface_hub import InferenceClient
from gtts import gTTS
from io import BytesIO
import gdown
import streamlit as st
import base64

# Check or download the model file
MODEL_FILENAME = "aircraft_detector_model.h5"
if not os.path.exists(MODEL_FILENAME):
    file_id = "1DbQ18dFFRiqDJ_G89hyq9HGASo9s9yWz"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, MODEL_FILENAME, quiet=False)

# Aircraft detector class
class AircraftDetector:
    def __init__(self, model_path, api_key, target_size=(224, 224)):
        self.target_size = target_size
        self.model = tf.keras.models.load_model(model_path)
        self.client = InferenceClient(token = api_key)

    def preprocess_image(self, image):
        img = image.convert('RGB').resize(self.target_size)
        img_array = np.asarray(img)
        img_array = np.expand_dims(np.asarray(img), axis=0)
        return img_array

    def detect_aircraft(self, image):
        img_array = self.preprocess_image(image)
        predictions = self.model.predict(img_array)
        classes = [
            'A10', 'A400M', 'AG600', 'AH64', 'AV8B', 'An124', 'An22', 'An225',
            'An72', 'B1', 'B2', 'B21', 'B52', 'Be200', 'C130', 'C17', 'C2',
            'C390', 'C5', 'CH47', 'CL415', 'E2', 'E7', 'EF2000', 'F117',
            'F14', 'F15', 'F16', 'F18', 'F22', 'F35', 'F4', 'H6', 'J10', 'J20',
            'JAS39', 'JF17', 'JH7', 'KC135', 'KF21', 'KJ600', 'Ka27', 'Ka52',
            'MQ9', 'Mi24', 'Mi26', 'Mi28', 'Mig29', 'Mig31', 'Mirage2000', 'P3',
            'RQ4', 'Rafale', 'SR71', 'Su24', 'Su25', 'Su34', 'Su57', 'TB001',
            'TB2', 'Tornado', 'Tu160', 'Tu22M', 'Tu95', 'U2', 'UH60', 'US2',
            'V22', 'Vulcan', 'WZ7', 'XB70', 'Y20', 'YF23', 'Z19'
        ]
        predicted_class = classes[np.argmax(predictions)]
        return predicted_class, float(np.max(predictions))

    def run_model(self, text):
        messages = [{"role": "user", "content": text}]
        completion = self.client.chat.completions.create(
            model="Qwen/Qwen2.5-Coder-32B-Instruct",
            messages=messages,
            max_tokens=500,
        )
        return completion.choices[0].message.content.replace("&", "and")

    def generate_description(self, aircraft_name):
        prompt = f"Provide a short and detailed description of the aircraft named {aircraft_name}."
        return self.run_model(prompt)

    def text_to_speech(self, text):
        try:
            audio_stream = BytesIO()
            info = gTTS(text, lang='en')
            info.write_to_fp(audio_stream)
            audio_stream.seek(0)
            return audio_stream
        except Exception as e:
            st.error(f"Error during text-to-speech conversion: {e}")
            return None

# Streamlit app
st.title("Aircraft Classifier & Info Generator")

uploaded_image = st.file_uploader("Upload an Aircraft Image", type=["jpg", "jpeg", "png"])
api_key = st.text_input("Enter your Hugging Face API Key", type="password")

if st.button("Predict"):
    if uploaded_image is None or not api_key:
        st.error("Please upload an image and provide a valid API key.")
    else:
        try:
            image = Image.open(uploaded_image)
            detector = AircraftDetector(MODEL_FILENAME, api_key)

            with st.spinner("Detecting aircraft..."):
                name, confidence = detector.detect_aircraft(image)
                description = detector.generate_description(name)
                audio_stream = detector.text_to_speech(description)

            st.success("Detection Complete")
            st.image(image, caption=f"Detected: {name} ({confidence:.2%})")
            st.write(f"**Aircraft Name:** {name}")
            st.write(f"**Confidence:** {confidence:.2%}")
            st.write(f"**Description:** {description}")
            if audio_stream:
                audio_bytes = audio_stream.read()
                b64 = base64.b64encode(audio_bytes).decode()
                # st.audio(audio_bytes, format='audio/mp3')

                # Automatically play audio with controls for pause/play
                st.markdown(f'''
                    <audio controls autoplay>
                        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                    </audio>
                ''', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {str(e)}")

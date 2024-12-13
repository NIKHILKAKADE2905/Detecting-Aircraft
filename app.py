# import all necessary libraries
import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, Response
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model # type: ignore
from huggingface_hub import InferenceClient
from PIL import Image
import gdown
from gtts import gTTS
from io import BytesIO



# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'  # Directory to save uploaded images
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Get the current directory
current_folder = os.getcwd()

# List all .h5 files in the current directory
h5_files = [f for f in os.listdir(current_folder) if f.endswith(".h5")]

if "aircraft_detector_model.h5" in h5_files:
    file_path = "aircraft_detector_model.h5"
else:
    file_id = "1DbQ18dFFRiqDJ_G89hyq9HGASo9s9yWz"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "aircraft_detector_model.h5"
    gdown.download(url, output, quiet=False)
    file_path = "aircraft_detector_model.h5"  # Update with your model path

# detect aircraft name using input image and output an information about the aircraft in text and audio format 
class AircraftDetector:
    def __init__(self, model, api_key, target_size=(224, 224)):
        self.target_size = target_size
        self.model = load_model(model)
        self.client = InferenceClient(api_key=api_key)

    def preprocess_image(self, image_path):
        img = Image.open(image_path).convert('RGB')
        if img is None:
            raise FileNotFoundError(f"Image file not found: {image_path}")
        img_resized = img.resize(self.target_size)
        img_array = np.asarray(img_resized)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array

    def detect_aircraft(self, image_path):
        img_array = self.preprocess_image(image_path)
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
        return predicted_class, np.max(predictions)

    def run_model(self, text):
        messages = [{"role": "user", "content": text}]
        completion = self.client.chat.completions.create(
            model="Qwen/Qwen2.5-Coder-32B-Instruct",         
            messages=messages,
            max_tokens=500,
        )
        mytext = completion.choices[0].message.content
        output_text = mytext.replace("&", "and")
        return output_text

    def generate_description(self, aircraft_name):
        prompt = f"Provide a short and detailed description of the aircraft named {aircraft_name}."
        response = {"generated_text": self.run_model(prompt)}
        return response.get("generated_text", "No description generated.")

    def text_to_speech(self, text):
        try:
            audio_stream = BytesIO()
            info = gTTS(text, lang = 'en' )
            info.write_to_fp(audio_stream)
            audio_stream.seek(0)
            return audio_stream 

        except Exception as e:
            print(f"Error during text-to-speech conversion: {e}")
            return None


# Flask route for home page
@app.route('/')
def home():
    return render_template('index.html')  # HTML template for uploading images and entering API key


# Flask route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files or 'api_key' not in request.form:
        return render_template('error.html', error="Image file and API key are required!")

    # Save the uploaded image
    image = request.files['image']
    api_key = request.form['api_key']
    filename = secure_filename(image.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(image_path)

    # Initialize detector
    detector = AircraftDetector(model=file_path, api_key=api_key)

    try:
        # Detect aircraft and generate description
        aircraft_name, confidence = detector.detect_aircraft(image_path)
        description = detector.generate_description(aircraft_name)

        # delete the image after detection
        if os.path.exists(image_path):
                os.remove(image_path)

        # Convert NumPy confidence to a Python float
        confidence = float(confidence)

        # Render the results on the webpage
        return render_template('results.html', 
                               aircraft_name=aircraft_name, 
                               confidence=f"{confidence:.2%}",  # Display confidence as percentage
                               description=description,
                               audio_url= f"/audio?description={description}&api_key={api_key}")

    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/audio', methods=['GET'])
def audio():
    description = request.args.get('description', '')
    if not description:
        return "No description provided", 400

    # Initialize the detector to generate audio
    api_key = request.args.get('api_key')
    detector = AircraftDetector(model=file_path, api_key=api_key)

    audio_stream = detector.text_to_speech(description)

    if not audio_stream:
        return "Error generating audio", 500

    # Serve audio bytes
    return Response(audio_stream, mimetype="audio/mpeg")


# Run the Flask app
if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host="0.0.0.0", port=5000, debug=False)


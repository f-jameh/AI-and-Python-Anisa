# In the name of God


import os
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Set image size (must match training size)
IMG_SIZE = 64  
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create uploads folder if it doesn't exist

# Load the trained CNN model
model = load_model("cnn_image_classification.keras")

# Define class labels (must match training labels)
class_labels = ['fire', 'non_fire']

le = LabelEncoder()
le.fit(class_labels)  # Encode class names

# Function to preprocess an image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None  # Return None if image cannot be read
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize
    img = img.astype(np.float32) / 255.0  # Normalize
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Route for Web UI
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", message="No file part")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", message="No selected file")

        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        img = preprocess_image(file_path)
        if img is None:
            return render_template("index.html", message="Invalid image format")

        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)
        predicted_label = le.inverse_transform([predicted_class])[0]
        confidence = prediction[0][predicted_class]

        return render_template("index.html", message=f"Predicted Class: {predicted_label} with {confidence:.2f} confidence")
    
    return render_template("index.html")

# API route to get predictions via POST request
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    img = preprocess_image(file_path)
    if img is None:
        return jsonify({"error": "Invalid image format"}), 400

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    predicted_label = le.inverse_transform([predicted_class])[0]
    confidence = prediction[0][predicted_class]

    return jsonify({"predicted_class": predicted_label, "confidence": round(float(confidence), 2)})

# Run Flask app
if __name__ == "__main__":
    app.run(host='172.16.1.1', debug=True, use_reloader=False, port=5002)


#### Appendix
# use curl to make prediction
# curl -X POST -F "file=@/path/to/test_image.jpg" http://172.16.1.1:5002/predict

# create API request using python
# import requests

# url = "http://127.0.0.1:5000/predict"
# file_path = "/path/to/test_image.jpg"

# with open(file_path, "rb") as file:
#     response = requests.post(url, files={"file": file})

# print(response.json())

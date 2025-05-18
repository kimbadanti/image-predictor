from flask import Flask, render_template, request
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load the MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Preprocessing function
def prepare_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Prediction function
def predict_image(img):
    processed = prepare_image(img)
    preds = model.predict(processed)
    decoded = decode_predictions(preds, top=1)[0][0]
    return f"{decoded[1]} ({decoded[2]*100:.2f}%)"

# Routes
@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No image uploaded', 400
    file = request.files['image']
    img = Image.open(file.stream)
    prediction = predict_image(img)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

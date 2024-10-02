# main.py
from flask import Flask, render_template, request, redirect, url_for
import os
import tensorflow as tf
from src.test import load_and_preprocess_image

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('model/fake_profile_detector.h5')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join('static', file.filename)
        file.save(file_path)

        # Run prediction
        img = load_and_preprocess_image(file_path)
        predictions = model.predict(img)
        class_names = ['Fake', 'Real']
        prediction = class_names[int(predictions.argmax())]

        # Clean up uploaded file
        os.remove(file_path)

        return render_template('index.html', result=prediction)

if __name__ == "__main__":
    app.run(debug=True)

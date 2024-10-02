# src/test.py
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import sys

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def test_model(image_path):
    model = tf.keras.models.load_model('model/fake_profile_detector.h5')
    img = load_and_preprocess_image(image_path)
    
    predictions = model.predict(img)
    class_names = ['Fake', 'Real']
    prediction = np.argmax(predictions)
    print(f'The profile is {class_names[prediction]}.')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test.py <image_path>")
    else:
        test_model(sys.argv[1])

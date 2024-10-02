import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_model

def train_model():
    dataset_dir = 'C:/fakesocialmedia/data'

    # Debugging: Print directory structure
    print("Dataset directory content:", os.listdir(dataset_dir))
    print("Fake images:", os.listdir(os.path.join(dataset_dir, 'training_fake')))
    print("Real images:", os.listdir(os.path.join(dataset_dir, 'training_real')))

    # Data Preprocessing
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        subset='training',
        classes=['training_fake', 'training_real']
    )

    validation_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        subset='validation',
        classes=['training_fake', 'training_real']
    )

    # Debugging: Check if generators have data
    print(f"Found {train_generator.samples} training samples.")
    print(f"Found {validation_generator.samples} validation samples.")

    # Check if any images were found
    if train_generator.samples == 0 or validation_generator.samples == 0:
        raise ValueError("No images found in the provided dataset directories. Please check your dataset paths and ensure there are images in the 'training_fake' and 'training_real' directories.")

    # Create model
    model = create_model()

    # Train model
    model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator
    )

    # Save model
    if not os.path.exists('model'):
        os.makedirs('model')
    model.save('model/fake_profile_detector.h5')

if __name__ == "__main__":
    train_model()

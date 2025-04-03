import tensorflow as tf
import os
import numpy as np
from keras_preprocessing import image

# Path to the model and the test image
model_path = 'beanie_classifier_model.keras'  # Or the path where your model is saved
test_image_path = r"C:\Users\Dell\Documents\Hackathon\Phase 1 Beaniee\testbeanie"  # Your test image folder

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Load and preprocess the test image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to match model's input size
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Test the image (You can loop through multiple images if needed)
test_image = os.path.join(test_image_path, "test.jpg")  # Replace with actual image filename
image_data = preprocess_image(test_image)

# Predict using the model
prediction = model.predict(image_data)

# Display the result
if prediction[0] > 0.5:
    print("Beanie detected")
else:
    print("No beanie detected")

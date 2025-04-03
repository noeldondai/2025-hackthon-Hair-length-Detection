import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths to your folders
short_hair_path = r"C:\Users\Dell\Documents\Hackathon\Phase 1 Beaniee\shorthair"
long_hair_path = r"C:\Users\Dell\Documents\Hackathon\Phase 1 Beaniee\longhair"

# Load and preprocess images: focus on head region
def crop_to_head_region(img_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cropped = img[y:y+h, x:x+w]
        return cv2.resize(cropped, (224, 224))  # Resize to match model input size
    return None  # Return None if no face is detected

def preprocess_images(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        cropped = crop_to_head_region(img_path)
        if cropped is not None:
            images.append(cropped / 255.0)  # Normalize
            labels.append(label)
    return images, labels

# Preprocess both categories
images_short_hair, labels_short_hair = preprocess_images(short_hair_path, 0)
images_long_hair, labels_long_hair = preprocess_images(long_hair_path, 1)

# Combine data
images = np.array(images_short_hair + images_long_hair)
labels = np.array(labels_short_hair + labels_long_hair)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Pretrained model with transfer learning
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base layers

# Build the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with data augmentation
train_gen = datagen.flow(X_train, y_train, batch_size=32)
model.fit(train_gen, epochs=10, validation_data=(X_test, y_test))

# Save the model
model.save('hair_length_model.keras')

print("Hair length model training completed and saved!")

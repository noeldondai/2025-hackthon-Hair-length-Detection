import cv2
import numpy as np
from keras_preprocessing import image
import tensorflow as tf

# Load the trained model (make sure to specify the correct path)
model = tf.keras.models.load_model('beanie_classifier_model.h5')

# Load the pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Flip the frame horizontally to correct the mirroring effect
    frame = cv2.flip(frame, 1)
    
    # Convert the frame to grayscale (for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Check if faces are detected
    if len(faces) > 0:
        # For each detected face, crop the region of interest (ROI) to focus on the head
        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]
            
            # Preprocess the cropped image for the beanie classifier model
            roi = cv2.resize(roi, (224, 224))  # Resize to match model input size
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)  # Convert to RGB
            roi = np.array(roi) / 255.0  # Normalize
            roi = np.expand_dims(roi, axis=0)  # Add batch dimension
            
            # Predict if the person is wearing a beanie
            prediction = model.predict(roi)
            
            # Display text based on the prediction
            if prediction[0] > 0.5:
                message = "You are wearing a beanie!"
            else:
                message = "No beanie detected."
            
            # Draw rectangle around the face and display the message
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, message, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    else:
        message = "No face detected."
        cv2.putText(frame, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Headwear Detection', frame)
    
    # Check if 'q' is pressed or if the window is closed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty('Headwear Detection', cv2.WND_PROP_VISIBLE) < 1:
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

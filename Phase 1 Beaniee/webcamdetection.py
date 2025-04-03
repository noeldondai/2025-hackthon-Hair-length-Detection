import cv2
import numpy as np
from keras.models import load_model
import requests
import mediapipe as mp
import time

# Load the models
beanie_model = load_model('beanie_classifier_model.keras')
hair_model = load_model(r'C:\Users\Dell\Documents\Hackathon\Phase 1 Beaniee\hair_length_model.keras')

# Load the face detection cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Discord webhook URL
WEBHOOK_URL = 'https://discord.com/api/webhooks/1330446850268069951/pAXbt5VmbF7ZXTBDccm8EiYkk4_Rag0H6eb2IkEcpG55RcyXctP4s5LGS5pcRZz4z73q'

# Function to send a message to Discord
def send_discord_message(content):
    try:
        data = {"content": content}
        response = requests.post(WEBHOOK_URL, json=data)
        if response.status_code == 204:
            print(f'Message sent: {content}')
        else:
            print(f'Failed to send message. Status code: {response.status_code}')
    except Exception as e:
        print(f"Error sending message: {e}")

# Function to detect thumbs-up using MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Function to check if thumbs-up is detected
def detect_thumbsup(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Check if thumb is up by comparing y-coordinates of thumb tip and base
            thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_base = landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
            if thumb_tip.y < thumb_base.y:
                return True
    return False

# Function for beanie detection
def detect_beanie_and_face(frame, prev_state):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return frame, prev_state

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face, (224, 224))
        face_img = np.expand_dims(face_img, axis=0)
        face_img = face_img / 255.0
        prediction = beanie_model.predict(face_img)

        if prediction[0][0] > 0.5:  # Beanie detected
            text = "Beanie Detected"
            color = (0, 255, 0)
            prev_state = "beanie_detected"
        else:  # No beanie detected
            text = "No Beanie Detected"
            color = (0, 0, 255)
            prev_state = "no_beanie"

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return frame, prev_state

# Function to detect hair length
def detect_hair_length(frame, faces):
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face, (224, 224))
        face_img = np.expand_dims(face_img, axis=0)
        face_img = face_img / 255.0

        # Hair length prediction
        prediction = hair_model.predict(face_img)
        if prediction[0][0] > 0.5:
            text = "Please cut your hair!"
        else:
            text = "Good hair, smart boy!"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # If hair is long, send Discord message
        if prediction[0][0] > 0.5:
            send_discord_message("A student with long hair has appeared!")

    return frame, text

# Initialize video capture
cap = cv2.VideoCapture(0)
prev_state = "none"
thumbsup_detected = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally

    # Detect beanie and face first
    frame, prev_state = detect_beanie_and_face(frame, prev_state)

    # Check for thumbs-up after no beanie detected
    if prev_state == "no_beanie" and not thumbsup_detected:
        if detect_thumbsup(frame):
            thumbsup_detected = True
            print("Thumbs-up detected. Starting hair detection...")
            time.sleep(1)  # Wait for a short period before moving to hair detection

    # Once thumbs-up is detected, perform hair detection
    if thumbsup_detected:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            frame, hair_message = detect_hair_length(frame, faces)
            cv2.putText(frame, hair_message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Check the hair type
            if hair_message == "Please cut your hair!":
                send_discord_message("A student with long hair has appeared!")
                print("A student with long hair has appeared!")
                break  # Close the webcam

            # If short hair, print message and close webcam
            elif hair_message == "Good hair, smart boy!":
                print("Good hair, smart boy!")
                break  # Close the webcam

    cv2.imshow("Hair and Beanie Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

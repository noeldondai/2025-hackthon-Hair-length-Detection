import cv2
import numpy as np

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect beard and blur the background
def detect_beard(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    
    # Create a blurred version of the frame
    blurred_frame = cv2.GaussianBlur(frame, (51, 51), 30)
    
    # Create a mask for isolating the face region
    mask = np.zeros_like(frame, dtype=np.uint8)
    result_text = ""  # Placeholder for result text
    text_color = (0, 0, 255)  # Default color for text (red)

    if len(faces) == 0:
        result_text = "No face detected. Position your face properly."
    else:
        for (x, y, w, h) in faces:
            # Draw the face region on the mask
            cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)
            
            # Define the ROI for beard detection (lower 60% of the face)
            roi = frame[y + int(h * 0.6):y + h, x:x + w]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(roi_gray, 50, 150)
            
            # Use a threshold to determine edge density
            edge_density = cv2.countNonZero(edges) / (roi.shape[0] * roi.shape[1])

            if edge_density > 0.05:  # Adjust threshold for sensitivity
                result_text = "Beard Detected"
                text_color = (0, 255, 0)  # Green for detection
                cv2.rectangle(frame, (x, y + int(h * 0.6)), (x + w, y + h), (0, 255, 0), 2)
            else:
                result_text = "Clean Shave Detected"
                text_color = (0, 0, 255)  # Red for clean shave
                cv2.rectangle(frame, (x, y + int(h * 0.6)), (x + w, y + h), (0, 0, 255), 2)

    # Combine the blurred frame with the original frame using the mask
    mask_inv = cv2.bitwise_not(mask)
    blurred_background = cv2.bitwise_and(blurred_frame, mask_inv)
    face_region = cv2.bitwise_and(frame, mask)
    output = cv2.add(face_region, blurred_background)

    # Add the result text to the final output
    if result_text:
        cv2.putText(output, result_text, (50, frame.shape[0] - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

    return output

# Webcam feed for real-time detection
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_with_detection = detect_beard(frame)
    cv2.imshow("Beard Detection with Background Blur", frame_with_detection)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
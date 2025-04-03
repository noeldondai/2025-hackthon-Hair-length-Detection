from flask import Flask
import subprocess

app = Flask(__name__)

@app.route('/run-webcam-detection')
def run_webcam_detection():
    print("Received request for webcam detection")  # Debugging message
    subprocess.Popen(["python", "C:/Users/Dell/Documents/Hackathon/Phase 1 Beaniee/webcamdetection.py"])
    return 'Webcam detection started', 200

@app.route('/run-beard-detection')
def run_beard_detection():
    print("Received request for beard detection")  # Debugging message
    subprocess.Popen(["python", "C:/Users/Dell/Documents/Hackathon/Phase 1 Beaniee/beard.py"])
    return 'Beard detection started', 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

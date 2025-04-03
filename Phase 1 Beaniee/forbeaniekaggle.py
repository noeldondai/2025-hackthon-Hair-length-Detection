import kagglehub

# Download the dataset and save it in a new folder
path = kagglehub.dataset_download("andrewmvd/hard-hat-detection", folder="phase_2_helmet_detection")

print("Path to dataset files:", path)

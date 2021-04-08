import cv2
from tensorflow.keras.models import load_model

# Loading the trained CNN model for age classification, and
# defining a list of age-ranges as defined in the model.
model = load_model("age_detect_cnn_model.h5")
age_ranges = ['1-2', '3-9', '10-20', '21-27', '28-45', '46-65', '66-116']

# Importing the Haar Cascades classifier XML file.
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")





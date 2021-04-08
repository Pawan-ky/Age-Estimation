import numpy as np
import cv2
from Models import face_cascade, age_ranges, model


def shrink_face_roi(x, y, w, h, scale=0.9):
    # Defining a function to shrink the detected face region by a scale for better prediction in the model
    wh_multiplier = (1 - scale) / 2
    x_new = int(x + (w * wh_multiplier))
    y_new = int(y + (h * wh_multiplier))
    w_new = int(w * scale)
    h_new = int(h * scale)
    return x_new, y_new, w_new, h_new


# Defining a function to find faces in an image and then classify each found face into age-ranges defined above
def classify_age(img):
    # Making a copy of the image for overlay of ages and making a grayscale copy for passing to the loaded model for
    # age classification.
    img_copy = np.copy(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detecting faces in the image using the face_cascade loaded above and storing their coordinates into a list.
    faces = face_cascade.detectMultiScale(img_copy, scaleFactor=1.2, minNeighbors=6, minSize=(100, 100))
    # print(f"{len(faces)} faces found.")

    if len(faces) < 0:
        print('No Face Found Please try again')
        return 'No Face'
    elif len(faces) > 1:
        print('More than 1 face found')
        return 'More faces'
    else:

        # Looping through each face found in the image.
        for i, (x, y, w, h) in enumerate(faces):
            # Drawing a rectangle around the found face.
            face_rect = cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 100, 0), thickness=2)

            # Predicting the age of the found face using the model loaded above.
            x2, y2, w2, h2 = shrink_face_roi(x, y, w, h)
            face_roi = img_gray[y2:y2 + h2, x2:x2 + w2]
            face_roi = cv2.resize(face_roi, (200, 200))
            face_roi = face_roi.reshape(-1, 200, 200, 1)
            face_age = age_ranges[np.argmax(model.predict(face_roi))]
            face_age_pct = round(np.max(model.predict(face_roi)) * 100, 2)  # estimation confidence

    return face_age, face_age_pct  # returns age range and confidence

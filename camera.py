import cv2
from model import FacialExpressionModel
import numpy as np
import base64

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX
last_detected_emotion = None

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions

    def get_frame(self):
        global last_detected_emotion
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) # Convert to black and white
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)

        detected_emotion = None

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]

            roi = cv2.resize(fc, (48, 48))
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            # Save the detected emotion
            last_detected_emotion = pred

            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)

        _, jpeg = cv2.imencode('.jpg', fr)

        # Create a dictionary with emotion and frame
        return  jpeg.tobytes()
    
    def get_last_detected_emotion(self):
        global last_detected_emotion
        return last_detected_emotion

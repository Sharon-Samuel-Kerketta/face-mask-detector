import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

casc_path = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
face_cascade = cv2.CascadeClassifier(casc_path)
model = load_model('./src/modelling/model_resnet_50/best_model.hdf5')


def process_frame(frame):

    frame = cv2.flip(frame, flipCode = 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.05, minNeighbors= 5, minSize = (60,60), flags = cv2.CASCADE_SCALE_IMAGE)

    faces_list = []
    preds = []
    label = ""
    color = ()
    for (x,y,w,h) in faces:
        face_frame = frame[y:y+h, x :x+w]
        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        face_frame = cv2.resize(face_frame, (224,224))
        face_frame = img_to_array(face_frame)
        # face_frame = face_frame / 255.0
        face_frame = np.expand_dims(face_frame, axis = 0)
        face_frame = preprocess_input(face_frame)

        preds = model.predict(face_frame)
    
        for pred in preds:
            print(pred)
            (withoutmask, mask) = pred

            label = "Mask" if mask > withoutmask else "No Mask"
            color = (0,255,0) if label == "Mask" else (0,0,255)
            label = "{} : {:.2f}%".format(label, max(mask, withoutmask))
        
        cv2.putText(frame, label, (x, y -10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (x,y), (x + w, y + h), color, 2)

    return frame

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

try :
    model = load_model('./src/best_model.hdf5')
except :
    print("Wrong Model Directory provided")
    exit()


video_name = "vid4.mp4"
path_to_video = "./video/input/"+ video_name

if os.path.isfile(path_to_video):
    video_capture = cv2.VideoCapture(path_to_video)
    codec = int(video_capture.get(cv2.CAP_PROP_FOURCC))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    frame_width  = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height  = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video = cv2.VideoWriter(f"./video/output/{video_name}", codec, fps, (frame_width, frame_height))

    length = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_number = 0

    while True:
        ret, frame = video_capture.read()

        if not ret:
            break

        frame_number += 1
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

        cv2.imshow("vid", frame)
        print(f"Writing frame {frame_number}/{length}")
        output_video.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    output_video.release()
    cv2.destroyAllWindows()

else :
    print("No file exists!")
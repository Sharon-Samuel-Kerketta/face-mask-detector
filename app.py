from src.inference import detect_realtime
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template, request, url_for, send_from_directory, redirect
import threading
import argparse
import datetime
import imutils
import cv2
import time
import os

output_frame = None
lock = threading.Lock()

## Streaming froom the Webcam
video_capture = None

## initializing the Flask
app = Flask(__name__)

UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/video/input/'

@app.route('/', methods=['POST', 'GET'])
def index():

    global video_capture

    if request.method == 'POST':
        if 'file' not in request.files:
            print("No files attatched")
            return redirect(request.url)
        file = request.files['file']
        filename = file.filename
        file.save(os.path.join(UPLOAD_FOLDER, filename))

    if video_capture is not None:
        video_capture.release()

    return render_template("index.html")

def detect():
    global output_frame, lock
    while True: 
        if video_capture.read() is not None:
            try :
                ret, frame = video_capture.read()
                frame = detect_realtime.process_frame(frame)
                with lock:
                    output_frame = frame.copy()
            except:
                pass
        else:
            break
    return


def generate():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                continue
            flag, encoded_image = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n' )  

@app.route("/video_feed")
def video_feed():
    global video_capture
    video_capture = cv2.VideoCapture(0)
    t = threading.Thread(target = detect)
    t.daemon = True
    t.start()
    return Response(generate(), mimetype= "multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run()
    video_capture.release()
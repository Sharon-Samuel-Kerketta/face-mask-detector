from src.inference import detect_realtime
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import cv2
import time

output_frame = None
lock = threading.Lock()

app = Flask(__name__)

video_capture = VideoStream(src= 0).start()

@app.route('/')
def index():
    return render_template("index.html")

def detect():

    global output_frame, lock
    while True:
        frame = video_capture.read()
        frame = detect_realtime.process_frame(frame)


        with lock:
            output_frame = frame.copy()

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
    return Response(generate(), mimetype= "multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    t = threading.Thread(target = detect)
    t.daemon = True
    t.start()

    app.run()
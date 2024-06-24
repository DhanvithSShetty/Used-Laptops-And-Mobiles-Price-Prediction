import os
import cv2
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, Response

app = Flask(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
laptop_model_path = os.path.join(current_dir, "models", "laptop_price_model.pkl")
mobile_model_path = os.path.join(current_dir, "models", "mobile_price_model.pkl")

yolo_dir = os.path.join(current_dir, "yolo")
yolo_config = os.path.join(yolo_dir, "yolov3.cfg")
yolo_weights = os.path.join(yolo_dir, "yolov3.weights")
yolo_names = os.path.join(yolo_dir, "coco.names")

try:
    laptop_model = joblib.load(laptop_model_path)
    mobile_model = joblib.load(mobile_model_path)
except FileNotFoundError as e:
    print(f"Error loading model: {e}")
    exit(1)

net = cv2.dnn.readNetFromDarknet(yolo_config, yolo_weights)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
with open(yolo_names, "r") as f:
    classes = [line.strip() for line in f.readlines()]

broken_sound_path = os.path.join(current_dir, "sounds", "broken_sound.wav")
not_broken_sound_path = os.path.join(current_dir, "sounds", "not_broken_sound.wav")
broken_sound = cv2.VideoCapture(broken_sound_path)
not_broken_sound = cv2.VideoCapture(not_broken_sound_path)

def preprocess_laptop_input(form_data):
    input_data = pd.DataFrame({
        "Age(Years)": [int(form_data['age'])],
        "Battery Life (hrs)": [int(form_data['battery'])],
        "Screen Size (inches)": [float(form_data['screen'])],
        "Brand": [form_data['brand']],
        "Model": [form_data['model']],
        "Condition": [form_data['condition']],
        "Specifications": [form_data['specs']],
        "Storage Type": [form_data['storage']]
    })
    return input_data

def preprocess_mobile_input(form_data):
    input_data = pd.DataFrame({
        "Age(Years)": [int(form_data['age'])],
        "Brand": [form_data['brand']],
        "Model": [form_data['model']],
        "Condition": [form_data['condition']],
        "Specifications": [form_data['specs']]
    })
    return input_data

def preprocess_scan_input(form_data):
    input_data = pd.DataFrame({
        "Age(Years)": [int(form_data['age'])],
        "Brand": [form_data['brand']],
        "Model": [form_data['model']],
        "Condition": [form_data['condition']],
        "Specifications": [form_data['specs']],
        "Original_Price": [int(form_data['original_price'])],
        "Is_Broken": [form_data['is_broken'] == "yes"]
    })
    return input_data

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html")

@app.route("/contact", methods=["GET"])
def contact():
    return render_template("contact.html")

@app.route("/predict_laptop", methods=["GET", "POST"])
def predict_laptop():
    if request.method == "POST":
        form_data = request.form
        input_data = preprocess_laptop_input(form_data)
        prediction = laptop_model.predict(input_data)[0]
        return render_template("predict_laptop.html", prediction=prediction)
    return render_template("predict_laptop.html")

@app.route("/predict_mobile", methods=["GET", "POST"])
def predict_mobile():
    if request.method == "POST":
        form_data = request.form
        input_data = preprocess_mobile_input(form_data)
        prediction = mobile_model.predict(input_data)[0]
        return render_template("predict_mobile.html", prediction=prediction)
    return render_template("predict_mobile.html")

@app.route("/scan_gadget", methods=["GET", "POST"])
def scan_gadget():
    if request.method == "POST":
        form_data = request.form
        input_data = preprocess_scan_input(form_data)
        amount = input_data["Original_Price"].values[0] * (0.5 if input_data["Is_Broken"].values[0] else 0.8)
        
        if input_data["Is_Broken"].values[0]:
            broken_sound.set(cv2.CAP_PROP_POS_FRAMES, 0)
            broken_sound.read()
        else:
            not_broken_sound.set(cv2.CAP_PROP_POS_FRAMES, 0)
            not_broken_sound.read()
        
        return render_template("scan_gadget.html", amount=amount)
    return render_template("scan_gadget.html")

def locate_objects(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            object_image = frame[y:y+h, x:x+w]
            is_broken_cellphone = False
            prediction_text = "Broken Cell Phone" if is_broken_cellphone else "Not Broken Cell Phone"
            cv2.putText(frame, prediction_text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if is_broken_cellphone:
                broken_sound.set(cv2.CAP_PROP_POS_FRAMES, 0)
                broken_sound.read()
            else:
                not_broken_sound.set(cv2.CAP_PROP_POS_FRAMES, 0)
                not_broken_sound.read()

    return frame

def gen_frames():
    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = locate_objects(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import ultralytics
from ultralytics import YOLO
import torch
import numpy as np
app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
#from roboflow import Roboflow
#rf = Roboflow(api_key="eEuewAprXEIsFbptBZTC")
#project = rf.workspace("myproject-vm4hg").project("id-card-detection-xtiwy")
#dataset = project.version(3).download("yolov8")
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
model = YOLO('best.pt')
#model.train(data="D:/AI projects/SB_ID Detect/data.yaml" , epochs=10,conf=0.25)
import cv2
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        results = model.predict(image, conf=0.25)
        box = results[0].boxes.xyxy

        int(box[0][0])
        x1= int(box[0][0])
        y1= int(box[0][1])
        x2= int(box[0][2])
        y2= int(box[0][3])
        #image = image[y1:y2, x1:x2]
        cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,0),5)
        cv2.putText(image,"ID-CARD",(x1,y1),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),5)
        filename = secure_filename(file.filename)
        #file.reshape(500,800)
        #cropped_filename = 'cropped_' + filename
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'],filename), image)
        
        #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display/<filename>')
def display_image(filename):
    print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 
if __name__ == "__main__":
    
    app.run()
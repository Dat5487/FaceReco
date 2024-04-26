import flask
from flask import render_template, request
import os
from flask import request,jsonify
import uuid
import cv2
from flask import Flask
from search_face import SearchFace
from inference import Inference
from face_detection import FaceDetection
app = Flask(__name__)



inference = Inference()
faceSearch = SearchFace()
face_detection = FaceDetection()
@app.route('/upload/<mahocsinh>', methods=['GET', 'POST'])
def upload(mahocsinh=None):
    if request.method == 'POST':
        file = request.files['file']
        extension = os.path.splitext(file.filename)[1]
        f_name = str(uuid.uuid4()) + extension
        path = 'faces/'+f_name
        file.save(path)
        image = cv2.imread('faces/'+f_name)
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        boxes = face_detection.detect(image)
        print(len(boxes))
        if len(boxes)>0:
            box = boxes[0]  
            print(box)
            box = list(map(int, box))
            x_min, y_min, x_max, y_max = box
            face = image[y_min:y_max,x_min:x_max]
            cv2.imwrite(path,face)
            emb = inference.inference(face)
            faceSearch.save_file(emb,mahocsinh)
            return jsonify('ok')
        else:
            return jsonify('-1')
    return render_template("index2.html")

@app.route('/face_reg',methods = ['GET','POST'])
def search_face():
    if request.method == 'POST':
        file = request.files['file']
        extension = os.path.splitext(file.filename)[1]
        f_name = str(uuid.uuid4()) + extension
        path = 'faces/'+f_name
        file.save(path)
        image = cv2.imread('faces/'+f_name)
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        boxes = face_detection.detect(image)
        if len(boxes)>0:
            box = boxes[0]  
            box = list(map(int, box))
            x_min, y_min, x_max, y_max = box
            face = image[y_min:y_max,x_min:x_max]
            cv2.imwrite(path,face)
            emb = inference.inference(face)
            id = faceSearch.search(emb)
            return jsonify(str(id))
        else:
            return jsonify('err')
    return render_template("index.html")
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1234)

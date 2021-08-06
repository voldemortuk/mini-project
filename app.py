from flask import Flask, render_template, request, session, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
import keras
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torchvision import transforms
import pickle
from PIL import Image
import torchvision
import torch.nn as nn
import torch
from flask import Flask
from flask_ngrok import run_with_ngrok

basedir = os.path.abspath(os.path.dirname(__file__))

class DenseNet121(nn.Module):
    def __init__(self):
        super(DenseNet121, self).__init__()
        self.model = torchvision.models.densenet121(pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Linear(1024, 15),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x

class MyCustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "app"
        return super().find_class(module, name)

with open(basedir+'/chexnet.pkl','rb') as f:
    unpickler = MyCustomUnpickler(f)
    model = unpickler.load()


# model=pickle.load(open(basedir+'/LR_model.pkl','rb'))
model=pickle.load(open(basedir+'/chexnet.pkl','rb'))
covid_model=pickle.load(open(basedir+'/LR_model.pkl','rb'))
app=Flask(__name__)
run_with_ngrok(app)   

UPLOAD_FOLDER="static/image"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif','jfif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/')
def hello_world():
    return render_template("index.html")
@app.route('/overall',methods=["GET","POST"])
def overall():
    if request.method=="POST":
        image_file=request.files['file']
        print("image ",image_file)
        if image_file:
            filename = secure_filename(image_file.filename)
            image_file.save(os.path.join(basedir,app.config['UPLOAD_FOLDER'], filename))
            location = os.path.join(basedir,app.config['UPLOAD_FOLDER'], filename)
            print(location)
            # test_image = cv2.imread(location)
            # print(test_image)
            # test_image = cv2.cvtColor(test_image, cv2.IMREAD_GRAYSCALE)
            # test_image = cv2.resize(test_image, (224, 224))
            # test_img = test_image.flatten().reshape(1, -1)

            classes = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis','Effusion', 'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Hernia', 'Mass', 'No Finding']
            image = Image.open(location).convert('RGB')
            # normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # preprocess = transforms.Compose([
            #     transforms.Resize(256),
            #     transforms.TenCrop(224),
            #     transforms.Lambda
            #     (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            #     transforms.Lambda
            #     (lambda crops: torch.stack([normalize(crop) for crop in crops]))
            # ])
            preprocess = transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                   ])
            image = preprocess(image)
            image = image.unsqueeze(0)
            print(image.size())
            # images = image.cuda()
            print(image.size())
            outputs = model(image)
            index_tensor = torch.argmax(outputs)
            index = index_tensor.item()
            print(classes[index])
            # prediction=model.predict(test_img)
            # prediction=model(image)

            # output = model(image)
            # index_tensor = torch.argmax(output)
            # prediction = index_tensor.item()
            # print(classes[prediction])
    return render_template('overall.html',prediction=classes[index],xray_image=image_file.filename)


@app.route('/covid',methods=["GET","POST"])
def covid():
    if request.method=="POST":
        image_file=request.files['file']
        print("image ",image_file)
        if image_file:
            filename = secure_filename(image_file.filename)
            image_file.save(os.path.join(basedir,app.config['UPLOAD_FOLDER'], filename))
            location = os.path.join(basedir,app.config['UPLOAD_FOLDER'], filename)
            print("covid ",location)
            test_image = cv2.imread(location)
            test_image = cv2.cvtColor(test_image, cv2.IMREAD_GRAYSCALE)
            test_image = cv2.resize(test_image, (224, 224))
            test_img = test_image.flatten().reshape(1, -1)

            prediction=covid_model.predict(test_img)
            print(prediction)
    return render_template('covid.html',prediction=prediction,xray_image=image_file.filename)


print(UPLOAD_FOLDER)
if __name__=="__main__":
    app.run()


# X-ray Abnormality detection model
## Introduction 
The Machine learning model was developed in Python programming lan- guage using Pytorch and implemented in JavaScript. 
The UI was created using HTML and Tailwind CSS. ChexNet is a deep learning algorithm that can detect and localize 14 kinds of diseases from chest X-ray images. 
Here we have implemented the [CheXNet](https://stanfordmlgroup.github.io/projects/chexnet/) using Python3 (Pytorch) and used Flask to integrate with our web-app.
Our Web-app also detects COVID-19 using X-ray images for which we used the [COVID-19 chest xray](https://www.kaggle.com/bachrr/covid-chest-xray) dataset on different ML models.



## Dataset

The [ChestX-ray14 dataset](https://www.kaggle.com/nih-chest-xrays/datae) comprises 112,120 frontal-view chest X-ray images of 30,805 unique patients with 14 disease labels. 

## Prerequisites

- Python 3.9
- [PyTorch](http://pytorch.org/) and its dependencies
- Flask 2.0.1

## Usage

1. Download images of ChestX-ray14 from this [released page](https://nihcc.app.box.com/v/ChestXray-NIHCC) and decompress them to the directory [images](./ChestX-ray14/images).

2. Download this [colab notebook](https://github.com/ravijyoti3/mini-project/blob/main/web_mini_project.ipynb) and run locally 

## Web-App

### Home Page 
![ModelOutput](https://github.com/ravijyoti3/mini-project/blob/main/Images/homepage.jpeg)

### Image Upload
![ModelOutput](https://github.com/ravijyoti3/mini-project/blob/main/Images/upload.jpeg)
##Result
ChexNet outputs a vector t of binary labels indicating the ab- sence or presence of each of the following 14 pathology classes: Atelec- tasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nod-ule, Pleural Thickening, Pneumonia, and Pneumotho- rax. 

### Model Output 
![ModelOutpu](https://github.com/ravijyoti3/mini-project/blob/main/Images/Model%20output.png)
We have implemented it using py- torch and replaced the final fully connected layer in CheXNet with a fully connected layer producing a 15-dimensional output, after which we apply an elementwise sigmoid nonlinearity. The final output is the predicted probability of the presence of each pathology class.

## Conclusion 

The main objective of the project was to build a Web-app inte- grated with our ML model is to detect and localise different diseases through X-ray images.
Also in recent times COVID-19 virus has spread primarily through droplets of saliva or discharge from the nose when an infected person coughs or sneezes, so it’s important that you also practice respiratory etiquette. So our final web-app also will detect if the input X-ray image has symptoms or COVID-19 or not.
In conclusion this algorithm can and should save lives in many parts of the world by assisting medical staff which lacks skilled radi- ologists or assist radiologists directly.

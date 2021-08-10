# X-ray Abnormality detection model
## Introduction 

The Machine learning model was developed in Python programming lan- guage using Pytorch and implemented in JavaScript. The UI was created us- ing HTML and Tailwind CSS.

## Dataset

The [ChestX-ray14 dataset](http://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf) comprises 112,120 frontal-view chest X-ray images of 30,805 unique patients with 14 disease labels. To evaluate the model, we randomly split the dataset into training (70%), validation (10%) and test (20%) sets, following the work in paper. Partitioned image names and corresponding labels are placed under the directory [labels](./ChestX-ray14/labels).

## Prerequisites

- Python 3.9
- [PyTorch](http://pytorch.org/) and its dependencies
- Flask 2.0.1

##Web-App

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

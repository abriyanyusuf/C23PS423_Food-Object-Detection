![LOGO IDOEAT](https://drive.google.com/uc?export=view&id=1DYFfsy8oyIwLu7Q9Ztt-X11UhZcCg67a)
<h2>A Food Object Detection Model Build With TensorFlow Object Detection</h2>
<h1></h1>
<p>Hi. This is repository for Capstone Project Bangkit 2023. This repository contains files to build food object detection features.</p>

## Contributor
|            Name          |  Bangkit ID  |       Path       |
|:------------------------:|:------------:|:----------------:|
|  Abriyan Yusuf           |  M169DSX1911 | Machine Learning |

## Simple Description About Our Project
We aim to detect several food inside an image. For food object recognition, we use the TensorFlow Object Detection base which is able to recognize food objects as follows.
1. Apple
2. Banana
3. Bread
4. Egg
5. Bun
6. Pear
7. Litchi
8. Orange
9. Qiwi
10. Tomato
11. Sachima
12. Fire Dough Twist
13. Grape
14. Mango
15. Lemon
16. Mooncake
17. Peach
18. Plum
19. Doughnut
<br>

## Our Infrastructure
![LOGO TFLITE](https://drive.google.com/uc?export=view&id=1xVOU6Uj_XXDDpkU2bqBn84tKeDuK70kc)
<br>
![Base Model SSD MobileNet V2](https://84771188-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FGEgcCk4PkS5Pa6uBabld%2Fuploads%2Fgit-blob-519584b8c0f51d08a60b0a4b0821ac572eb9de5c%2Fcnn-network-example.jpeg?alt=media)
<br>
We use SSD MobileNet V2 FPNlite 320x320 model. In the MobileNetV2 SSD FPN-Lite, we have a base network (MobileNetV2), a detection network (Single Shot Detector or SSD) and a feature extractor (FPN-Lite). We choose this model because we want deploy our model in Android or Edge Device. 

## mAP for Current Model and Dataset
We already did the training session for 40.000 epochs and it consumed 12 Hours in Colab. We decided to use standard model (non quantized) that support float32 image. The Overall mAP is 94.81%
<br>
![mAP](https://drive.google.com/uc?export=view&id=16Y7IKysIcEiB-oslmevqzhse6QJD-c96)

## Important Notes
The accuracy of this model depends on the quality of the input image and the shape of the image itself. As we can see, the images contained in the dataset only contain the food object itself, a white plate, and also coins. This shows that there are not too many objects in the image which makes the features more numerous, allowing for high detection accuracy. The technique in testing the accuracy is to upload the image to the model and not taken directly using a camera device. If the input image is taken directly using a webcam or camera on a smartphone when we deploy the model in the application, the detection results will be different depending on the ability of the camera to provide good image quality. 
<br>

![inference test](https://drive.google.com/uc?export=view&id=1EJExsiQCU1dcAFpoArKJ3uhM9vjQEQpP)

<br>
In the documentation of the dataset used, the coin in the dataset is an additional object intended as a reference for calculating the volume of food. In this project we do not use it for that purpose. Therefore, this coin object can be ignored when it appears in the detection results.

## Files Inside This Repository
- `TrainObjDetect.ipynb` => This notebook is our main file that we use to run this project
- `Create_CSV_from_VOC.py` => This file is used inside main file to convert out dataset in VOC format to CSV
- `CSV_to_TFRecord` => This file is used inside main file to convert our CSV dataset to TFRecord
- `Split_to_TrainValTest.py` => This file is used inside main file to split our data into train, validation, and test folder randomly.
- `Inference_Testing_Model_and_mAP_calculation.ipynb` => This notebook is used to calculate mAP of our model start by importing dataset.
- `calculate_map.py` => This file is used in our Inference Testing notebook to calculate mAP.
- `Script_For_Testing_and_Analyze_TFLite_Model.ipynb` => This notebook is used to check input and output of our model to fix issue when we deploy it to Android Apps.
- `Adding_Metadata_Object_Detection.ipynb` => This notebook is used to add metadata to our model because Android Studio only accept model that has metadata inside. **Note : You only can run this file using local machine (e.g. Jupyter Notebook) because in Colab it always error**
- `detect_food.tflite` => This is model that we used and included metadata inside
- `requirements.txt` => This is file that contain all dependencies that we used to run this project

## Getting Started With Our Project
To using our project you must clone this machine learning project at first, this below the link:
`git clone https://github.com/abriyanyusuf/C23PS423_Food-Object-Detection.git`
after you clone you can use our machine learning project. Then you can use our main machine learning file
`TrainObjDetect.ipynb`. Inside the file we already provide instructions from A to Z. 

## Model that has been trained 40.000 epoch
You can download our model contain `.tflite`, label, and savedmodel from this link below. Upload it to your google drive.
<br>
[Model Trained 40.000 Epochs](https://drive.google.com/file/d/1F9Yf3i3FddNVICqc9d9YKqo45J_Og3Gg/view?usp=sharing)

## Data that has been split 
Here is the dataset that we have processed splitting it into 3 folders (train, val, and test). You can download it and upload to your google drive.
<br>
[Splitted Data](https://drive.google.com/file/d/1LxlvVIVaiZIs-EH9QcqZCgKOnbLF4el9/view?usp=sharing)

## Dataset and Additional Dataset
Dataset : <br>
[ECUST Food Dataset](https://github.com/Liang-yc/ECUSTFD-resized-) <br>
[Calories in Food Items (per 100 grams)](https://www.kaggle.com/datasets/kkhandekar/calories-in-food-items-per-100-grams)
<br>
We Also modified Calories in Food Items (per 100 grams) dataset by adding marketplace link for each item. You can access the data from link below.
<br>
Additional dataset Google Drive : [Calories in Food Items per 100 grams Modified Dataset](https://docs.google.com/spreadsheets/d/1NpLukLfHSIKOz2mk2QtLYaef0Ka5PaM5/edit?usp=sharing&ouid=109136081789719236546&rtpof=true&sd=true)


# YOLOv3 Object Detector PyTorch Implementation

This repository is built upon [Ayoosh Kathuria's](https://github.com/ayooshkathuria) work.

## Requirements

* Python 3.6+
* PyTorch 0.4.0
* OpenCV 3.4.1+

## Prerequistes

* Place all of the images in the images folder

## Convert bbox-annotations to txt format

We can covert the given bbox-annotations.json format file to TXT format which will have annotations for all the GT images to do that we can run the 
```sh
YOLO_JSON IPython notebook
```
This will take all the images from the image directory and then create a TXT file with bbox coordinates for each image in the gt_annotations folder. The format of the GT TXT file will be - class_name left top right bottom

## Download YOLOv3 Weights

Download the weights using the following command
```sh
wget -O yolov3.weights https://pjreddie.com/media/files/yolov3.weights
```
## Running Detection on Images

Run the folllowing script to start the detector
```sh
python3 annotate_frames --path images
```

Hold on, there are some command line arguments you can utilize
* --confidence - To set the confidence level
* --nms_thresh - To set the NMS Threshold
* --reso - To set the input resolution
* --source - To set the input camera source
* --skip - To skip every alternate frame or not, for faster processing speed
* --path - path to the images folder

This will take all the images from the image directory and then run YOLO v3 algorithm on it. The resultant bbox coordinate for each image will be stored in the TXT file in images directory. The format in the TXT file will be - class_name confidence left top right bottom

## Calculating the mAP score

* We will take all of the GT annotations in TXT format from the gt_annotations folder and place it inside the mAP\input\ground-truth directory.
* Next, we will take the detection results in TXT format from the images folder and place it inside the mAP\input\detection-results directory.
* Finally we will run the main.py file inside the mAP directory.
 ```sh
python3 main.py
```
This will output the mAP scores for the two classes and also save the result in the mAP\output folder.
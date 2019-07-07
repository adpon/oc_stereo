# Object-Centric Stereo Matching for 3D Object Detection

This repository contains code for Object-Centric Stereo Matching for 3D Object Detection.

## Getting Started

Implemented and tested on Ubuntu 16.04 with Python 2.7

Install Python dependencies
```
cd oc_stereo
pip install -r requirements.txt
```
Add to your PYTHONPATH
```
# For virtualenvwrapper users
add2virtualenv .
add2virtualenv ./src
```
Compile ROI align
```
cd src/faster_rcnn_pytorch/lib/pycocotools
python setup.py build_ext --inplace
cd ..
python setup.py build develop
```

## Training
For KITTI Object Detection dataset training
* Download the dataset and place it at `~/Kitti/object`
* Download the custom datasplits and place them into `~/Kitti/object`
The folder structure should be like:
```
Kitti
    object
        testing
        training
            calib
            image_2
            label_2
            planes
            velodyne
        train.txt
        val.txt
```
* Download our pretrained model here and place it in `data/pretrained`
* Get the MS-CNN detections here and place them in `data/mscnn` or run MS-CNN and use our box 
association algorithm
* Download the SRGT instance masks and our instance masks here
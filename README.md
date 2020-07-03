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
* Download the [KITTI Object Detection dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and place it in your home directory
* Place the attached custom data splits and place them into `~/Kitti/object`
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
* NOTE: check the options in each Python script before running
* Due to space constraints, a pre-trained model is not attached. However, you can try using 
a pretrained model from [PSMNet](https://github.com/JiaRenChang/PSMNet) and place the model in 
`data/pretrained`
* Run [MS-CNN](https://github.com/zhaoweicai/mscnn) or another 2D detection and convert the detections to KITTI format 
and place them as `data/mscnn/kitti_fmt`.
* Download the SRGT instance masks [here](http://liangchiehchen.com/projects/beat_the_MTurkers.html) and place them as `~/Kitti/object/training/instance_2_srgt`
* Generate our instance masks using `python src/oc_stereo/utils/gen_instance_masks.py` and place the outputs as `~/Kitti/object/training/instance_2_depth_2_multiscale`
* Obtain the instance masks that match to the MS-CNN detections using 
`python src/oc_stereo/utils/save_match_instance_mask.py`

### Generate Ground truth
* Generate depth completed LiDAR maps `python src/oc_stereo/utils/save_lidar_depth_maps.py`
* Place them in your Kitti folder
* Convert these depth maps to disparity maps `python src/save_disp_from_depth.py`

### Start Training and Inference
* Run training using `python src/oc_stereo/experiments/train.py`
* Produce disparity maps using `python src/oc_stereo/experiments/inference.py`

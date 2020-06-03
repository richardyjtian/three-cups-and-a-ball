# threecupsoneball

## To run YOLO stuff (adapted from <a href="https://www.learnopencv.com/training-yolov3-deep-learning-based-custom-object-detector/):" target="_blank">this tutorial</a>):

### 0. Split Train and Test Images (Optional)
`cd YOLO`  
`python3 splitTrainAndTest.py`

### 1. Install awscli
`sudo pip3 install awscli` 

### 2. Compile darknet
`cd ../darknet`  
`make`

### 3. Run darknet
`./darknet detector train ../YOLO/darknet.data ../YOLO/darknet-yolov3.cfg ./darknet53.conv.74`

## Debugging
If the process is killed:  
Decrease batch number or increase subdivisions in YOLO/darknet-yolov3.cfg by factors of 2

If you want to use GPU instead of CPU:
Set GPU and CUDNN to 1 in darknet/Makefile  
(I think you need a NVIDIA GPU for this, but it will make it way faster)

# threecupsoneball

To run YOLO stuff:

0. Split Train and Test Images (Optional)
cd YOLO
python3 splitTrainAndTest.py

1. Make darknet
cd ../darknet
make

2. Run darknet
./darknet detector train ../YOLO/darknet.data ../YOLO/darknet-yolov3.cfg ./darknet53.conv.74

###### Debugging #####
If the process is killed:
Decrease batch number or increase subdivisions in YOLO/darknet-yolov3.cfg by factors of 2 or increase

If you want to use GPU instead of CPU:
Set GPU and CUDNN to 1 in darknet/Makefile
(I think you need a NVIDIA GPU for this, but it will make it way faster)

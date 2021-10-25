#!/bin/bash

python yolo_counter.py --img_size=1920 --files "C:\Users\Klaudia\Desktop\project\object_counter\playlist_20210406T070000+0200.mp4" --detector_path yolov5l.pt --track_points=centroid --classes 0 1 2
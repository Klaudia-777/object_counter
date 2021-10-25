#!/bin/bash

python yolo_counter.py --img_size=1920 --files http://player.webcamera.pl/krakowsan_cam_480f1a?fbclid=IwAR0To2bIIT1ekH6ZrMEZhW-TahQr9wJeIIjXquaWBxuhuCnYuR1kcbbqz3I --detector_path yolov5l.pt --track_points=centroid --classes 0 1 2 80
import argparse
import csv
from typing import Union, List, Optional

import cv2
import norfair
import numpy as np
import torch
import yolov5
from norfair import Detection, Tracker, Video
from yolov5.utils.plots import Colors

max_distance_between_points: int = 40
colors = Colors()  # create instance for 'from utils.plots import colors'


def convert_id_to_class(id_number: int):
    color_dict = {
        0: "person",
        1: "bicycle",
        2: "car",
        80: "cargovelo"
    }
    return color_dict.get(id_number)


class TrackedObject:
    distance_threshold = 30

    def __init__(self, id, clazz, points, frame):
        self.id = id
        self.clazz = clazz
        self.points = points
        self.first_frame = frame
        self.current_frame = frame
        self.idle = 0
        self.max_idle = 0
        self.previous_idle = 0

    def step(self, frame, new_points):
        if self.clazz == 2:
            if computeDistance(self.points, new_points) <= max_distance_between_points:
                self.idle += 1
            else:
                if (self.previous_idle > 100):
                    tmp = self.previous_idle
                    self.previous_idle = self.idle
                    self.idle += tmp
                    self.max_idle = max(self.max_idle, self.idle)
                else:
                    self.max_idle = max(self.max_idle, self.idle)
                    self.previous_idle = self.idle
                    self.idle = 0
        self.points = new_points
        self.current_frame = frame


class New_tracked_object:
    def __init__(self, identity, i):
        self.identity = identity
        self.id = self.convert_id_to_class(id)


class YOLO:
    def __init__(self, model_path: str, device: Optional[str] = None):
        if device is not None and "cuda" in device and not torch.cuda.is_available():
            raise Exception(
                "Selected device='cuda', but cuda is not available to Pytorch."
            )
        # automatically set device if its None
        elif device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # load model
        self.model = yolov5.load(model_path, device=device)
        self.names = self.model.names

    def __call__(
            self,
            img: Union[str, np.ndarray],
            conf_threshold: float = 0.25,
            iou_threshold: float = 0.45,
            image_size: int = 720,
            classes: Optional[List[int]] = None
    ) -> torch.tensor:

        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        if classes is not None:
            self.model.classes = classes
        detections = self.model(img, size=image_size)
        return detections


def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)


def yolo_detections_to_norfair_detections(
        yolo_detections: torch.tensor,
        track_points: str = 'centroid'  # bbox or centroid
) -> List[Detection]:
    """convert detections_as_xywh to norfair detections
    """
    norfair_detections: List[Detection] = []

    if track_points == 'centroid':
        detections_as_xywh = yolo_detections.xywh[0]
        for detection_as_xywh in detections_as_xywh:
            centroid = np.array(
                [
                    detection_as_xywh[0].item(),
                    detection_as_xywh[1].item()
                ]
            )
            scores = np.array([detection_as_xywh[4].item()])
            norfair_detections.append(
                Detection(points=centroid, scores=scores, data=int(detection_as_xywh[5].item()))
            )
    elif track_points == 'bbox':
        detections_as_xyxy = yolo_detections.xyxy[0]
        for detection_as_xyxy in detections_as_xyxy:
            bbox = np.array(
                [
                    [detection_as_xyxy[0].item(), detection_as_xyxy[1].item()],
                    [detection_as_xyxy[2].item(), detection_as_xyxy[3].item()]
                ]
            )
            scores = np.array([detection_as_xyxy[4].item(), detection_as_xyxy[4].item()])
            norfair_detections.append(
                Detection(points=bbox, scores=scores, data=int(detection_as_xyxy[5].item()))
            )
    # detections_as_pred = yolo_detections.pred[0]
    # for detection_as_pred in detections_as_pred:
    #     # for *xyxy, conf, cls in reversed(detection_as_pred):
    #     print(int(detection_as_pred[-1]))
    return norfair_detections


parser = argparse.ArgumentParser(description="Track objects in a video.")
parser.add_argument("--files", type=str, nargs="+", help="Video files to process")
parser.add_argument("--detector_path", type=str, default="yolov5m6.pt", help="YOLOv5 model path")
parser.add_argument("--img_size", type=int, default="720", help="YOLOv5 inference size (pixels)")
parser.add_argument("--conf_thres", type=float, default="0.25", help="YOLOv5 object confidence threshold")
parser.add_argument("--iou_thresh", type=float, default="0.45", help="YOLOv5 IOU threshold for NMS")
parser.add_argument('--classes', nargs='+', type=int, help='Filter by class: --classes 0, or --classes 0 2 3')
parser.add_argument("--device", type=str, default=None, help="Inference device: 'cpu' or 'cuda'")
parser.add_argument("--track_points", type=str, default="centroid", help="Track points: 'centroid' or 'bbox'")
args = parser.parse_args()

model = YOLO(args.detector_path, device=args.device)
detected_objects = {}
detected_classes = set()


def computeCenterPoint(points):
    x = points[0][0]


def computeDistance(points1, points2):
    return np.linalg.norm(points1 - points2)
    # return pow(pow(points1[0][0] - points2[1][0], 2) + pow(points1[0][1] - points2[1][1], 2), -2)


def findClosestDetectedObject(trackedObject, detected_objects, tracked_objects):
    minDistance = 11111111
    closest_object = None
    trackedIds = map(lambda x: x.id, tracked_objects)
    for key in detected_objects:
        detected_object = detected_objects[key]
        distance = computeDistance(detected_object.points, trackedObject.last_detection.points)
        if distance < minDistance and trackedObject.last_detection.data == detected_object.clazz and detected_object.id not in trackedIds:
            closest_object = detected_object
            minDistance = distance
    return (minDistance, closest_object)


def addTrackedObject(frame, tracked_object):
    if tracked_object.id in detected_objects:
        detected_objects[tracked_object.id].step(frame, tracked_object.last_detection.points)
    else:
        detected_objects[tracked_object.id] = TrackedObject(tracked_object.id, tracked_object.last_detection.data,
                                                            tracked_object.last_detection.points, frame)


def replaceClosestTrackedObject(frame, closest_tracked_object, tracked_object):
    if closest_tracked_object.id in detected_objects:
        detected_objects.pop(closest_tracked_object.id)
        closest_tracked_object.id = tracked_object.id
        closest_tracked_object.step(frame, tracked_object.last_detection.points)
        detected_objects[closest_tracked_object.id] = closest_tracked_object
    else:
        detected_objects[tracked_object.id] = TrackedObject(tracked_object.id, tracked_object.last_detection.data,
                                                            tracked_object.last_detection.points, frame)


def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


for input_path in args.files:
    video = Video(input_path=input_path)
    tracker = Tracker(
        distance_function=euclidean_distance,
        distance_threshold=max_distance_between_points,
    )

    i = 0
    for frame in video:
        if i % 5 == 0:
            yolo_detections = model(
                frame,
                conf_threshold=args.conf_thres,
                iou_threshold=args.conf_thres,
                image_size=args.img_size,
                classes=args.classes
            )

            detections = yolo_detections_to_norfair_detections(yolo_detections, track_points=args.track_points)
            tracked_objects = tracker.update(detections=detections)
            for d in detections:
                detected_classes.add(d.data)
            # if args.track_points == 'centroid':
            #     # norfair.draw_points(frame, detections)
            # elif args.track_points == 'bbox':
            #     # norfair.draw_boxes(frame, detections, line_width=3)
            if len(tracked_objects) > 0:
                detectedIds = map(lambda x: x.id, detected_objects)
                for tracked_object in tracked_objects:
                    if tracked_object.id not in detected_objects:
                        distance, closest_object = findClosestDetectedObject(tracked_object, detected_objects,
                                                                             tracked_objects)
                        if closest_object is None:
                            addTrackedObject(i, tracked_object)

                        elif distance < 60 and closest_object.clazz == tracked_object.last_detection.data:
                            replaceClosestTrackedObject(i, closest_object, tracked_object)
                        else:
                            addTrackedObject(i, tracked_object)
                    else:
                        addTrackedObject(i, tracked_object)

            trackedIds = map(lambda x: x.id, tracked_objects)
            # for key in detected_objects:
            #  detected_object = detected_objects[key]
            # if detected_object.id not in trackedIds:
            #    detected_object.step(detected_object.current_frame, detected_object.points)
            new_tracked_objects = []
            # for d in detections:
            #     points = [d.points[0][0], d.points[0][1], d.points[1][0], d.points[1][1]]
            #     plot_one_box(points, frame, label=convert_id_to_class(d.data), color=colors(d.data, True), line_thickness=3)

            # for to in tracked_objects:
            #         new_tracked_objects.append(New_tracked_object(identity=to.id, id=to.last_detection.data))
            # norfair.draw_tracked_objects(frame, new_tracked_objects)
            video.write(frame)
        i += 1
counters = {0: 0, 1: 0, 2: 0}
size = len(str(args.files))
csv_file = open(str(args.files).split('.')[0][2:] + '.csv', 'w')
writer = csv.writer(csv_file)
writer.writerow(["Object ID", "Class", "Object First Frame", "Object Last Frame", "Object Max Idle"])

for key in detected_objects:
    detected_object = detected_objects[key]
    writer.writerow(
        [detected_object.id, detected_object.clazz, detected_object.first_frame, detected_object.current_frame,
         detected_object.max_idle])
    counters[detected_object.clazz] = counters[detected_object.clazz] + 1

summary_file = open(str(args.files).split('.')[0][2:] + '_summary.txt', 'w')
summary_writer = csv.writer(summary_file)
for cls in detected_classes:
    print("detected: " + str(counters[cls]) + " " + model.names[cls])
    summary_writer.writerow([str(model.names[cls]), str(counters[cls])])

summary_file.close()
csv_file.close()

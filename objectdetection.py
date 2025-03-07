##when a movement is detected in camera it checks for the gps location of our men if that matches with
##the gps of camera that means the movement detected by camera is non other than our men
##but if gps locations dosent match then there's a high chance that movement is actually an intrusion and raises alert
##for demo we have taken static gps of camera and men[soldier] when connected to external device the code requires few changes to
##make it real time gps tracking

import cv2
import torch
import numpy as np
import pandas as pd
import requests
from geopy.distance import geodesic

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
camera_gps = {"camera_1": (28.6139, 77.2090)}
soldiers_gps = {"soldier_1": (28.6140, 77.2092), "soldier_2": (28.6129, 77.2085)}

distance_threshold = 50

def detect_objects(frame):
    results = model(frame)
    objects_detected = results.pandas().xyxy[0]
    return objects_detected

def check_suspicious_activity(objects_detected):
    suspicious_classes = ['person', 'dog', 'cat', 'firearm']
    detected_classes = objects_detected['name'].tolist()
    for cls in detected_classes:
        if cls in suspicious_classes:
            return True
    return False

def find_nearest_soldiers(camera_location, soldiers_locations, n=3):
    distances = {}
    for soldier, location in soldiers_locations.items():
        distances[soldier] = geodesic(camera_location, location).meters
    nearest_soldiers = sorted(distances.items(), key=lambda x: x[1])[:n]
    return nearest_soldiers


def process_camera_feed(camera_id, camera_location):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Camera {camera_id}: Unable to read feed")
            break

        results = model(frame)
        objects_detected = results.pandas().xyxy[0]

        if check_suspicious_activity(objects_detected):
            print(f"Suspicious activity detected by {camera_id}")

            nearby_soldiers = find_nearest_soldiers(camera_location, soldiers_gps)
            if nearby_soldiers:
                print(f"Nearby soldiers to {camera_id}: {nearby_soldiers}")

                for soldier, distance in nearby_soldiers:
                    print(f"Alert sent to {soldier} ({distance:.2f} meters away)")

            else:
                print(f"High Alert: No soldiers near {camera_id}")

        results.render()
        annotated_frame = np.squeeze(results.render())
        cv2.imshow(f"Camera {camera_id}", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

for camera_id, gps_location in camera_gps.items():
    print(f"Processing {camera_id}...")
    process_camera_feed(camera_id, gps_location)

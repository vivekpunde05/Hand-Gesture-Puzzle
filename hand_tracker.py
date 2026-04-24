import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import urllib.request
import os

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")

# Hand connections for drawing
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

class HandTracker:
    def __init__(self):
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.detection_result = None

    def find_hands(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.detection_result = self.detector.detect(mp_image)

    def draw_hands(self, frame):
        if not self.detection_result or not self.detection_result.hand_landmarks:
            return
        h, w, _ = frame.shape
        for hand in self.detection_result.hand_landmarks:
            # draw connections
            for a, b in HAND_CONNECTIONS:
                x1, y1 = int(hand[a].x * w), int(hand[a].y * h)
                x2, y2 = int(hand[b].x * w), int(hand[b].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # draw landmarks
            for lm in hand:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)

    def get_pinch(self):
        if self.detection_result and self.detection_result.hand_landmarks:
            for hand in self.detection_result.hand_landmarks:
                x1, y1 = hand[4].x, hand[4].y   # thumb tip
                x2, y2 = hand[8].x, hand[8].y   # index tip
                dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                if dist < 0.05:
                    return True, x2, y2
            hand = self.detection_result.hand_landmarks[0]
            return False, hand[8].x, hand[8].y
        return False, 0, 0

    def get_index_pos(self):
        if self.detection_result and self.detection_result.hand_landmarks:
            hand = self.detection_result.hand_landmarks[0]
            return True, hand[8].x, hand[8].y
        return False, 0, 0

    def get_two_hand_indices(self):
        points = []
        if self.detection_result and self.detection_result.hand_landmarks:
            for hand in self.detection_result.hand_landmarks:
                points.append((hand[8].x, hand[8].y))
        if len(points) >= 2:
            return True, points[0], points[1]
        return False, (0, 0), (0, 0)

    def get_two_hand_positions(self):
        points = []
        if self.detection_result and self.detection_result.hand_landmarks:
            for hand in self.detection_result.hand_landmarks:
                points.append((hand[8].x, hand[8].y))
        return points

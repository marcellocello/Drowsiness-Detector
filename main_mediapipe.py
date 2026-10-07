import cv2
import numpy as np
from scipy.spatial import distance
import mediapipe as mp
import torch
import time

THRES_TIDUR = 2.0
THRES_NGANTUK = 0.8
THRES_EAR = 0.23

LEFT_EYE_COORDINATE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_COORDINATE = [362, 385, 387, 263, 373, 380]

def select_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"[INFO] Running on device: {device.upper()}")
    return device

def calculate_ear(eye_points):
    point_a = distance.euclidean(eye_points[1], eye_points[5])
    point_b = distance.euclidean(eye_points[2], eye_points[4])
    point_c = distance.euclidean(eye_points[0], eye_points[3])
    if point_c == 0:
        return 0.0
    return (point_a + point_b) / (2.0 * point_c)

def draw_eye_contours(frame, eye_points, color):
    pts = np.array(eye_points, dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

def draw_face_box(frame, landmarks, w, h, color, pad=10):
    xs = [p.x * w for p in landmarks]
    ys = [p.y * h for p in landmarks]
    x1 = max(int(min(xs)) - pad, 0)
    y1 = max(int(min(ys)) - pad, 0)
    x2 = min(int(max(xs)) + pad, w - 1)
    y2 = min(int(max(ys)) + pad, h - 1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    return x1, y1

def eye_pts(landmarks, idx, w, h):
    return [(landmarks[i].x * w, landmarks[i].y * h) for i in idx]

def run_drowsiness_detector(camera_index=0):
    device = select_device()
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(camera_index)
    closed_start_times = {}

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        now = time.monotonic()
        faces = result.multi_face_landmarks or []
        active_face_ids = set()

        for idx, face in enumerate(faces):
            active_face_ids.add(idx)

            landmarks = face.landmark
            left = eye_pts(landmarks, LEFT_EYE_COORDINATE, w, h)
            right = eye_pts(landmarks, RIGHT_EYE_COORDINATE, w, h)
            ear = (calculate_ear(left) + calculate_ear(right)) / 2.0

            if ear < THRES_EAR:
                closed_start_times.setdefault(idx, now)
                dur = now - closed_start_times[idx]
            else:
                closed_start_times.pop(idx, None)
                dur = 0.0

            if dur >= THRES_TIDUR:
                status, color = "TIDUR", (0, 0, 255)
            elif dur >= THRES_NGANTUK:
                status, color = "NGANTUK", (0, 165, 255)
            else:
                status, color = "MELEK", (0, 255, 0)

            draw_eye_contours(frame, left, color)
            draw_eye_contours(frame, right, color)

            x1, y1 = draw_face_box(frame, landmarks, w, h, color)
            cv2.putText(frame, f"Wajah #{idx+1}: {status} - EAR:{ear:.2f}", (x1, max(10, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for old_id in list(closed_start_times):
            if old_id not in active_face_ids:
                closed_start_times.pop(old_id)

        cv2.imshow("Drowsiness Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_drowsiness_detector(camera_index=0)
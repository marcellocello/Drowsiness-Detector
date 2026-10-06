import urllib
import os
import cv2
import dlib
from scipy.spatial import distance
import torch
import time

PREDICTOR_MODEL = "shape_predictor_68_face_landmarks.dat"
THRES_TIDUR = 2
THRES_NGANTUK = 0.8
THRES_EAR = 0.23

def ensure_predictor_exists():
    if os.path.exists(PREDICTOR_MODEL) and os.path.getsize(PREDICTOR_MODEL) < 90 * 1024 * 1024:
        os.remove(PREDICTOR_MODEL)
    if not os.path.exists(PREDICTOR_MODEL):
        print("[INFO] File predictor 68 landmark belum ada. Mengunduh model (~99MB)...")

        url = f"https://raw.githubusercontent.com/tzutalin/dlib-android/master/data/{PREDICTOR_MODEL}"

        print(f"[INFO] Downloading {PREDICTOR_MODEL} ...")

        urllib.request.urlretrieve(url, PREDICTOR_MODEL)
        print("[INFO] Download berhasil & file utuh!")

def select_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"[INFO] Running on device: {device.upper()}")
    return device

def detect_eye(eye):
    point_a = distance.euclidean(eye[1], eye[5])
    point_b = distance.euclidean(eye[2], eye[4])
    point_c = distance.euclidean(eye[0], eye[3])
    aspec_ratio_eye = (point_a + point_b) / (2 * point_c)
    return aspec_ratio_eye

def draw_eye_contours(frame, face_landmarks, eye_indices, color):
    eye_points = []
    num_points = len(eye_indices)

    for i in range(num_points):
        n = eye_indices[i]
        x = face_landmarks.part(n).x
        y = face_landmarks.part(n).y
        eye_points.append((x, y))

        next_n = eye_indices[(i + 1) % num_points]
        x2 = face_landmarks.part(next_n).x
        y2 = face_landmarks.part(next_n).y

        cv2.line(frame, (x, y), (x2, y2), color, 2)
    
    return eye_points

def run_drowsiness_detector(camera_index=0):
    ensure_predictor_exists()

    device = select_device()
    cap = cv2.VideoCapture(camera_index)
    face_detector = dlib.get_frontal_face_detector()
    dlib_facelandmark = dlib.shape_predictor(PREDICTOR_MODEL)

    line_color = (0, 255, 0)
    status_state = "MELEK"
    closed_start_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray_scale)
        face_found = False

        for face in faces:
            face_found = True
            face_landmarks = dlib_facelandmark(gray_scale, face)
            lefteye = [(face_landmarks.part(n).x, face_landmarks.part(n).y) for n in range(36, 42)]
            righteye = [(face_landmarks.part(n).x, face_landmarks.part(n).y) for n in range(42, 48)]

            leftear = detect_eye(lefteye)
            rightear = detect_eye(righteye)
            eye_ratio = round((leftear + rightear) / 2, 2)

            is_curr_closed = eye_ratio < THRES_EAR
            if not is_curr_closed:
                closed_start_time = None
                line_color = (0, 255, 0)
                status_state = "MELEK"
            else:
                if closed_start_time is None:
                    closed_start_time = time.time()
                
                closed_duration = time.time() - closed_start_time

                if closed_duration >= THRES_TIDUR:
                    status_state = "TERDETEKSI TIDUR"
                    line_color = (0, 0, 255)
                    cv2.putText(frame, "ALERT!! BANGUNNN!!!!", (40, 120), cv2.FONT_HERSHEY_PLAIN, 2, line_color, 3)
                elif closed_duration >= THRES_NGANTUK:
                    line_color = (0, 165, 255)
                    status_state = "TERDETEKSI NGANTUK"
                    cv2.putText(frame, "ALERT!! BANGUNNN!!!!", (40, 120), cv2.FONT_HERSHEY_PLAIN, 2, line_color, 3)
                else:
                    status_state = "MELEK"
                    line_color = (0, 255, 0)

            cv2.putText(frame, status_state, (40, 80), cv2.FONT_HERSHEY_PLAIN, 2, line_color, 3)

            draw_eye_contours(frame, face_landmarks, range(42, 48), line_color)
            draw_eye_contours(frame, face_landmarks, range(36, 42), line_color)

        if not face_found:
            closed_start_time = None

        cv2.imshow("Drowsiness Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_drowsiness_detector(camera_index=0)
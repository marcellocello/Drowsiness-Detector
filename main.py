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

    closed_start_time = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray_scale)
        active_face_ids = set()

        for idx, face in enumerate(faces):
            active_face_ids.add(idx)

            face_landmarks = dlib_facelandmark(gray_scale, face)
            lefteye = [(face_landmarks.part(n).x, face_landmarks.part(n).y) for n in range(36, 42)]
            righteye = [(face_landmarks.part(n).x, face_landmarks.part(n).y) for n in range(42, 48)]

            leftear = detect_eye(lefteye)
            rightear = detect_eye(righteye)
            eye_ratio = round((leftear + rightear) / 2, 2)

            is_curr_closed = eye_ratio < THRES_EAR
            if not is_curr_closed:
                closed_start_time[idx] = None
                line_color = (0, 255, 0)
                eye_state = "MELEK"
            else:
                if closed_start_time.get(idx) is None:
                    closed_start_time[idx] = time.time()
                
                closed_duration = time.time() - closed_start_time[idx]

                if closed_duration >= THRES_TIDUR:
                    eye_state = "TIDUR"
                    line_color = (0, 0, 255)
                elif closed_duration >= THRES_NGANTUK:
                    line_color = (0, 165, 255)
                    eye_state = "NGANTUK"
                else:
                    eye_state = "MELEK"
                    line_color = (0, 255, 0)

            x, y = face.left(), face.top()
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), line_color, 2)
            cv2.putText(frame, f"Wajah #{idx+1}: {eye_state}", (x, max(10, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_color, 2)

            if eye_state in ["NGANTUK", "TIDUR"]:
                cv2.putText(frame, "ALERT!! BANGUNNN!!!!", (x, face.bottom() + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_color, 2)

            draw_eye_contours(frame, face_landmarks, range(42, 48), line_color)
            draw_eye_contours(frame, face_landmarks, range(36, 42), line_color)

        for old_id in list(closed_start_time.keys()):
            if old_id not in active_face_ids:
                del closed_start_time[old_id]

        cv2.imshow("Drowsiness Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_drowsiness_detector(camera_index=0)
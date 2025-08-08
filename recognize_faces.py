import os
import cv2
import dlib
import time
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import argparse

from log_utils import log_attendance  # pakai punyamu

# =========================
# Config
# =========================
DESCRIPTOR_CSV = 'data/descriptors.csv'
UNKNOWN_SAVE_DIR = 'data/unknown_faces'
MODEL_PREDICTOR = 'data/data_dlib/shape_predictor_68_face_landmarks.dat'
MODEL_RECOG = 'data/data_dlib/dlib_face_recognition_resnet_model_v1.dat'

THRESHOLD = 0.6
MAX_ATTEMPTS_PER_PERSON = 10

os.makedirs(UNKNOWN_SAVE_DIR, exist_ok=True)

# =========================
# Load dlib models (identik websocket.py)
# =========================
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(MODEL_PREDICTOR)
face_reco_model = dlib.face_recognition_model_v1(MODEL_RECOG)

def load_known_descriptors(path=DESCRIPTOR_CSV):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} tidak ditemukan. Jalankan register terlebih dahulu.")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("descriptors.csv kosong. Jalankan register terlebih dahulu.")

    names = df['name'].tolist()
    ids = df['id'].tolist()
    descs = []
    for s in df['descriptor']:
        vec = np.array([float(x) for x in s.split(',')], dtype=np.float64)  # mirror websocket (float64)
        descs.append(vec)
    return ids, names, np.stack(descs, axis=0)

def compute_descriptors_bgr(frame_bgr):
    """
    Deteksi semua wajah → compute_face_descriptor (tanpa RGB/align), mirror websocket.
    """
    faces = detector(frame_bgr, 1)
    encs, rects = [], []
    for face_rect in faces:
        shape = predictor(frame_bgr, face_rect)
        desc = face_reco_model.compute_face_descriptor(frame_bgr, shape)
        encs.append(np.array(desc, dtype=np.float64))
        rects.append(face_rect)
    return encs, rects

def recognize(cam_index=0, expected='known', max_attempts=MAX_ATTEMPTS_PER_PERSON, threshold=THRESHOLD):
    ids, names, known_descs = load_known_descriptors()

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("❌ Tidak bisa membuka kamera.")

    print("🔍 Mulai pengenalan wajah (dlib, mirror websocket). Tekan 'q' untuk berhenti.")
    recognition_counter = defaultdict(int)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        t0 = time.time()
        encs, rects = compute_descriptors_bgr(frame)
        t1 = time.time()
        runtime_ms = (t1 - t0) * 1000.0

        for enc, rect in zip(encs, rects):
            # Euclidean distance (aturanmu)
            dists = np.linalg.norm(known_descs - enc, axis=1)
            min_idx = int(np.argmin(dists))
            min_dist = float(dists[min_idx])

            face_hash = hashlib.sha256(enc.tobytes()).hexdigest()

            if min_dist < threshold:
                name = names[min_idx]
                user_id = ids[min_idx]
                if recognition_counter[name] >= max_attempts:
                    continue
                recognition_counter[name] += 1

                label = f"{name} ({min_dist:.3f})"
                color = (0, 255, 0)
                result = 'success'
            else:
                name = "Unknown"
                user_id = "unknown"
                label = f"Unknown ({min_dist:.3f})"
                color = (0, 0, 255)
                result = 'fail'

                # Simpan frame unknown untuk audit
                filename = f"unknown_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                cv2.imwrite(os.path.join(UNKNOWN_SAVE_DIR, filename), frame)

            # Logging ke CSV via log_utils
            try:
                log_attendance(user_id, name, face_hash, runtime_ms, result, expected)
            except Exception as e:
                print(f"⚠️ Gagal logging: {e}")

            # Draw
            cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), color, 2)
            cv2.putText(frame, label, (rect.left(), rect.top()-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Face Recognition (dlib)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n📌 Jumlah pengujian per wajah:")
    for name, count in recognition_counter.items():
        print(f"{name}: {count} kali")

def main():
    parser = argparse.ArgumentParser(description="Realtime face recognition (dlib, mirror websocket) + logging.")
    parser.add_argument("--cam", type=int, default=0, help="Index kamera (default: 0).")
    parser.add_argument("--expected", type=str, default="known", choices=["known", "unknown"],
                        help="Label ground-truth untuk sesi ini (default: known).")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help=f"Ambang Euclidean (default: {THRESHOLD}).")
    parser.add_argument("--max_attempts", type=int, default=MAX_ATTEMPTS_PER_PERSON,
                        help=f"Batas log per orang (default: {MAX_ATTEMPTS_PER_PERSON}).")
    args = parser.parse_args()

    recognize(cam_index=args.cam, expected=args.expected,
              max_attempts=args.max_attempts, threshold=args.threshold)

if __name__ == "__main__":
    main()

import cv2
import face_recognition
import pandas as pd
import numpy as np
import os
import time
import hashlib
from datetime import datetime
from collections import defaultdict
from log_utils import log_attendance

DESCRIPTOR_CSV = 'data/descriptors.csv'
UNKNOWN_SAVE_DIR = 'data/unknown_faces'
THRESHOLD = 0.6
MAX_ATTEMPTS_PER_PERSON = 10

os.makedirs(UNKNOWN_SAVE_DIR, exist_ok=True)

def load_known_descriptors(path=DESCRIPTOR_CSV):
    df = pd.read_csv(path)
    names = df['name'].tolist()
    descriptors = [np.array([float(x) for x in row.split(',')]) for row in df['descriptor']]
    return names, descriptors

def recognize():
    names, descriptors = load_known_descriptors()
    cap = cv2.VideoCapture(0)
    print("🔍 Mulai pengenalan wajah...")

    recognition_counter = defaultdict(int)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb)
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        for (top, right, bottom, left), face_enc in zip(face_locations, face_encodings):
            start = time.time()
            distances = [np.linalg.norm(face_enc - desc) for desc in descriptors]
            min_dist = min(distances) if distances else float('inf')
            end = time.time()

            runtime_ms = (end - start) * 1000
            face_hash = hashlib.sha256(face_enc.tobytes()).hexdigest()
            expected = 'known'  # kamu bisa ubah ke 'unknown' saat uji wajah asing

            if min_dist < THRESHOLD:
                idx = distances.index(min_dist)
                name = names[idx]
                user_id = f"user{idx+1}"

                if recognition_counter[name] >= MAX_ATTEMPTS_PER_PERSON:
                    continue

                recognition_counter[name] += 1
                label = f"{name}"
                color = (0, 255, 0)
                result = 'success'
            else:
                name = "Unknown"
                user_id = "unknown"
                label = "Unknown"
                color = (0, 0, 255)
                result = 'fail'

                filename = f"unknown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                path = os.path.join(UNKNOWN_SAVE_DIR, filename)
                cv2.imwrite(path, frame)

            # Logging
            log_attendance(user_id, name, face_hash, runtime_ms, result, expected)

            # Display
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n📌 Jumlah pengujian per wajah:")
    for name, count in recognition_counter.items():
        print(f"{name}: {count} kali")

if __name__ == "__main__":
    recognize()

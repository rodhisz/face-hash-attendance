import cv2
import face_recognition
import numpy as np
import pandas as pd
import os
from datetime import datetime

DESCRIPTOR_CSV = 'data/descriptors.csv'
IMAGE_SAVE_DIR = 'data/registered_faces'
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

def capture_images(name, user_id, num_images=50):
    cap = cv2.VideoCapture(0)
    count = 0
    descriptors = []

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb)
        if not face_locations:
            cv2.imshow("Register Wajah", frame)
            if cv2.waitKey(1) == ord('q'):
                break
            continue

        top, right, bottom, left = face_locations[0]
        face_enc = face_recognition.face_encodings(rgb, [face_locations[0]])[0]
        descriptors.append(face_enc)

        filename = f"{user_id}_{count+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(os.path.join(IMAGE_SAVE_DIR, filename), frame)
        print(f"✅ Wajah {count+1} berhasil direkam & disimpan.")

        count += 1
        cv2.imshow("Register Wajah", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return descriptors

def save_descriptor(user_id, name, avg_descriptor, csv_path=DESCRIPTOR_CSV):
    desc_str = ','.join(map(str, avg_descriptor))
    df = pd.DataFrame([[user_id, name, desc_str]], columns=['id', 'name', 'descriptor'])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)
    print(f"✅ Descriptor wajah untuk {name} disimpan ke {csv_path}.")

if __name__ == "__main__":
    name = input("Masukkan nama: ")
    user_id = input("Masukkan ID unik: ")

    descriptors = capture_images(name, user_id)
    if descriptors:
        avg_desc = np.mean(descriptors, axis=0)
        save_descriptor(user_id, name, avg_desc)
    else:
        print("❌ Tidak ada wajah yang berhasil direkam.")

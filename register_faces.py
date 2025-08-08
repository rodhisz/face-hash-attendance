import os
import cv2
import dlib
import numpy as np
import pandas as pd
from datetime import datetime
import argparse

# =========================
# Config paths
# =========================
DESCRIPTOR_CSV = 'data/descriptors.csv'
IMAGE_SAVE_DIR = 'data/registered_faces'
MODEL_PREDICTOR = 'data/data_dlib/shape_predictor_68_face_landmarks.dat'
MODEL_RECOG = 'data/data_dlib/dlib_face_recognition_resnet_model_v1.dat'

os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DESCRIPTOR_CSV), exist_ok=True)

# =========================
# Load dlib models (identik websocket.py)
# =========================
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(MODEL_PREDICTOR)
face_reco_model = dlib.face_recognition_model_v1(MODEL_RECOG)

def compute_descriptor_bgr(frame_bgr):
    """
    Hitung 128D descriptor dari frame BGR (tanpa RGB, tanpa alignment),
    sama seperti websocket.py (detector(img,1) → predictor → compute_face_descriptor).
    Ambil wajah pertama.
    """
    faces = detector(frame_bgr, 1)
    if len(faces) == 0:
        return None, None

    face_rect = faces[0]
    shape = predictor(frame_bgr, face_rect)
    desc = face_reco_model.compute_face_descriptor(frame_bgr, shape)  # dlib vector of float64
    return np.array(desc, dtype=np.float64), face_rect

def capture_images(name, user_id, num_images=50, cam_index=0):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("❌ Tidak bisa membuka kamera.")

    count = 0
    descriptors = []
    print("🎥 Register dimulai. Tekan 'q' untuk berhenti lebih awal.")

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            continue

        desc, rect = compute_descriptor_bgr(frame)

        # Tampilan
        if rect is not None:
            cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)
            cv2.putText(frame, f"Captured: {count}/{num_images}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("Register Wajah (dlib)", frame)

        if desc is None:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Simpan frame untuk audit
        filename = f"{user_id}_{count+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(os.path.join(IMAGE_SAVE_DIR, filename), frame)
        descriptors.append(desc)
        count += 1
        print(f"✅ Wajah {count} tersimpan: {filename}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return descriptors

def save_descriptor(user_id, name, avg_descriptor, csv_path=DESCRIPTOR_CSV):
    # Simpan sebagai string float “apa adanya” (preserve float64 precision)
    desc_str = ','.join(map(str, avg_descriptor.tolist()))
    row = pd.DataFrame([[user_id, name, desc_str]], columns=['id', 'name', 'descriptor'])
    if os.path.exists(csv_path):
        row.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        row.to_csv(csv_path, index=False)
    print(f"💾 Descriptor rata-rata untuk {name} disimpan ke {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Register wajah ke descriptors.csv (dlib, mirror websocket).")
    parser.add_argument("--name", type=str, help="Nama user (opsional, jika tidak diisi akan prompt).")
    parser.add_argument("--user_id", type=str, help="ID unik user (opsional, jika tidak diisi akan prompt).")
    parser.add_argument("--num", type=int, default=50, help="Jumlah frame untuk dirata-rata (default: 50).")
    parser.add_argument("--cam", type=int, default=0, help="Index kamera (default: 0).")
    args = parser.parse_args()

    name = args.name or input("Masukkan nama: ").strip()
    user_id = args.user_id or input("Masukkan ID unik: ").strip()

    descriptors = capture_images(name, user_id, num_images=args.num, cam_index=args.cam)
    if len(descriptors) == 0:
        print("❌ Tidak ada wajah yang berhasil direkam.")
        return

    avg_desc = np.mean(np.stack(descriptors, axis=0, dtype=np.float64), axis=0)
    save_descriptor(user_id, name, avg_desc)
    print("✅ Register selesai.")

if __name__ == "__main__":
    main()

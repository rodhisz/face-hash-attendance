import csv
import os
from datetime import datetime
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def log_attendance(user_id, name, face_hash, runtime_ms, result, expected='known', log_file='data/attendance_log.csv'):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    log_exists = os.path.exists(log_file)

    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not log_exists:
            writer.writerow(['timestamp', 'user_id', 'name', 'hash', 'runtime_ms', 'result', 'expected'])

        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            user_id,
            name,
            face_hash,
            runtime_ms,
            result,
            expected
        ])

def analyze_log(log_file='data/attendance_log.csv'):
    if not os.path.exists(log_file):
        print("❌ Log file belum tersedia.")
        return

    df = pd.read_csv(log_file)

    if 'expected' not in df.columns:
        print("⚠️ Kolom 'expected' belum tersedia untuk analisis klasifikasi penuh.")
        return

    y_true = df['expected']
    y_pred = df['result'].apply(lambda x: 'known' if x == 'success' else 'unknown')

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label='known', zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label='known', zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label='known', zero_division=0)
    avg_runtime = df['runtime_ms'].mean()

    print(f"\n📊 Hasil Evaluasi:")
    print(f"Total Absensi : {len(df)}")
    print(f"Akurasi       : {acc*100:.2f}%")
    print(f"Precision     : {prec*100:.2f}%")
    print(f"Recall        : {rec*100:.2f}%")
    print(f"F1-Score      : {f1*100:.2f}%")
    print(f"Rata2 Latency : {avg_runtime:.2f} ms")

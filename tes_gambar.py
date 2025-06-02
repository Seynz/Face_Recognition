import cv2
import numpy as np
from pymilvus import Collection, connections
from function import FaceRecognizer
import tensorflow as tf
import json

# Nonaktifkan log TensorFlow yang mengganggu
tf.get_logger().setLevel('ERROR')

# Koneksi ke Milvus
connections.connect(alias="default", host="localhost", port="19530")

# Ambil koleksi
collection = Collection("face_embeddings_mobilefacenet")

# Buat index jika belum ada
if not collection.indexes:
    collection.create_index(
        field_name="embedding",
        index_params={
            "index_type": "IVF_FLAT",     # Gunakan IVF_FLAT atau FLAT
            "metric_type": "IP",          # Sesuai dengan pembuatan koleksi
            "params": {"nlist": 128}
        }
    )
    print("Index berhasil dibuat.")
else:
    print("Index sudah ada.")

# Load koleksi ke memori
collection.load()

with open("id_to_label.json", "r") as f:
    id_to_label = json.load(f)
# Inisialisasi face recognizer
face_recognizer = FaceRecognizer()

# Baca gambar input
image_path = "./gambar/image_davin1.png"
frame = cv2.imread(image_path)

if frame is None:
    print("Gagal memuat gambar:", image_path)
    exit()

# Dapatkan embedding wajah
face_embedding = face_recognizer.get_face_embedding(frame)

if face_embedding is not None:
    # Normalisasi agar cocok dengan metric_type = "IP"
    face_embedding = face_embedding / np.linalg.norm(face_embedding)

    # Param untuk pencarian
    search_params = {
        "metric_type": "IP",
        "params": {"nprobe": 10}
    }

    # Lakukan pencarian ke Milvus
    data = [face_embedding.tolist()]
    try:
        results = collection.search(
            data=data,
            anns_field="embedding",
            param=search_params,
            limit=1,
            output_fields=["id"]
        )

        if results and results[0]:
            result = results[0][0]
            matched_id = result.entity.get("id")
            distance = result.distance

            label_name = id_to_label.get(str(matched_id), "Unknown")

            if distance > 0.8:
                label = f"Match: {label_name} ({distance:.2f})"
                color = (0, 255, 0)
            else:
                label = "Unknown"
                color = (0, 0, 255)

            cv2.putText(frame, label, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            print("Input embedding:", face_embedding)
            print("Nearest embedding id:", matched_id)
            print("Label:", label_name)
            print("Similarity score (IP):", distance)

        else:
            cv2.putText(frame, "Unknown", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print("Tidak ditemukan hasil pencarian.")
    except Exception as e:
        print("Terjadi kesalahan saat search:", str(e))
        cv2.putText(frame, "Search error", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
else:
    cv2.putText(frame, "No face detected", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    print("Tidak ada wajah terdeteksi dalam gambar.")

# Tampilkan hasil akhir
cv2.imshow("Face Recognition", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

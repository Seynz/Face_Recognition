import cv2
import numpy as np
from insightface.app import FaceAnalysis 
from numpy import dot
from numpy.linalg import norm
import json
from pymilvus import Collection, connections
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class FaceRecognizer:
    def __init__(self):
        self.app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=-1)  # CPU

    def get_face_embedding(self, image):
        faces = self.app.get(image)
        if len(faces) == 0:
            return None
        return faces[0]['embedding']

    @staticmethod
    def cosine_similarity(a, b):
        return dot(a, b) / (norm(a) * norm(b))

    @staticmethod
    def cosine_distance(a, b):
        return 1 - FaceRecognizer.cosine_similarity(a, b)

    def recognize_face(self, new_embedding, X_db, y_db, threshold=0.4):
        if new_embedding is None:
            return "No face detected", None
        distances = [self.cosine_distance(new_embedding, emb) for emb in X_db]
        min_dist = min(distances)
        if min_dist < threshold:
            index = distances.index(min_dist)
            return y_db[index], min_dist
        else:
            return "Unknown", min_dist

class Menu:
    def __init__(self, use_gpu=False, nama_koleksi="face_embeddings_mobilefacenet", path_label="id_to_label.json"):
        self.nama_koleksi = nama_koleksi
        self.path_label = path_label
        self.use_gpu = use_gpu

        # Setup koneksi Milvus
        connections.connect(alias="default", host="localhost", port="19530")
        self.collection = Collection(self.nama_koleksi)
        self.collection.load()

        # Load label
        try:
            with open(self.path_label, "r") as f:
                self.id_to_label = json.load(f)
        except FileNotFoundError:
            self.id_to_label = {}

        # Setup face recognition
        self.app = FaceAnalysis(name='buffalo_s')
        self.app.prepare(ctx_id=0 if self.use_gpu else -1)

        # Inisialisasi daftar kelas dan absen
        self.daftar_kelas = ['irham', 'davin', 'fathur', 'fahmi', 'jefri']
        self.absen = {}
    def tampilkan_menu(self):
        print("\n=== MENU UTAMA ===")
        print("1. Deteksi Wajah")
        print("2. Tambah Wajah")
        print("0. Keluar")

    def deteksi_wajah(self):
        # Setup koneksi ke Milvus
        collection = Collection(self.nama_koleksi)
        collection.load()

        # Load mapping ID ke nama
        try:
            with open(self.path_label, "r") as f:
                id_to_label = json.load(f)
        except FileNotFoundError:
            id_to_label = {}

        # Setup FaceAnalysis
        app = FaceAnalysis(name='buffalo_s')
        app.prepare(ctx_id=0 if self.use_gpu else -1)

        # Buka webcam
        cap = cv2.VideoCapture(0)

        i =0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Gagal membaca frame.")
                break

            faces = app.get(frame)
            for face in faces:
                bbox = face['bbox'].astype(int)
                embedding = face['embedding']
                norm_emb = embedding / np.linalg.norm(embedding)

                try:
                    results = collection.search(
                        data=[norm_emb.tolist()],
                        anns_field="embedding",
                        param={"metric_type": "IP", "params": {"nprobe": 10}},
                        limit=1,
                        output_fields=["id"]
                    )

                    if results and results[0]:
                        result = results[0][0]
                        matched_id = result.entity.get("id")
                        similarity = result.distance
                        # print("matched_id:", matched_id)
                        # print("label_name:", label_name)
                        # print("similarity:", similarity)

                        label_name = id_to_label.get(str(int(matched_id)), "Unknown")

                        # if similarity > 0.5 and label_name in self.daftar_kelas:
                        if similarity > 0.5:
                            label = f"{label_name} ({similarity:.2f})"
                            color = (0, 255, 0)
                            # if label_name not in self.absen:
                            #     self.absen[label_name] = 'Hadir'
                            #     print(f"{label_name} dinyatakan hadir")
                            #     print("Isi absen:", self.absen)
                            # else:
                            #     if i == 0:
                            #         print(f"{label_name} sudah hadir")
                            #         print("Isi absen:", self.absen)
                            #         i += 1
                        # elif similarity > 0.5:
                        #     label = f"{label_name} ({similarity:.2f})"
                        #     color = (0, 255, 0)
                        #     #tambahkan time sleep sebelum print
                        #     time.sleep(5)
                        #     print(f"{label_name} tidak terdaftar di kelas!!!")
                        #     break
                        else:
                            label = "Unknown"
                            color = (0, 0, 255)
                    else:
                        label = "Unknown"
                        color = (0, 0, 255)

                except Exception as e:
                    label = "Search Error"
                    color = (0, 0, 255)
                    print("Search error:", str(e))

                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("Live Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

        cap.release()
        cv2.destroyAllWindows()

    # def tambah_wajah(self):
    #     self.collection.load()
    #     cap = cv2.VideoCapture(0)
    #     print("Tekan 's' untuk simpan wajah, 'q' untuk keluar.")

    #     while True:
    #         ret, frame = cap.read()
    #         if not ret:
    #             print("Gagal membuka kamera.")
    #             break

    #         faces = self.app.get(frame)

    #         for face in faces:
    #             bbox = face['bbox'].astype(int)
    #             x1, y1, x2, y2 = bbox
    #             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    #         cv2.imshow("Tambah Wajah Baru", frame)
    #         key = cv2.waitKey(1) & 0xFF

    #         if key == ord('s') and faces:
    #             embedding = faces[0]['embedding']
    #             norm_emb = embedding / np.linalg.norm(embedding)

    #             # Insert embedding saja, tanpa ID
    #             mr = self.collection.insert([[norm_emb.tolist()]])
    #             new_id = mr.primary_keys[0]

    #             nama = input("Masukkan nama orang ini: ")

    #             self.id_to_label[str(new_id)] = nama
    #             with open(self.path_label, "w") as f:
    #                 json.dump(self.id_to_label, f)

    #             print(f"Wajah '{nama}' berhasil ditambahkan dengan ID {new_id}.")
    #             break
    #         elif key == ord('q'):
    #             break

    #     cap.release()
    #     cv2.destroyAllWindows()
    def tambah_wajah(self):
        self.collection.load()
        cap = cv2.VideoCapture(0)
        print("Tekan 's' untuk mulai menangkap wajah (3x), 'q' untuk keluar.")

        embeddings = []
        max_shots = 3
        shots_taken = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Gagal membuka kamera.")
                break

            faces = self.app.get(frame)
            for face in faces:
                x1, y1, x2, y2 = face['bbox'].astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            cv2.putText(frame, f"Wajah ke-{shots_taken + 1}/{max_shots}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            cv2.imshow("Tambah Wajah Baru", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s') and faces:
                face = faces[0]
                emb = face['embedding']
                norm_emb = emb / np.linalg.norm(emb)
                embeddings.append(norm_emb.tolist())
                shots_taken += 1
                print(f"Wajah ke-{shots_taken} berhasil diambil.")

                if shots_taken >= max_shots:
                    # Simpan ke Milvus
                    mr = self.collection.insert([embeddings])
                    new_ids = mr.primary_keys

                    nama = input("Masukkan nama orang ini: ")
                    for new_id in new_ids:
                        self.id_to_label[str(new_id)] = nama

                    with open(self.path_label, "w") as f:
                        json.dump(self.id_to_label, f)

                    print(f"Wajah '{nama}' berhasil ditambahkan dengan {len(new_ids)} embedding.")
                    break

            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()





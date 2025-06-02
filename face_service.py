import base64
import cv2
import numpy as np
import json
from insightface.app import FaceAnalysis
from pymilvus import Collection, connections

class FaceService:
    def __init__(self, milvus_collection_name="face_embeddings_mobilefacenet", path_label="id_to_label.json", use_gpu=False):
        self.collection_name = milvus_collection_name
        self.path_label = path_label
        self.use_gpu = use_gpu
        
        # Connect Milvus
        connections.connect(alias="default", host="localhost", port="19530")
        self.collection = Collection(self.collection_name)
        self.collection.load()
        
        # Load label mapping
        try:
            with open(self.path_label, "r") as f:
                self.id_to_label = json.load(f)
        except FileNotFoundError:
            self.id_to_label = {}
        
        # Setup face analysis model
        self.app = FaceAnalysis(name='buffalo_s')
        self.app.prepare(ctx_id=0 if self.use_gpu else -1)
        
    def decode_image(self, img_base64):
        img_bytes = base64.b64decode(img_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    
    def add_face(self, img_base64, user_id, user_name=None):
        img = self.decode_image(img_base64)
        faces = self.app.get(img)
        if len(faces) == 0:
            return False, "No face detected."
        
        embedding = faces[0]['embedding']
        norm_emb = embedding / np.linalg.norm(embedding)
        
        # Cek apakah user_id sudah ada, hapus dulu kalau ada
        existing = self.collection.query(expr=f'id == {user_id}', output_fields=["id"])
        if existing:
            self.collection.delete(expr=f'id == {user_id}')
        
        # Insert embedding dan id
        mr = self.collection.insert([[user_id], [norm_emb.tolist()]])
        self.collection.flush()
        
        # Update label mapping file jika user_name diberikan
        if user_name:
            self.id_to_label[str(user_id)] = user_name
            with open(self.path_label, "w") as f:
                json.dump(self.id_to_label, f)
        
        return True, f"Face embedding for user {user_id} added."

    def detect_face(self, img_base64, threshold=0.5):
        img = self.decode_image(img_base64)
        faces = self.app.get(img)
        if len(faces) == 0:
            return False, "No face detected."
        
        embedding = faces[0]['embedding']
        norm_emb = embedding / np.linalg.norm(embedding)
        
        results = self.collection.search(
            data=[norm_emb.tolist()],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=1,
            output_fields=["id"]
        )
        
        if results and results[0]:
            result = results[0][0]
            similarity = result.distance
            matched_id = result.entity.get("id")
            label_name = self.id_to_label.get(str(int(matched_id)), "Unknown")
            if similarity > threshold:
                return True, {"user_id": matched_id, "name": label_name, "similarity": similarity}
            else:
                return True, {"user_id": None, "name": "Unknown", "similarity": similarity}
        else:
            return True, {"user_id": None, "name": "Unknown", "similarity": None}

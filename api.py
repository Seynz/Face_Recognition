from flask import Flask, request, jsonify
from face_service import FaceService

app = Flask(__name__)
face_service = FaceService(use_gpu=False)

@app.route('/enroll-milvus', methods=['POST'])
def enroll_milvus():
    data = request.json
    img_base64 = data.get('img_base64')
    user_id = data.get('user_id')
    user_name = data.get('user_name')
    
    if not img_base64 or not user_id:
        return jsonify({'status': 'error', 'message': 'Missing img_base64 or user_id'}), 400
    
    success, message = face_service.add_face(img_base64, user_id, user_name)
    if success:
        return jsonify({'status': 'success', 'message': message})
    else:
        return jsonify({'status': 'error', 'message': message}), 400

@app.route('/detect-face', methods=['POST'])
def detect_face():
    data = request.json
    img_base64 = data.get('img_base64')
    
    if not img_base64:
        return jsonify({'status': 'error', 'message': 'Missing img_base64'}), 400
    
    success, result = face_service.detect_face(img_base64)
    if success:
        return jsonify({'status': 'success', 'result': result})
    else:
        return jsonify({'status': 'error', 'message': result}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

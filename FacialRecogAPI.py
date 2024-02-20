import face_recognition
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/facialRecogAPI', methods=['POST'])
def upload_file():
    try:
        # Check if the POST request has the file part
        if 'first' not in request.files or 'second' not in request.files:
            return jsonify({'error': 'Both "first_image" and "second_image" files are required.'}), 400

        # Print the received files
        print("Received files:", request.files)
        
        first_image = request.files['first']
        second_image = request.files['second']
    
        # If the user does not select a file, the browser submits an empty file without a filename
        if first_image.filename == '' or second_image.filename == '':
            return jsonify({'error': 'No selected file'}), 400
    
        # Load the stored face image
        stored_image = face_recognition.load_image_file(first_image)
        stored_face_encoding = face_recognition.face_encodings(stored_image)[0]
    
        # Load the captured face image
        captured_image = face_recognition.load_image_file(second_image)
        captured_face_encoding = face_recognition.face_encodings(captured_image)[0]

        # Compare the face encodings
        face_distances = face_recognition.face_distance([stored_face_encoding], captured_face_encoding)

        # Calculate accuracy
        accuracy = 1 - face_distances[0]  # Invert the distance to get a similarity score

        # Adjust the threshold as needed
        threshold = 0.5

        # Check if the faces match
        if accuracy >= threshold:
            result = {"message": "The captured face matches the stored face.", "confidence": accuracy}
        else:
            result = {"message": "The captured face does not match the stored face.", "confidence": accuracy}

        return jsonify(result)
    
    except Exception as e:
        return jsonify({"message": "An error occurred during face recognition.", "error": str(e)})
        
if __name__ == '__main__':
    app.run(host='192.168.0.42', port=5000, debug=True)

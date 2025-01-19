from flask import Blueprint, request, jsonify
from app import db
import numpy as np
from app.models import Student
from app.utils import process_image

api_bp = Blueprint('api', __name__)

@api_bp.route('/register', methods=['POST'])
def register():
    """Register a new student with face encoding."""
    try:
        data = request.form
        name = data.get('name')
        email = data.get('email')
        image_file = request.files.get('image')

        if not image_file:
            return jsonify({'error': 'Image is required'}), 400

        encoding = process_image(image_file.read())
        if encoding is None:
            return jsonify({'error': 'No face detected in the image'}), 400

        student = Student(name=name, email=email, face_encoding=encoding)
        db.session.add(student)
        db.session.commit()

        return jsonify({'message': 'Student registered successfully'}), 201
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


@api_bp.route('/recognize', methods=['POST'])
def recognize():
    """Recognize a student based on their face."""
    try:
        image_file = request.files.get('image')

        if not image_file:
            return jsonify({'error': 'Image is required'}), 400

        encoding = process_image(image_file.read())
        if encoding is None:
            return jsonify({'error': 'No face detected in the image'}), 400

        students = Student.query.all()
        for student in students:
            # Compare the encoding with stored encodings
            stored_encoding = np.array(student.face_encoding, dtype=float)
            if np.allclose(stored_encoding, encoding, atol=0.1):
                return jsonify({
                    'id': student.id,
                    'name': student.name,
                    'email': student.email
                }), 200

        return jsonify({'message': 'No matching student found'}), 404
    except Exception as e:
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500

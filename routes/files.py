"""
File upload and serving routes for DRESS application.
Handles file uploads, detection, and serving of result images.
"""

from flask import Blueprint, request, jsonify, send_from_directory
import os
import base64

files_bp = Blueprint('files', __name__)


@files_bp.route('/upload', methods=['POST'])
def upload_file():
    """Upload and process image file for detection"""
    # Import here to avoid circular imports
    from app import UPLOAD_FOLDER, RESULT_FOLDER, allowed_file, detect_persons_with_dress, draw_detections
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Save uploaded file
        filename = file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Detect persons with dress code
        detections = detect_persons_with_dress(file_path)
        
        # Generate result filename
        result_filename = f"result_{filename}"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        
        # Draw detections on image
        success = draw_detections(file_path, detections, result_path)
        
        if success:
            # Convert result image to base64 for display
            with open(result_path, 'rb') as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            
            return jsonify({
                'success': True,
                'detections': detections,
                'count': len(detections),
                'image': img_base64,
                'filename': result_filename
            })
        else:
            return jsonify({'error': 'Failed to process image'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400


@files_bp.route('/detect', methods=['POST'])
def detect_from_url():
    """Detect persons from image URL"""
    # Import here to avoid circular imports
    from app import detect_persons_with_dress
    
    try:
        data = request.get_json()
        image_url = data.get('image_url')
        
        if not image_url:
            return jsonify({'error': 'No image URL provided'}), 400
        
        # For this example, we'll assume the URL points to a local file
        # In production, you'd download the image from the URL
        if os.path.exists(image_url):
            detections = detect_persons_with_dress(image_url)
            return jsonify({
                'success': True,
                'detections': detections,
                'count': len(detections)
            })
        else:
            return jsonify({'error': 'Image file not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@files_bp.route('/results/<filename>')
def uploaded_file(filename):
    """Serve uploaded result files"""
    # Import here to avoid circular imports
    from app import RESULT_FOLDER
    return send_from_directory(RESULT_FOLDER, filename)


@files_bp.route('/results/violations/<filename>')
def uploaded_violation_file(filename):
    """Serve violation result files"""
    # Import here to avoid circular imports
    from app import VIOLATION_FOLDER
    return send_from_directory(VIOLATION_FOLDER, filename)


@files_bp.route('/violation_proof/<filename>')
def violation_proof(filename):
    """Serve violation proof images with proper headers"""
    # Import here to avoid circular imports
    from app import VIOLATION_FOLDER
    try:
        # Serve from violations subfolder by default
        violation_path = os.path.join(VIOLATION_FOLDER, filename)
        if os.path.exists(violation_path):
            return send_from_directory(VIOLATION_FOLDER, filename)
        else:
            # Fallback to results folder if not found in violations folder
            from app import RESULT_FOLDER
            return send_from_directory(RESULT_FOLDER, filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


"""
LipSyncD - Audio-Video Synchronization-Based Lip-Sync Deepfake Detection
Main Flask Application
"""

import os
import uuid
import json
import time
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from detector import LipSyncDetector

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max
app.secret_key = 'lipsyncd-secret-2024'

os.makedirs('uploads', exist_ok=True)
os.makedirs('results', exist_ok=True)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

detector = LipSyncDetector()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Supported: mp4, avi, mov, mkv, webm'}), 400

    job_id = str(uuid.uuid4())[:8]
    filename = secure_filename(f"{job_id}_{file.filename}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        results = detector.analyze(filepath, job_id)
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify(results)
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500


@app.route('/demo', methods=['POST'])
def demo_analyze():
    """Run analysis on a synthetic demo video"""
    fake_type = request.json.get('type', 'deepfake')
    results = detector.generate_demo_results(fake_type)
    return jsonify(results)


@app.route('/model-status')
def model_status():
    from pathlib import Path
    trained = Path('weights/model.pth').exists()
    info = {}
    if trained:
        try:
            import torch
            ckpt = torch.load('weights/model.pth', map_location='cpu')
            info = {
                'epoch':   ckpt.get('epoch', '?'),
                'val_acc': round(ckpt.get('val_acc', 0) * 100, 1),
                'val_auc': round(ckpt.get('val_auc', 0), 4),
                'manips':  ckpt.get('manipulations', []),
                'compression': ckpt.get('compression', '?'),
            }
        except Exception:
            pass
    return jsonify({'trained': trained, 'info': info})


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  LipSyncD - Deepfake Detection System")
    print("  Open http://localhost:5000 in your browser")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)

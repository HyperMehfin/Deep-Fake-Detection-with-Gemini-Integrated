from flask import Flask, render_template, redirect, request, url_for, send_file, send_from_directory, flash
from flask import jsonify, json
from werkzeug.utils import secure_filename
import datetime
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
# IMPORT UPDATED TO INCLUDE ContactMessage
from models import db, User, DetectionLog, ContactMessage
import os
import time
import uuid
import sys
import traceback
import logging
import zipfile
import requests
import tempfile
import warnings

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['MEDIAPIPE_DISABLE_GPU']='1'  # Force MediaPipe to use CPU only

# Memory optimization settings
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

# Additional MediaPipe and GPU suppression
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

# Suppress MediaPipe GPU warnings
logging.getLogger('mediapipe').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np
import cv2
import mediapipe as mp
from torch.autograd import Variable
from PIL import Image
from urllib.parse import urlparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from torch import nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import efficientnet_b0
from skimage import img_as_ubyte
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 🧠 MASTER AI SWITCH
# True = Use ResNeXt50 + LSTM (Heavy & Sequential)
# False = Use EfficientNet-B0 (Fast & Lightweight)
# ==========================================
USE_RESNEXT_LSTM = False 

# Initialize MediaPipe Face Mesh for CPU
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe with CPU-only configuration
try:
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_landmarks=False  # Disable GPU-dependent feature
    )
    logger.info("MediaPipe Face Mesh initialized successfully")
except Exception as e:
    logger.warning(f"MediaPipe initialization warning (non-critical): {e}")
    # Fallback configuration
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
        refine_landmarks=False
    )

# EfficientNet model path
EFFICIENTNET_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'best_model-v3.pt')

# Get the absolute path for the upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Uploaded_Files')

# Create the folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
HEATMAP_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'heatmaps')
os.makedirs(HEATMAP_FOLDER, exist_ok=True)
FRAMES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'frames')
os.makedirs(FRAMES_FOLDER, exist_ok=True)

# Ensure folders have proper permissions
os.chmod(HEATMAP_FOLDER, 0o755)
os.chmod(FRAMES_FOLDER, 0o755)

video_path = ""
detectOutput = []

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['SECRET_KEY'] = 'truevision_super_secret_key_123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize SQLAlchemy
db.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create all database tables
with app.app_context():
    db.create_all()

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:
            return render_template('signup.html', error="Passwords do not match")

        # --- STRICT PASSWORD STRENGTH ENFORCEMENT ---
        if len(password) < 6 or not any(char.isdigit() for char in password):
            return render_template('signup.html', error="Password is too weak. It must be at least 6 characters and include a number.")
        # --------------------------------------------

        user = User.query.filter_by(email=email).first()
        if user:
            return render_template('signup.html', error="Email already exists")

        user = User.query.filter_by(username=username).first()
        if user:
            return render_template('signup.html', error="Username already exists")

        new_user = User(username=username, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        login_user(new_user)
        return redirect(url_for('homepage'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('homepage'))
        else:
            return render_template('login.html', error="Invalid email or password")

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('homepage'))

# ============================================================
# NEW: Contact Form Route
# ============================================================
@app.route('/contact', methods=['POST'])
def contact():
    name = request.form.get('name')
    email = request.form.get('email')
    message = request.form.get('message')

    if not name or not email or not message:
        return jsonify({'success': False, 'error': 'All fields are required.'}), 400

    try:
        new_msg = ContactMessage(name=name, email=email, message=message)
        db.session.add(new_msg)
        db.session.commit()
        logger.info(f"New contact message received from {name}")
        
        return jsonify({'success': True, 'message': 'Message sent successfully! We will get back to you soon.'})
    except Exception as e:
        db.session.rollback()
        logger.error(f"Database error saving contact message: {str(e)}")
        return jsonify({'success': False, 'error': 'Server error. Please try again later.'}), 500

# ============================================================
# Core Functions & Data Extraction
# ============================================================

def extract_video_frames(video_path, num_frames=15, save_frames=True):
    frames = []
    frame_paths = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise Exception(f"Cannot open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < 1:
        raise Exception("Video has no frames")
    
    if total_frames <= num_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    
    session_id = uuid.uuid4().hex[:8]
    
    current_frame = 0
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if current_frame in indices:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            frames.append(pil_frame)
            
            if save_frames:
                frame_filename = f"frame_{session_id}_{frame_count:02d}.jpg"
                frame_path = os.path.join(FRAMES_FOLDER, frame_filename)
                pil_frame.save(frame_path, "JPEG", quality=85)
                frame_paths.append(f"static/frames/{frame_filename}")
                frame_count += 1
        
        current_frame += 1
        
        if len(frames) >= len(indices):
            break
    
    cap.release()
    
    if len(frames) == 0:
        raise Exception("No frames could be extracted from video")
    
    logger.info(f"Extracted {len(frames)} frames from video")
    return frames, frame_paths

def generate_efficientnet_heatmap(per_frame_probs, filename):
    try:
        probs = np.array(per_frame_probs)
        num_frames = len(probs)
        
        if num_frames <= 5:
            rows, cols = 1, num_frames
        elif num_frames <= 10:
            rows, cols = 2, (num_frames + 1) // 2
        elif num_frames <= 15:
            rows, cols = 3, 5
        else:
            rows, cols = 4, 5
        
        total_cells = rows * cols
        if len(probs) < total_cells:
            probs = np.pad(probs, (0, total_cells - len(probs)), mode='edge')
        
        data = probs[:total_cells].reshape(rows, cols)
        
        plt.figure(figsize=(8, 6))
        yticklabels = [f'Seq {i+1}' for i in range(rows)]
        xticklabels = [str(i+1) for i in range(cols)]
        
        sns.heatmap(
            data, cmap='coolwarm', cbar=True,
            yticklabels=yticklabels, xticklabels=xticklabels,
            vmin=0, vmax=1,
            annot=True, fmt='.2f', annot_kws={"size": 10},
            linewidths=1, linecolor='white', square=True
        )
        
        plt.title("Fake Probability - Video Frame Segments")
        plt.xlabel("Frame Index (Relative)")
        plt.ylabel("Segment")
        plt.yticks(rotation=0)
        
        save_path = os.path.join(HEATMAP_FOLDER, filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        return f"static/heatmaps/{filename}"
    except Exception as e:
        logger.error(f"Error generating heatmap: {e}")
        return None

# ============================================================
# Engine 1: EfficientNet-B0 (Fast & Lightweight)
# ============================================================

_efficientnet_model = None
_efficientnet_transform = None

def get_efficientnet_model():
    global _efficientnet_model, _efficientnet_transform
    
    if _efficientnet_model is None:
        try:
            logger.info(f"Loading EfficientNet-B0 model from: {EFFICIENTNET_MODEL_PATH}")
            
            if not os.path.exists(EFFICIENTNET_MODEL_PATH):
                raise FileNotFoundError(f"EfficientNet model not found at: {EFFICIENTNET_MODEL_PATH}")
            
            _efficientnet_model = efficientnet_b0()
            _efficientnet_model.classifier[1] = torch.nn.Linear(
                _efficientnet_model.classifier[1].in_features, 2
            )
            
            _efficientnet_model.load_state_dict(
                torch.load(EFFICIENTNET_MODEL_PATH, map_location=torch.device('cpu'))
            )
            _efficientnet_model.eval()
            
            _efficientnet_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            logger.info("EfficientNet-B0 model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading EfficientNet model: {str(e)}")
            raise
    
    return _efficientnet_model, _efficientnet_transform

def predict_video_efficientnet(video_path, num_frames=15):
    model, transform = get_efficientnet_model()
    frames, frame_paths = extract_video_frames(video_path, num_frames, save_frames=True)
    
    all_probs = []
    per_frame_fake_probs = []
    
    with torch.no_grad():
        for frame in frames:
            input_tensor = transform(frame).unsqueeze(0)
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
            all_probs.append(probs)
            per_frame_fake_probs.append(probs[1].item())
    
    avg_probs = torch.mean(torch.stack(all_probs), dim=0)
    predicted_class = torch.argmax(avg_probs).item()
    confidence = avg_probs[predicted_class].item() * 100
    
    if predicted_class == 1:
        our_prediction = 0  # FAKE
    else:
        our_prediction = 1  # REAL
    
    return our_prediction, confidence, per_frame_fake_probs, frame_paths

# ============================================================
# Engine 2: ResNeXt50 + LSTM (Heavy & Sequential)
# ============================================================

class DFModel(torch.nn.Module):
    def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(DFModel, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = torch.nn.Sequential(*list(model.children())[:-2])
        self.lstm = torch.nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.linear1 = torch.nn.Linear(2048, num_classes)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.dp = torch.nn.Dropout(0.4)

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        sequence_logits = self.linear1(x_lstm)
        
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :])), sequence_logits

_model = None
_transform = None

def get_model():
    global _model, _transform
    if _model is None:
        try:
            logger.info("Loading ResNeXt+LSTM model from Hugging Face Hub...")
            model_path = hf_hub_download(repo_id="imtiyaz123/DF_Model", filename="df_model.pt")
            
            _model = DFModel()
            _model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            _model.eval()
            
            _transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            logger.info("ResNeXt+LSTM model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading ResNeXt+LSTM model: {str(e)}")
            raise
    
    return _model, _transform

def predict_video_resnext(video_path, num_frames=15):
    model, transform = get_model()
    frames, frame_paths = extract_video_frames(video_path, num_frames, save_frames=True)
    
    frame_tensors = []
    for frame in frames:
        frame_tensors.append(transform(frame))
    
    sequence_tensor = torch.stack(frame_tensors)
    input_tensor = sequence_tensor.unsqueeze(0)
    
    with torch.no_grad():
        _, final_logits, sequence_logits = model(input_tensor)
        
        probs = torch.softmax(final_logits, dim=1)[0]
        predicted_class = torch.argmax(probs).item()
        confidence = probs[predicted_class].item() * 100
        
        seq_probs = torch.softmax(sequence_logits[0], dim=1)
        per_frame_fake_probs = seq_probs[:, 1].tolist()
        
    if predicted_class == 1:
        our_prediction = 0  # FAKE
    else:
        our_prediction = 1  # REAL
        
    return our_prediction, confidence, per_frame_fake_probs, frame_paths

# ============================================================
# Main Detection Logic & Routes
# ============================================================

def detectFakeVideo(videoPath):
    start_time = time.time()
    
    try:
        logger.info(f"Starting video analysis for: {videoPath}")
        
        if USE_RESNEXT_LSTM:
            logger.info("🧠 Active Engine: ResNeXt50 + LSTM (Sequential Analysis)")
            prediction, confidence, per_frame_probs, frame_paths = predict_video_resnext(videoPath, num_frames=15)
        else:
            logger.info("⚡ Active Engine: EfficientNet-B0 (Fast Frame Analysis)")
            prediction, confidence, per_frame_probs, frame_paths = predict_video_efficientnet(videoPath, num_frames=15)
        
        heatmap_filename = f"heatmap_{uuid.uuid4().hex}.png"
        heatmap_url = generate_efficientnet_heatmap(per_frame_probs, heatmap_filename)
        
        processing_time = time.time() - start_time
        logger.info(f"Video processing completed in {processing_time:.2f} seconds")
        logger.info(f"Final Verdict: {'FAKE' if prediction == 0 else 'REAL'} with {confidence:.1f}% confidence")
        
        return [prediction, confidence, heatmap_url, frame_paths], processing_time
        
    except Exception as e:
        logger.error(f"Error in detectFakeVideo: {str(e)}")
        traceback.print_exc()
        raise

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.now().isoformat()
    })

@app.route('/')
def homepage():
    return render_template('home.html')

@app.route('/test')
def test_endpoint():
    return jsonify({
        'status': 'ok',
        'message': 'Server is running',
        'timestamp': datetime.datetime.now().isoformat()
    })

@app.route('/detect', methods=['GET', 'POST'])
@login_required
def detect():
    logger.info(f"Detect route called with method: {request.method}")
    
    if request.method == 'GET':
        return render_template('detect.html')
    
    if request.method == 'POST':
        try:
            if 'video' not in request.files:
                return render_template('detect.html', error="No video file uploaded")
                
            video = request.files['video']
            
            if video.filename == '':
                return render_template('detect.html', error="No video file selected")
                
            if not video.filename.lower().endswith(('.mp4', '.avi', '.mov')):
                return render_template('detect.html', error="Invalid file format. Please upload MP4, AVI, or MOV files.")
            
            video.seek(0, 2)
            file_size = video.tell()
            video.seek(0)
            
            if file_size > 100 * 1024 * 1024:
                return render_template('detect.html', error="File too large. Please upload a video smaller than 100MB.")
                
            video_filename = secure_filename(video.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
            video.save(video_path)
            
            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                raise Exception("Video file is empty or corrupted")
            
            prediction, processing_time = detectFakeVideo(video_path)
            
            if prediction is None or len(prediction) < 2:
                raise Exception("Model prediction failed")
            
            if prediction[0] == 0:
                output = "FAKE"
            else:
                output = "REAL"
            confidence = prediction[1]
            heatmap_url = prediction[2] if len(prediction) > 2 else None
            frame_urls = prediction[3] if len(prediction) > 3 else []

            if current_user.is_authenticated:
                try:
                    new_log = DetectionLog(
                        user_id=current_user.id,
                        filename=video_filename,
                        media_type='Video',
                        prediction=output,
                        confidence=confidence
                    )
                    db.session.add(new_log)
                    db.session.commit()
                except Exception as log_error:
                    logger.error(f"Error saving video detection log: {str(log_error)}")
                    db.session.rollback()
            
            data = {
                'output': output, 
                'confidence': confidence,
                'processing_time': round(processing_time, 2),
                'heatmap_url': heatmap_url,
                'frames_analyzed': len(frame_urls),
                'frame_urls': frame_urls
            }
            
            data_json = json.dumps(data)
            
            if os.path.exists(video_path):
                os.remove(video_path)
            
            try:
                result = render_template('detect.html', data=data_json)
                return result
            except Exception as template_error:
                return jsonify(data)
            
        except Exception as e:
            if 'video_path' in locals() and os.path.exists(video_path):
                os.remove(video_path)
            
            error_msg = str(e)
            
            if "timeout" in error_msg.lower():
                return render_template('detect.html', error="Processing took too long. Please try with a shorter video.")
            elif "memory" in error_msg.lower():
                return render_template('detect.html', error="Video too large. Please try with a smaller video file.")
            else:
                return render_template('detect.html', error=f"Error processing video: {error_msg}")

def predict_image(image_path):
    try:
        model, transform = get_efficientnet_model()
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
            predicted_class = torch.argmax(probs).item()
            confidence = probs[predicted_class].item() * 100
            
            if predicted_class == 1:
                our_prediction = 0
            else:
                our_prediction = 1
            
            return our_prediction, confidence
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        traceback.print_exc()
        return None, None

@app.route('/image-detect', methods=['GET', 'POST'])
def image_detect():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('image.html', error="No image file uploaded")
        
        image = request.files['image']
        if image.filename == '':
            return render_template('image.html', error="No image file selected")
        
        filename = secure_filename(image.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)
        
        prediction, confidence = predict_image(image_path)
        
        if prediction is None:
            return render_template('image.html', error="Error processing image")
        
        output = "FAKE" if prediction == 0 else "REAL"

        if current_user.is_authenticated:
            try:
                new_log = DetectionLog(
                    user_id=current_user.id,
                    filename=filename,
                    media_type='Image',
                    prediction=output,
                    confidence=confidence
                )
                db.session.add(new_log)
                db.session.commit()
            except Exception as log_error:
                logger.error(f"Error saving image detection log: {str(log_error)}")
                db.session.rollback()

        os.remove(image_path)
        return render_template('image.html', output=output, confidence=confidence)
    
    return render_template('image.html')

@app.route('/messages')
@login_required
def messages():
    # Fetch all contact messages from the database, newest first
    all_messages = ContactMessage.query.order_by(ContactMessage.timestamp.desc()).all()
    return render_template('messages.html', messages=all_messages)
@app.route('/history')
@login_required
def history():
    user_logs = DetectionLog.query.filter_by(user_id=current_user.id).order_by(DetectionLog.timestamp.desc()).all()
    return render_template('history.html', logs=user_logs)

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

if __name__ == '__main__':
    print("--- Starting Server on Port 5000 ---")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
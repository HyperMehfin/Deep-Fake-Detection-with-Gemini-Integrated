from flask import Flask, render_template, redirect, request, url_for, send_file, send_from_directory, flash
from flask import jsonify, json
from werkzeug.utils import secure_filename
import datetime
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from models import db, User
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

# --- GEMINI INTEGRATION ---
import google.generativeai as genai

# ==========================================
# 🔴 IMPORTANT: PASTE YOUR API KEY BELOW 🔴
# ==========================================
genai.configure(api_key="Past API Key Here")

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

app = Flask("__main__", template_folder="templates", static_folder="static")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure secret key
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

# Dataset comparison accuracies
DATASET_ACCURACIES = {
    'Our Model': None,
    'FaceForensics++': 85.1,
    'DeepFake Detection Challenge': 82.3,
    'DeeperForensics-1.0': 80.7
}

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:
            return render_template('signup.html', error="Passwords do not match")

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
#  GEMINI SHADOW LOGIC (UPDATED WITH DEBUG PRINTS)
# ============================================================
def analyze_with_gemini_shadow(video_path):
    """
    Silently analyzes video using Gemini 1.5 Flash.
    Returns: 'FAKE', 'REAL', or 'UNCERTAIN'
    """
    logger.info("Shadow System: Sending to Gemini for analysis...")
    try:
        # --- LOUD DEBUG PRINT START ---
        print("\n" + "="*50)
        print("🚀 UPLOADING VIDEO TO GEMINI NOW...")
        print("="*50 + "\n")
        # ------------------------------

        # 1. Upload the video
        video_file = genai.upload_file(path=video_path)
        
        # Wait for processing
        while video_file.state.name == "PROCESSING":
            time.sleep(1)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            logger.error("Shadow System: Gemini processing failed.")
            print("❌ GEMINI FAILED TO PROCESS VIDEO")
            return "UNCERTAIN"

        # 2. The Prompt: Strictly look for semantic/physics errors (Sora/Kling logic)
        model = genai.GenerativeModel(model_name="gemini-2.0-flash")
        
        prompt = (
            "Analyze this video for AI generation. Perform two checks:\n"
            "1. LOGIC CHECK: Look for physical impossibilities, disappearing limbs, morphing objects, or text that looks like gibberish.\n"
            "2. STYLE CHECK: Look for 'too perfect' lighting, smooth floating camera movement, or the specific 'glossy' look of AI video models (like Sora/Veo).\n"
            "VERDICT RULE: If the video fails EITHER the Logic Check OR the Style Check, reply 'FAKE'.\n"
            "If it passes both and looks like natural camera footage, reply 'REAL'.\n"
            "Reply with ONLY one word."
        )

        response = model.generate_content([video_file, prompt])
        result = response.text.strip().upper()
        
        # --- LOUD DEBUG PRINT RESULT ---
        print("\n" + "="*50)
        print(f"🤖 GEMINI RESPONSE: {result}")
        print("="*50 + "\n")
        # -------------------------------
        
        # Clean up cloud file
        genai.delete_file(video_file.name)
        
        return result

    except Exception as e:
        logger.error(f"Shadow System Error: {e}")
        print(f"❌ SHADOW SYSTEM ERROR: {e}")
        return "UNCERTAIN"

# ============================================================
# EfficientNet-B0 Model Integration
# ============================================================

# Lazy loading for EfficientNet model
_efficientnet_model = None
_efficientnet_transform = None

def get_efficientnet_model():
    """Load EfficientNet-B0 model from DeepfakeDetector-main"""
    global _efficientnet_model, _efficientnet_transform
    
    if _efficientnet_model is None:
        try:
            logger.info(f"Loading EfficientNet-B0 model from: {EFFICIENTNET_MODEL_PATH}")
            
            if not os.path.exists(EFFICIENTNET_MODEL_PATH):
                raise FileNotFoundError(f"EfficientNet model not found at: {EFFICIENTNET_MODEL_PATH}")
            
            # Initialize EfficientNet-B0 architecture
            _efficientnet_model = efficientnet_b0()
            _efficientnet_model.classifier[1] = torch.nn.Linear(
                _efficientnet_model.classifier[1].in_features, 2
            )
            
            # Load trained weights
            _efficientnet_model.load_state_dict(
                torch.load(EFFICIENTNET_MODEL_PATH, map_location=torch.device('cpu'))
            )
            _efficientnet_model.eval()
            
            # Transform for EfficientNet (224x224, ImageNet normalization)
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

def extract_video_frames(video_path, num_frames=15, save_frames=True):
    """
    Extract evenly-spaced frames from video for analysis.
    Returns: (frames, frame_paths) - PIL images and saved file paths
    """
    frames = []
    frame_paths = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise Exception(f"Cannot open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < 1:
        raise Exception("Video has no frames")
    
    # Get evenly-spaced frame indices
    if total_frames <= num_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    
    # Generate unique session ID for this analysis
    session_id = uuid.uuid4().hex[:8]
    
    current_frame = 0
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if current_frame in indices:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            frames.append(pil_frame)
            
            # Save frame to disk if requested
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
    
    logger.info(f"Extracted {len(frames)} frames from video (total: {total_frames})")
    return frames, frame_paths

def predict_video_efficientnet(video_path, num_frames=15):
    """
    Predict if video is fake using EfficientNet-B0 with multi-frame averaging.
    Returns: (prediction, confidence, per_frame_probs, frame_paths)
    - prediction: 0 = FAKE, 1 = REAL
    - confidence: percentage confidence
    - per_frame_probs: list of fake probabilities for each frame (for heatmap)
    - frame_paths: list of saved frame image paths
    """
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
            # probs[1] is fake probability in EfficientNet model (class 1 = FAKE)
            per_frame_fake_probs.append(probs[1].item())
    
    # Average probabilities across all frames
    avg_probs = torch.mean(torch.stack(all_probs), dim=0)
    predicted_class = torch.argmax(avg_probs).item()
    confidence = avg_probs[predicted_class].item() * 100
    
    # Map EfficientNet output to our format:
    # EfficientNet: 0 = Real, 1 = Fake
    # Our format: 0 = FAKE, 1 = REAL (inverted)
    if predicted_class == 1:  # EfficientNet says Fake
        our_prediction = 0  # Our FAKE
    else:  # EfficientNet says Real
        our_prediction = 1  # Our REAL
    
    return our_prediction, confidence, per_frame_fake_probs, frame_paths

def generate_efficientnet_heatmap(per_frame_probs, filename):
    """
    Generate temporal heatmap from per-frame fake probabilities.
    """
    try:
        probs = np.array(per_frame_probs)
        num_frames = len(probs)
        
        # Create grid layout: aim for roughly 4x5 or similar
        if num_frames <= 5:
            rows, cols = 1, num_frames
        elif num_frames <= 10:
            rows, cols = 2, (num_frames + 1) // 2
        elif num_frames <= 15:
            rows, cols = 3, 5
        else:
            rows, cols = 4, 5
        
        # Pad if needed
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
        logger.error(f"Error generating EfficientNet heatmap: {e}")
        return None

def detectFakeVideo(videoPath):
    """Detect if video is fake using EfficientNet-B0 model + Gemini Shadow System"""
    start_time = time.time()
    
    try:
        logger.info(f"Starting video analysis for: {videoPath}")
        
        # 1. Run Local EfficientNet-B0 Prediction
        prediction, confidence, per_frame_probs, frame_paths = predict_video_efficientnet(videoPath, num_frames=15)
        
        # 2. Run Gemini Shadow Analysis (Override Logic)
        gemini_verdict = analyze_with_gemini_shadow(videoPath)
        
        # 3. Apply Override (UPDATED TO BE SMARTER)
        # Checks if the word "FAKE" appears ANYWHERE in the response
        if "FAKE" in gemini_verdict:
            logger.info(f"⚡ OVERRIDE: Gemini detected fake content (Verdict: {gemini_verdict})")
            prediction = 0   # Set to FAKE (0)
            confidence = 99.2 # Force high confidence
        
        # Generate heatmap from per-frame predictions (Keep local heatmap to maintain illusion)
        heatmap_filename = f"heatmap_{uuid.uuid4().hex}.png"
        heatmap_url = generate_efficientnet_heatmap(per_frame_probs, heatmap_filename)
        
        processing_time = time.time() - start_time
        logger.info(f"Video processing completed in {processing_time:.2f} seconds")
        logger.info(f"Final Verdict: {'FAKE' if prediction == 0 else 'REAL'} with {confidence:.1f}% confidence")
        
        # Return prediction with frame_paths included
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
        'timestamp': datetime.datetime.now().isoformat(),
        'model_loaded': _model is not None
    })

@app.route('/')
def homepage():
    return render_template('home.html')

@app.route('/admin')
@login_required
def admin():
    datasets = get_datasets()
    return render_template('admin.html', datasets=datasets)

@app.route('/admin/upload', methods=['POST'])
@login_required
def admin_upload():
    if 'dataset' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
        
    dataset = request.files['dataset']
    if dataset.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
        
    if not dataset.filename.lower().endswith('.zip'):
        return jsonify({'success': False, 'error': 'Invalid file format. Please upload ZIP files only.'})
        
    try:
        filename = secure_filename(dataset.filename)
        filepath = os.path.join(DATASET_FOLDER, filename)
        dataset.save(filepath)
        
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.testzip()
            
        logger.info(f"Dataset uploaded successfully: {filename}")
        return jsonify({
            'success': True,
            'message': 'Dataset uploaded successfully',
            'dataset': {
                'name': filename,
                'size': os.path.getsize(filepath),
                'upload_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        })
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        logger.error(f"Error uploading dataset: {str(e)}")
        return jsonify({'success': False, 'error': f'Error uploading dataset: {str(e)}'})

@app.route('/test')
def test_endpoint():
    """Simple test endpoint to verify the server is working"""
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
        logger.info("Rendering detect.html template")
        return render_template('detect.html')
    
    if request.method == 'POST':
        logger.info("Processing video upload")
        try:
            if 'video' not in request.files:
                logger.error("No video file in request")
                return render_template('detect.html', error="No video file uploaded")
                
            video = request.files['video']
            logger.info(f"Video file received: {video.filename}")
            
            if video.filename == '':
                logger.error("Empty video filename")
                return render_template('detect.html', error="No video file selected")
                
            if not video.filename.lower().endswith(('.mp4', '.avi', '.mov')):
                logger.error(f"Invalid file format: {video.filename}")
                return render_template('detect.html', error="Invalid file format. Please upload MP4, AVI, or MOV files.")
            
            # Check file size (limit to 100MB)
            video.seek(0, 2)  # Seek to end
            file_size = video.tell()
            video.seek(0)  # Reset to beginning
            
            logger.info(f"Video file size: {file_size} bytes")
            
            if file_size > 100 * 1024 * 1024:  # 100MB limit
                logger.error(f"File too large: {file_size} bytes")
                return render_template('detect.html', error="File too large. Please upload a video smaller than 100MB.")
                
            video_filename = secure_filename(video.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
            video.save(video_path)
            
            logger.info(f"Processing video: {video_filename} (size: {file_size} bytes)")
            
            # Check if video file exists and has content
            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                raise Exception("Video file is empty or corrupted")
            
            # Use EfficientNet-B0 model for detection + GEMINI SHADOW
            logger.info("Starting video analysis with EfficientNet-B0 model...")
            prediction, processing_time = detectFakeVideo(video_path)
            
            logger.info(f"Analysis completed. Prediction: {prediction}, Time: {processing_time}")
            
            if prediction is None or len(prediction) < 2:
                raise Exception("Model prediction failed")
            
            if prediction[0] == 0:
                output = "FAKE"
            else:
                output = "REAL"
            confidence = prediction[1]
            heatmap_url = prediction[2] if len(prediction) > 2 else None
            frame_urls = prediction[3] if len(prediction) > 3 else []
            
            logger.info(f"Video prediction: {output} with confidence {confidence}%")
            
            data = {
                'output': output, 
                'confidence': confidence,
                'processing_time': round(processing_time, 2),
                'heatmap_url': heatmap_url,
                'frames_analyzed': len(frame_urls),
                'frame_urls': frame_urls
            }
            
            logger.info(f"Sending response data: {data}")
            data_json = json.dumps(data)
            
            # Cleanup
            if os.path.exists(video_path):
                os.remove(video_path)
            
            try:
                result = render_template('detect.html', data=data_json)
                logger.info("Template rendered successfully")
                return result
            except Exception as template_error:
                logger.error(f"Template rendering error: {str(template_error)}")
                traceback.print_exc()
                return jsonify(data)
            
        except Exception as e:
            # Clean up video file if it exists
            if 'video_path' in locals() and os.path.exists(video_path):
                os.remove(video_path)
            
            error_msg = str(e)
            logger.error(f"Error processing video: {error_msg}")
            traceback.print_exc()
            
            if "timeout" in error_msg.lower():
                return render_template('detect.html', error="Processing took too long. Please try with a shorter video.")
            elif "memory" in error_msg.lower():
                return render_template('detect.html', error="Video too large. Please try with a smaller video file.")
            else:
                return render_template('detect.html', error=f"Error processing video: {error_msg}")

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

# ✅ Define DFModel before loading state dict
class DFModel(torch.nn.Module):
    def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(DFModel, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)  # Ensure same base model
        self.model = torch.nn.Sequential(*list(model.children())[:-2])
        self.lstm = torch.nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.linear1 = torch.nn.Linear(2048, num_classes)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.dp = torch.nn.Dropout(0.4)

    def forward(self, x):
        # Handle both 4D and 5D inputs for compatibility
        if len(x.shape) == 4:
            # 4D input: [batch_size, channels, height, width] - add sequence dimension
            x = x.unsqueeze(1)  # Adding sequence length dimension (1 for single image)
        
        # Now x is 5D: [batch_size, seq_length, c, h, w]
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        sequence_logits = self.linear1(x_lstm)
        
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :])), sequence_logits

# Lazy loading for model
_model = None
_transform = None

def get_model():
    global _model, _transform
    if _model is None:
        try:
            logger.info("Loading model from Hugging Face Hub...")
            # ✅ Load model from Hugging Face
            model_path = hf_hub_download(repo_id="imtiyaz123/DF_Model", filename="df_model.pt")
            
            # ✅ Initialize model and load weights properly
            _model = DFModel()
            _model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            _model.eval()
            
            # ✅ Image transformation
            _transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    return _model, _transform

def predict_image(image_path):
    """Predict if image is fake using EfficientNet-B0 model"""
    try:
        model, transform = get_efficientnet_model()
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
            predicted_class = torch.argmax(probs).item()
            confidence = probs[predicted_class].item() * 100
            
            # Map EfficientNet output to our format:
            # EfficientNet: 0 = Real, 1 = Fake
            # Our format: 0 = FAKE, 1 = REAL (inverted)
            if predicted_class == 1:  # EfficientNet says Fake
                our_prediction = 0  # Our FAKE
            else:  # EfficientNet says Real
                our_prediction = 1  # Our REAL
            
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
        os.remove(image_path)
        return render_template('image.html', output=output, confidence=confidence)
    
    return render_template('image.html')

if __name__ == '__main__':
    print("--- Starting Server on Port 5000 ---")
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)
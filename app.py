from flask import Flask, render_template, url_for, redirect, flash, request, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
import os
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image
import numpy as np

from predictor.predictor import classify_doppler_video

app = Flask(__name__)

# Atur konfigurasi database sebelum menginisialisasi SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'whoknowsthesecretkey'

db = SQLAlchemy(app)  # Inisialisasi setelah konfigurasi
bcrypt = Bcrypt(app)

# Setup Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)

class RegisterForm(FlaskForm):
    username = StringField(validators=[
        InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
        InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Register')

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(username=username.data).first()
        if existing_user_username:
            raise ValidationError('That username already exists. Please choose a different one.')

class LoginForm(FlaskForm):
    username = StringField(validators=[
        InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
        InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    remember_me = BooleanField('Remember Me')

    submit = SubmitField('Login')

UPLOAD_FOLDER = 'static/uploads'  # folder untuk menyimpan video
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def frame_to_base64(frame):
    """Convert numpy frame (H, W, 3) to base64-encoded PNG."""
    image = Image.fromarray(frame.astype(np.uint8))
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Pastikan folder upload ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
@login_required  # User harus login dulu
def home():
    return render_template('home.html')

@app.route('/about')
@login_required  # User harus login dulu
def about():
    return render_template('about.html')

@app.route('/detection', methods=['GET', 'POST'])
@login_required
def detection():
    if request.method == 'POST':
        file = request.files['videofile']

        if file.filename == '' or not allowed_file(file.filename):
            flash('❌ File tidak valid.')
            return redirect(request.url)

        filename = secure_filename(file.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        # Simpan path di session agar bisa digunakan saat klik "deteksi"
        session['video_path'] = save_path

        video_url = url_for('static', filename='uploads/' + filename)
        return render_template('detection.html', video_url=video_url)

    return render_template('detection.html')

@app.route('/predict_video', methods=['POST'])
@login_required
def predict_video():
    video_path = session.get('video_path')
    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": "Video tidak ditemukan"}), 400

    try:
        result = classify_doppler_video(video_path)  # ← dari predictor.py

        # Convert frames to base64
        top_k_frames_b64 = [frame_to_base64(f) for f in result["top_k_frames"]]
        return jsonify({
            "predicted_class": result["predicted_class"],
            "probabilities": result["probabilities"],
            "attention_scores": result["attention_scores"],
            "top_k_frames": top_k_frames_b64
        })
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

@app.route('/reset_upload', methods=['GET'])
def reset_upload():
    session.pop('video_url', None)  # atau variable lain sesuai implementasimu
    return redirect(url_for('detection'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user, remember=form.remember_me.data)
                flash('Login successful!', 'success')
                return redirect(url_for('home'))  # Arahkan ke home setelah login
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html', form=form)

if __name__ == '__main__':
    app.run() # debug=True

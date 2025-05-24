import os
import cv2
import ffmpeg
import tempfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras import layers
from tensorflow.keras.applications.convnext import preprocess_input

# [START]========== VIDEO TO NUMPY ARRAY ==========
def downsample_fps_with_minterpolate(input_path, output_path, intermediate_fps=120, target_fps=30):
    (
        ffmpeg
        .input(input_path)
        .filter('fps', fps=int(intermediate_fps))
        .filter('minterpolate', fps=int(target_fps), mi_mode='mci', mc_mode='aobmc', vsbmc=1)
        .output(output_path, vcodec='libx264', crf=18, preset='slow')
        .run(overwrite_output=True)
    )

def upsample_fps_with_minterpolate(input_path, output_path, target_fps=30):
    (
        ffmpeg
        .input(input_path)
        .filter('minterpolate', fps=int(target_fps), mi_mode='mci', mc_mode='aobmc', me_mode='bilat',
                me='epzs', mb_size=16, vsbmc=1, scd='none')
        .output(output_path, vcodec='libx264', crf=18, preset='slow')
        .run(overwrite_output=True)
    )

def adaptive_resize_with_padding(frame, target_size=(224, 224), pad_color=(0, 0, 0)):
    h, w = frame.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    delta_w = target_size[0] - new_w
    delta_h = target_size[1] - new_h
    top, bottom = delta_h // 2, delta_h - delta_h // 2
    left, right = delta_w // 2, delta_w - delta_w // 2
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)

def enhance_contrast(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def enhance_doppler(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
    blue_mask = cv2.inRange(hsv, (110, 50, 50), (130, 255, 255))
    mask = cv2.bitwise_or(red_mask, blue_mask)
    hsv[..., 1] = cv2.add(hsv[..., 1], mask // 4)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def video_to_npy_frames(input_video_path, target_fps=30, target_size=(224, 224)):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError(f"❌ Gagal membuka video: {input_video_path}")
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if abs(original_fps - target_fps) > 1e-2:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_video_path = temp_file.name
        if original_fps > target_fps:
            downsample_fps_with_minterpolate(input_video_path, temp_video_path, target_fps=target_fps)
        else:
            upsample_fps_with_minterpolate(input_video_path, temp_video_path, target_fps=target_fps)
    else:
        temp_video_path = input_video_path

    cap = cv2.VideoCapture(temp_video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = adaptive_resize_with_padding(frame, target_size)
        frame = enhance_contrast(frame)
        frame = enhance_doppler(frame)
        frames.append(frame)
    cap.release()

    if temp_video_path != input_video_path and os.path.exists(temp_video_path):
        os.remove(temp_video_path)

    frames_np = np.array(frames, dtype=np.uint8)

    return frames_np

# [END]========== VIDEO TO NUMPY ARRAY ==========


# [START]========== LOAD ENCODER ==========
@tf.keras.utils.register_keras_serializable()
class CBAM(layers.Layer):
    def __init__(self, reduction_ratio=8, kernel_size=7, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

    def build(self, input_shape):
        channel_dim = input_shape[-1]

        # Channel Attention
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.max_pool = layers.GlobalMaxPooling2D()

        self.shared_mlp = tf.keras.Sequential([
            layers.Dense(channel_dim // self.reduction_ratio, activation='relu', use_bias=False),
            layers.Dense(channel_dim, use_bias=False)
        ])

        # Spatial Attention
        self.conv_spatial = layers.Conv2D(1, kernel_size=self.kernel_size, padding='same', activation='sigmoid', use_bias=False)

    def call(self, inputs):
        # Channel Attention
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)

        mlp_avg = self.shared_mlp(avg_pool)
        mlp_max = self.shared_mlp(max_pool)

        channel_attention = tf.nn.sigmoid(mlp_avg + mlp_max)
        x = inputs * channel_attention

        # Spatial Attention
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        spatial_attention = self.conv_spatial(concat)
        out = x * spatial_attention

        # Menyimpan perhatian untuk visualisasi
        self.channel_attention = channel_attention
        self.spatial_attention = spatial_attention

        return out

# ========== CUSTOM LOSS DAN FUNGSIONAL ========== #
@tf.keras.utils.register_keras_serializable()
def weighted_sum(inputs):
    values, weights = inputs
    return tf.reduce_sum(values * weights, axis=1)

# ========== ENCODER SETUP ========== #
def load_encoder(model_path, custom_objects=None):
    """
    Fungsi untuk memuat model encoder yang telah disimpan.
    """
    if custom_objects is None:
        custom_objects={'perceptual_loss': perceptual_loss, 'CBAM': CBAM}

    # Muat encoder model dari path yang diberikan
    encoder_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    print(f"✅ Encoder loaded from: {model_path}")
    return encoder_model

# Tetap di luar loss function ya
feature_extractor = load_encoder(
    "predictor/convnext_encoder.keras",
    custom_objects={
        'perceptual_loss': lambda y_true, y_pred: y_pred,  # Dummy karena loss tidak dibutuhkan saat inference
        'CBAM': CBAM
        }
)
feature_extractor.trainable = False

@tf.function
def extract_features(x):
    return feature_extractor(x)

@tf.keras.utils.register_keras_serializable()
def perceptual_loss(y_true, y_pred):
    y_true_features = extract_features(y_true)
    y_pred_features = extract_features(y_pred)
    perceptual = tf.reduce_mean(tf.square(y_true_features - y_pred_features))
    pixel = tf.reduce_mean(tf.square(y_true - y_pred))
    return 0.3 * pixel + 0.7 * perceptual

# Load pre-trained ConvNeXt encoder
encoder = load_encoder(
    "predictor/convnext_encoder.keras",
    custom_objects={'perceptual_loss': perceptual_loss, 'CBAM': CBAM}
)

# Freeze semua layer dulu
for layer in encoder.layers:
    layer.trainable = False

# Hanya unfreeze layer tertentu
for layer in encoder.layers:
    if any(stage in layer.name for stage in ["stage_3", "stage_2"]):
        layer.trainable = True

# [END]========== LOAD ENCODER ==========

# [START]========== EXTRACT LATENT SEQUENCE ==========
def extract_latent_sequence(video_array):
    """
    Ekstrak representasi laten dari ConvNeXt encoder untuk input video.

    Input:
        video_array: np.ndarray, shape (224, 224, 3), dtype float32

    Output:
        latent: np.ndarray, shape (T, D), e.g. (T, 37632) or (T, 768) if pooled
    """
    video_tensor = tf.convert_to_tensor(video_array, dtype=tf.float32)  # Convert NumPy to Tensor
    latent = encoder.predict(video_tensor, verbose=0)  # Output shape: (T, 7, 7, 768)
    # Step: Output encoder ConvNeXt → (T, 7, 7, 768)
    # Global Average Pooling → reduce to (T, 768)
    latent = tf.reduce_mean(latent, axis=[1, 2])  # Output shape: (T, 768)
    return latent.numpy()

# [END]========== EXTRACT LATENT SEQUENCE ==========

# [START]========== TOP K FRAME SELECTOR ==========
def select_top_k_activity_frames(features: np.ndarray, k: int = 12):
    """
    Select top-k frames with highest temporal change based on delta norm of features.

    Returns:
        top_k_indices: np.array of indices
        delta_norms: np.array of all delta norms (T,)
    """
    delta_norms = np.linalg.norm(np.diff(features, axis=0), axis=1)  # shape: (T-1,)
    delta_norms = np.insert(delta_norms, 0, 0.0)  # align index with frames → shape: (T,)
    top_k_indices = np.argsort(delta_norms)[-k:]  # ambil indeks dengan perubahan terbesar
    top_k_indices = np.sort(top_k_indices)       # urutkan agar kronologis
    return top_k_indices, delta_norms

# [END]========== TOP K FRAME SELECTOR ==========

# [START]========== KLASIFIKASI VIDEO ==========
def classify_doppler_video(video_path, encoder_path="predictor/convnext_encoder.keras", classifier_path="predictor/fold4.keras", k=12, class_names=['MR', 'MS', 'Normal']):
    # update_progress(0.1, f"Loading video: {video_path}")
    print(f"Loading video: {video_path}")
    frames = video_to_npy_frames(video_path)

    # update_progress(0.2, f"Preprocessing {len(frames)} frames...")
    print(f"Preprocessing frames ({len(frames)} total)...")
    original_video = frames.astype(np.float32)
    preprocessed_video = preprocess_input(original_video.copy())

    # update_progress(0.3, "Loading encoder...")
    print("Loading encoder...")
    encoder = load_encoder(
        encoder_path,
        custom_objects={
            'perceptual_loss': perceptual_loss,
            'CBAM': CBAM
        }
    )
    encoder.trainable = False

    # update_progress(0.4, "Extracting latent features...")
    print("Extracting latent features...")
    latent_seq = extract_latent_sequence(preprocessed_video)

    # update_progress(0.5, f"Selecting top-{k} informative frames...")
    print(f"Selecting top-{k} informative frames...")
    top_k_indices, _ = select_top_k_activity_frames(latent_seq, k=k)
    top_k_indices = np.array(top_k_indices)
    top_k_video = original_video[top_k_indices]  # shape: (K, H, W, 3)
    video_input = preprocess_input(top_k_video.astype(np.float32))
    video_input = np.expand_dims(video_input, axis=0)  # (1, K, H, W, 3)

    # update_progress(0.6, "Loading classifier model...")
    print("Loading classifier model...")
    model = load_model(
        classifier_path,
        custom_objects={
            "CBAM": CBAM,
            "perceptual_loss": perceptual_loss,
            "weighted_sum": weighted_sum
        },
        compile=False
    )

    # Predict class probabilities
    # update_progress(0.7, "Predicting class probabilities...")
    print("Predicting class probabilities...")
    pred = model.predict(video_input, verbose=0)
    pred_class_idx = np.argmax(pred[0])
    pred_class = class_names[pred_class_idx]
    pred_probs = dict(zip(class_names, pred[0].tolist()))

    # Extract attention weights (assume layer named "attention_weights")
    # update_progress(0.8, "Extracting attention weights...")
    print("Extracting attention weights...")
    attention_model = Model(inputs=model.input, outputs=model.get_layer("attention_weights").output)
    attention_weights = attention_model.predict(video_input, verbose=0)[0]  # shape: (K, 1)
    attention_scores = attention_weights.squeeze().tolist()  # (K,)

    # update_progress(0.9, "✅ Done.")
    print("Prediction result:")
    print("Predicted class:", pred_class)
    for cls, prob in pred_probs.items():
        print(f"  {cls}: {prob:.4f}")
    print("Attention scores:", attention_scores)

    return {
        "predicted_class": pred_class,
        "probabilities": pred_probs,
        "top_k_frames": top_k_video.astype(np.uint8),  # return for base64 encoding
        "attention_scores": attention_scores
    }

# [END]========== KLASIFIKASI VIDEO ==========

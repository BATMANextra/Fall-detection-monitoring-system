import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from sklearn.metrics import (recall_score, confusion_matrix, accuracy_score)
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from datetime import datetime

# ================================
# 1. Load preprocessed data (8 channels: ax ay az gx gy gz acc_mag gyro_mag)
# ================================
data    = np.load("sisfall_preprocessed.npz")
X_train = data['X_train']
y_train = data['y_train']
X_val   = data['X_val']
y_val   = data['y_val']
X_test  = data['X_test']
y_test  = data['y_test']

print(f"Train: {X_train.shape}, falls {np.sum(y_train)}")
print(f"Val  : {X_val.shape},   falls {np.sum(y_val)}")
print(f"Test : {X_test.shape},  falls {np.sum(y_test)}")

N_CHANNELS = X_train.shape[2]   # 8
WINDOW_SIZE = X_train.shape[1]    # 200 (50 Hz × 4 s)

# ================================
# 2. Generate synthetic static windows (ADL, very low noise)
# ================================
def generate_static_windows(X, y, num_static=500, noise_std=0.008, seed=42):
    """
    Generate synthetic static windows by adding very low noise to existing ADL windows.
    These simulate a perfectly stationary sensor (lying on table or standing still).
    """
    rng = np.random.default_rng(seed)
    adl_idx = np.where(y == 0)[0]
    if len(adl_idx) == 0:
        return np.empty((0, X.shape[1], X.shape[2]), dtype=np.float32), np.empty(0, dtype=np.int32)
    
    # Randomly select ADL windows (with replacement) to reach num_static
    chosen = rng.choice(adl_idx, size=num_static, replace=True)
    static_windows = X[chosen].copy()
    
    # Add tiny Gaussian noise to simulate sensor noise (but no real movement)
    static_windows += rng.normal(0, noise_std, static_windows.shape).astype(np.float32)
    
    # Clip to reasonable range
    static_windows = np.clip(static_windows, -3.0, 3.0)
    
    # Labels: all 0 (ADL)
    static_labels = np.zeros(num_static, dtype=np.int32)
    
    return static_windows, static_labels

# ================================
# 3. Combined augmentation: static windows + reduced fall augmentation
# ================================
def augment_with_static_and_falls(X, y, num_static=800, static_noise_std=0.008, 
                                  fall_aug_ratio=0.25, seed=42):
    """
    Augment training set with:
      - num_static synthetic static windows (ADL)
      - fall_aug_ratio * (#original falls) additional fall variants
    Returns shuffled combined dataset.
    """
    rng = np.random.default_rng(seed)
    
    # 1. Static windows
    X_static, y_static = generate_static_windows(X, y, num_static, static_noise_std, seed)
    
    # 2. Fall augmentation (reduced from 50% to 25%)
    fall_idx = np.where(y == 1)[0]
    n_aug = max(1, int(len(fall_idx) * fall_aug_ratio))
    chosen = rng.choice(fall_idx, size=n_aug, replace=False)
    X_fall_aug = X[chosen].copy()
    X_fall_aug += rng.normal(0, 0.025, X_fall_aug.shape).astype(np.float32)
    shifts = rng.integers(-20, 21, size=n_aug)
    for i, s in enumerate(shifts):
        if s != 0:
            X_fall_aug[i] = np.roll(X_fall_aug[i], s, axis=0)
    y_fall_aug = np.ones(n_aug, dtype=np.int32)
    
    # 3. Combine original + static + augmented falls
    X_combined = np.concatenate([X, X_static, X_fall_aug], axis=0)
    y_combined = np.concatenate([y, y_static, y_fall_aug], axis=0)
    
    # 4. Shuffle
    perm = rng.permutation(len(X_combined))
    return X_combined[perm].astype(np.float32), y_combined[perm]

# Apply augmentation
X_train_aug, y_train_aug = augment_with_static_and_falls(
    X_train, y_train,
    num_static=800,          # 800 synthetic static windows
    static_noise_std=0.008,
    fall_aug_ratio=0.25      # only 25% fall augmentation (was 50%)
)

print(f"\nAfter augmentation (static + 25% fall augmentation):")
print(f"  Train: {X_train_aug.shape}  |  "
      f"falls {int(np.sum(y_train_aug))}  ADL {len(y_train_aug)-int(np.sum(y_train_aug))}")

# ================================
# 4. Build model (unchanged)
# ================================
def build_model(input_shape):
    inp = layers.Input(shape=input_shape)

    x = layers.SeparableConv1D(24, 5, padding='same', use_bias=False,
                               depthwise_regularizer=regularizers.l2(1e-4),
                               pointwise_regularizer=regularizers.l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.SeparableConv1D(48, 5, padding='same', use_bias=False,
                               depthwise_regularizer=regularizers.l2(1e-4),
                               pointwise_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.SeparableConv1D(96, 5, padding='same', use_bias=False,
                               depthwise_regularizer=regularizers.l2(1e-4),
                               pointwise_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling1D()(x)

    x   = layers.Dropout(0.5)(x)
    x   = layers.Dense(32, activation='relu',
                       kernel_regularizer=regularizers.l2(1e-4))(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    return models.Model(inp, out, name="FallGuard_SepCNN_50Hz")

model = build_model(input_shape=(WINDOW_SIZE, N_CHANNELS))
model.summary()

# ================================
# 5. Focal loss with reduced alpha (0.55 instead of 0.72)
# ================================
def focal_loss(gamma=2.0, alpha=0.55):
    def loss_fn(y_true, y_pred):
        y_true  = tf.cast(y_true, tf.float32)
        eps     = tf.keras.backend.epsilon()
        pt      = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_t = y_true * alpha  + (1.0 - y_true) * (1.0 - alpha)
        loss    = -alpha_t * tf.pow(1.0 - pt, gamma) * tf.math.log(pt + eps)
        return tf.reduce_mean(loss)
    return loss_fn

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=focal_loss(gamma=2.0, alpha=0.55),   # reduced from 0.72
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
    ]
)

# ================================
# 6. Callbacks (unchanged)
# ================================
lr_scheduler = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6, verbose=1
)
checkpoint = callbacks.ModelCheckpoint(
    'best_fall_model.keras',
    monitor='val_auc', mode='max',
    save_best_only=True, verbose=1
)
early_stop = callbacks.EarlyStopping(
    monitor='val_auc', patience=18,
    restore_best_weights=True, mode='max', verbose=1
)

# ================================
# 7. Training
# ================================
print("\nTraining: SepCNN + focal(alpha=0.55) + static augmentation (800 windows) + 25% fall augmentation\n")
history = model.fit(
    X_train_aug, y_train_aug,
    validation_data=(X_val, y_val),
    epochs=120, batch_size=64,
    callbacks=[lr_scheduler, checkpoint, early_stop],
    verbose=1
)

# ================================
# 8. Add static windows to validation set for threshold search
# ================================
X_static_val, y_static_val = generate_static_windows(X_val, y_val, num_static=200, noise_std=0.008, seed=99)
X_val_combined = np.concatenate([X_val, X_static_val], axis=0)
y_val_combined = np.concatenate([y_val, y_static_val], axis=0)

print(f"\nValidation set augmented with 200 static windows: {X_val_combined.shape} total")

# ================================
# 9. Threshold selection on combined validation set
# ================================
y_val_prob = model.predict(X_val_combined, verbose=0).flatten()

def search_threshold(y_true, y_prob, recall_min=0.90, far_max=0.12):
    best = dict(thresh=0.5, acc=0.0, rec=0.0, far=1.0, found=False)
    for t in np.arange(0.10, 0.90, 0.01):
        t      = round(t, 2)
        y_pred = (y_prob > t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        far = fp / (tn + fp) if (tn + fp) > 0 else 1.0
        if rec >= recall_min and far <= far_max:
            acc = (tp + tn) / len(y_true)
            if acc > best['acc']:
                best = dict(thresh=t, acc=acc, rec=rec, far=far, found=True)
    return best

print("\n--- Threshold search on validation+static (recall >= 0.90, FAR <= 0.12) ---")
result = search_threshold(y_val_combined, y_val_prob, recall_min=0.90, far_max=0.12)

if not result['found']:
    print("  Relaxing FAR constraint to 0.15 …")
    result = search_threshold(y_val_combined, y_val_prob, recall_min=0.90, far_max=0.15)

if not result['found']:
    print("  Relaxing to recall >= 0.85, FAR <= 0.20 …")
    result = search_threshold(y_val_combined, y_val_prob, recall_min=0.85, far_max=0.20)

best_thresh = result['thresh']
print(f"\nSelected threshold = {best_thresh:.2f}  |  "
      f"Val recall = {result['rec']:.4f}  FAR = {result['far']:.4f}  "
      f"accuracy = {result['acc']:.4f}")

# ================================
# 10. Test evaluation (unchanged)
# ================================
y_test_prob     = model.predict(X_test, verbose=0).flatten()
y_pred_raw      = (y_test_prob > best_thresh).astype(int)
y_pred_smoothed = medfilt(y_pred_raw.astype(float), kernel_size=5).astype(int)

def evaluate(y_true, y_pred, label):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    far  = fp / (tn + fp) if (tn + fp) > 0 else 1.0
    rec  = tp / (tp + fn)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc  = (tp + tn) / (tp + tn + fp + fn)
    print(f"\n{label}")
    print(f"  Accuracy : {acc:.4f}  |  Recall : {rec:.4f}  |  FAR : {far*100:.1f}%")
    print(f"  Precision: {prec:.4f}  |  F1     : {f1:.4f}")
    print(f"  Confusion matrix: ADL→{tn} {fp}, Fall→{fn} {tp}")
    return acc, rec, far

print("\n" + "=" * 60)
evaluate(y_test, y_pred_raw,      f"RAW (threshold={best_thresh:.2f})")
acc_s, rec_s, far_s = evaluate(y_test, y_pred_smoothed,
                                f"SMOOTHED (threshold={best_thresh:.2f}, median k=5)")
print("=" * 60)

print(f"\n✅ Final test accuracy : {acc_s*100:.1f}%  (target >90%)")
print(f"   Recall             : {rec_s*100:.1f}%  (target >=90%)")
print(f"   FAR                : {far_s*100:.1f}%  (target <=10%)")

if acc_s >= 0.90 and rec_s >= 0.90 and far_s <= 0.10:
    print("SUCCESS - Model meets all deployment targets.")
else:
    missing = []
    if acc_s < 0.90: missing.append(f"accuracy {acc_s*100:.1f}% < 90%")
    if rec_s < 0.90: missing.append(f"recall {rec_s*100:.1f}% < 90%")
    if far_s > 0.10: missing.append(f"FAR {far_s*100:.1f}% > 10%")
    print("Missing:", ", ".join(missing))

# ================================
# 11. Full threshold sweep (diagnostic)
# ================================
print("\n--- Full threshold sweep (validation+static) ---")
print(f"{'Thresh':>7}  {'Recall':>7}  {'FAR':>7}  {'Accuracy':>9}  {'F1':>7}")
for t in np.arange(0.20, 0.85, 0.05):
    t      = round(t, 2)
    yp     = (y_val_prob > t).astype(int)
    tn, fp, fn, tp_ = confusion_matrix(y_val_combined, yp).ravel()
    rec_   = tp_ / (tp_ + fn) if (tp_ + fn) > 0 else 0
    far_   = fp  / (tn + fp)  if (tn + fp)  > 0 else 1
    acc_   = (tp_ + tn) / len(y_val_combined)
    prec_  = tp_ / (tp_ + fp) if (tp_ + fp) > 0 else 0
    f1_    = 2 * prec_ * rec_ / (prec_ + rec_) if (prec_ + rec_) > 0 else 0
    flag   = " <--" if rec_ >= 0.90 and far_ <= 0.12 else ""
    print(f"  {t:5.2f}   {rec_:6.4f}   {far_:6.4f}   {acc_:8.4f}   {f1_:6.4f}{flag}")

# ================================
# 12. Plot training curves
# ================================
ep = range(1, len(history.history['loss']) + 1)
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
axes[0].plot(ep, history.history['loss'],     label='train')
axes[0].plot(ep, history.history['val_loss'], label='val')
axes[0].set_title('Focal Loss'); axes[0].legend()
axes[1].plot(ep, history.history['recall'],     label='train')
axes[1].plot(ep, history.history['val_recall'], label='val')
axes[1].axhline(0.90, color='r', linestyle='--', label='target')
axes[1].set_title('Recall'); axes[1].legend()
axes[2].plot(ep, history.history['precision'],     label='train')
axes[2].plot(ep, history.history['val_precision'], label='val')
axes[2].set_title('Precision'); axes[2].legend()
axes[3].plot(ep, history.history['auc'],     label='train')
axes[3].plot(ep, history.history['val_auc'], label='val')
axes[3].set_title('AUC'); axes[3].legend()
plt.tight_layout()
plt.savefig('training_history.png', dpi=120)
plt.show()

# ================================
# 13. TFLite INT8 conversion
# ================================
def representative_dataset():
    rng     = np.random.default_rng(seed=0)
    indices = rng.choice(len(X_train), size=300, replace=False)
    for idx in indices:
        yield [np.expand_dims(X_train[idx].astype(np.float32), axis=0)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations                   = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset         = representative_dataset
converter.target_spec.supported_ops      = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type           = tf.int8
converter.inference_output_type          = tf.int8
tflite_model = converter.convert()

with open("fall_detector_int8.tflite", "wb") as f:
    f.write(tflite_model)
print(f"\nTFLite model size: {len(tflite_model)/1024:.1f} KB")

interp = tf.lite.Interpreter(model_content=tflite_model)
interp.allocate_tensors()
in_det  = interp.get_input_details()[0]
out_det = interp.get_output_details()[0]
input_scale  = float(in_det ['quantization'][0])
input_zero   = int  (in_det ['quantization'][1])
output_scale = float(out_det['quantization'][0])
output_zero  = int  (out_det['quantization'][1])

# ================================
# 14. AUTO-GENERATE firmware_constants.h
# ================================
means  = data['scaler_mean'].tolist()
scales = data['scaler_scale'].tolist()

generated_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

fw_header = f"""// ================================================================
// AUTO-GENERATED by train03_fixed.py — DO NOT EDIT MANUALLY.
// Generated : {generated_ts}
// Regenerate: re-run train03_fixed.py after any model change.
// ================================================================
#ifndef FIRMWARE_CONSTANTS_H
#define FIRMWARE_CONSTANTS_H

// --- TFLite INT8 quantization parameters ---
constexpr float INPUT_SCALE    = {input_scale:.8f}f;
constexpr int   INPUT_ZERO     = {input_zero};
constexpr float OUTPUT_SCALE   = {output_scale:.8f}f;
constexpr int   OUTPUT_ZERO    = {output_zero};

// --- Detection threshold (from dual-constraint val+static search) ---
constexpr float FALL_THRESHOLD = {best_thresh:.2f}f;

// --- Channel / window constants ---
constexpr int   N_CHANNELS     = {N_CHANNELS};
constexpr int   WINDOW_SIZE    = {WINDOW_SIZE}; // {WINDOW_SIZE} samples @ 50 Hz = 4 seconds

// --- StandardScaler parameters from pre01.py ---
constexpr float MEANS [N_CHANNELS] = {{{', '.join(f'{v:.6f}f' for v in means)}}};
constexpr float SCALES[N_CHANNELS] = {{{', '.join(f'{v:.6f}f' for v in scales)}}};

#endif  // FIRMWARE_CONSTANTS_H
"""

with open("firmware_constants.h", "w", encoding='utf-8') as f:
    f.write(fw_header)

print("\n--- ESP32 constants ---")
print(f"const float INPUT_SCALE    = {input_scale:.8f}f;")
print(f"const int   INPUT_ZERO     = {input_zero};")
print(f"const float OUTPUT_SCALE   = {output_scale:.8f}f;")
print(f"const int   OUTPUT_ZERO    = {output_zero};")
print(f"const float FALL_THRESHOLD = {best_thresh:.2f}f;")
print(f"\nconst float MEANS [{N_CHANNELS}] = {{{', '.join(f'{v:.6f}f' for v in means)}}};")
print(f"const float SCALES[{N_CHANNELS}] = {{{', '.join(f'{v:.6f}f' for v in scales)}}};")
print("\n✅ firmware_constants.h written to disk.")

# ================================
# 15. AUTO-GENERATE fall_model.h
# ================================
hex_bytes = ', '.join(f'0x{b:02x}' for b in tflite_model)
with open("fall_model.h", "w") as f:
    f.write("// Auto-generated\n#ifndef FALL_MODEL_H\n#define FALL_MODEL_H\n\n")
    f.write(f"const unsigned char fall_detector_int8_tflite[] = {{{hex_bytes}}};\n")
    f.write(f"const unsigned int  fall_detector_int8_tflite_len = {len(tflite_model)};\n\n")
    f.write("#endif\n")

print("✅ fall_model.h written to disk.")
print("✅ fall_detector_int8.tflite written to disk.")
print("\nDone.")
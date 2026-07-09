import numpy as np
import os
import glob
from scipy.signal import decimate
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ================================
# Configuration
# ================================
DATA_ROOT = r"C:\Users\BATMAN\Desktop\Cnn\SisFall_dataset"

# --- Downsampling settings ---
ORIGINAL_FS = 200   # SisFall native sampling rate (Hz)
TARGET_FS   = 50    # ESP32 / MPU6050 friendly rate (Hz)
DOWNSAMPLE  = ORIGINAL_FS // TARGET_FS   # 4

# --- Windowing settings (4 seconds at 50 Hz) ---
WINDOW_SEC  = 4
WINDOW_SIZE = TARGET_FS * WINDOW_SEC    # 50 * 4 = 200 samples
STEP        = WINDOW_SIZE // 2          # 50% overlap = 100

N_CHANNELS = 8      # ax, ay, az, gx, gy, gz, acc_mag, gyro_mag

# Subjects held out for test
TEST_SUBJECTS = [3, 8, 13, 18, 23, 25, 30, 35]

# ================================
# Subject ID parser
# ================================
def parse_subject_id(base_name: str) -> int:
    parts = os.path.splitext(base_name)[0].split("_")
    sub = parts[1]          # e.g. 'SA03' or 'SE02'
    kind = sub[:2]          # 'SA' or 'SE'
    num = int(sub[2:])
    return num if kind == "SA" else 23 + num

# ================================
# Custom parser for SisFall .txt files
# ================================
def load_txt_file(file_path):
    data_list = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.endswith(';'):
                line = line[:-1]
            line = line.replace(',', ' ')
            tokens = line.split()
            try:
                nums = [float(x) for x in tokens]
            except ValueError:
                continue
            if len(nums) >= 7:
                data_list.append(nums)
    if not data_list:
        raise ValueError(f"No valid data rows found in {file_path}")
    data = np.array(data_list, dtype=np.float32)

    ACC_SCALE  = 256.0    # LSB/g  (ADXL345 ±16g)
    GYRO_SCALE = 14.375   # LSB/(°/s)
    acc  = data[:, 1:4] / ACC_SCALE
    gyro = data[:, 4:7] / GYRO_SCALE
    return acc, gyro

# ================================
# Anti-aliasing + decimation
# ================================
def downsample_signals(acc, gyro, factor=DOWNSAMPLE):
    """
    Downsample accelerometer and gyroscope signals from ORIGINAL_FS to TARGET_FS.
    Uses scipy.signal.decimate with an FIR anti-aliasing filter and zero-phase
    filtering to avoid shifting the impact peak in time.
    """
    # decimate works on 1-D arrays; process each axis separately
    acc_ds  = np.stack([decimate(acc[:, i],  factor, ftype='fir', zero_phase=True)
                        for i in range(3)], axis=1)
    gyro_ds = np.stack([decimate(gyro[:, i], factor, ftype='fir', zero_phase=True)
                        for i in range(3)], axis=1)
    return acc_ds.astype(np.float32), gyro_ds.astype(np.float32)

# ================================
# Sliding window segmentation
# Channels: ax ay az gx gy gz acc_mag gyro_mag  (8 total)
# acc_mag / gyro_mag = vector magnitude — the strongest fall discriminator
# ================================
def segment_signals(acc, gyro, label, subject_id, window_size, step):
    # Pre-compute per-sample magnitudes (shape: [T, 1])
    acc_mag  = np.linalg.norm(acc,  axis=1, keepdims=True)  # resultant acceleration
    gyro_mag = np.linalg.norm(gyro, axis=1, keepdims=True)  # resultant angular rate

    X, y, subjects = [], [], []
    total_len = acc.shape[0]
    for start in range(0, total_len - window_size, step):
        end = start + window_size
        window = np.hstack([
            acc [start:end],      # channels 0-2
            gyro[start:end],      # channels 3-5
            acc_mag [start:end],  # channel 6  ← peak here separates falls from ADLs
            gyro_mag[start:end],  # channel 7
        ])
        X.append(window)
        y.append(label)
        subjects.append(subject_id)
    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.int32),
            np.array(subjects, dtype=np.int32))

# ================================
# Load all files
# ================================
all_X, all_y, all_subjects = [], [], []
txt_files = (glob.glob(os.path.join(DATA_ROOT, "SA*", "*.txt")) +
             glob.glob(os.path.join(DATA_ROOT, "SE*", "*.txt")))

if not txt_files:
    raise FileNotFoundError(f"No .txt files found under {DATA_ROOT}")

print(f"Found {len(txt_files)} .txt files")
print(f"Downsampling: {ORIGINAL_FS} Hz → {TARGET_FS} Hz (factor={DOWNSAMPLE})")
print(f"Window: {WINDOW_SEC}s = {WINDOW_SIZE} samples @ {TARGET_FS} Hz")
print(f"Step: {STEP} samples ({100*STEP/WINDOW_SIZE:.0f}% overlap)\n")

skipped = 0
for file_path in sorted(txt_files):
    base = os.path.basename(file_path)
    if base.startswith('F'):
        label = 1
    elif base.startswith('D'):
        label = 0
    else:
        print(f"  [SKIP] Unknown prefix: {base}")
        skipped += 1
        continue

    try:
        subject_id = parse_subject_id(base)
        acc, gyro  = load_txt_file(file_path)

        # --- DOWNSAMPLE HERE ---
        acc, gyro = downsample_signals(acc, gyro)

        X_seg, y_seg, s_seg = segment_signals(acc, gyro, label, subject_id,
                                              WINDOW_SIZE, STEP)
        if len(X_seg) > 0:
            all_X.append(X_seg)
            all_y.append(y_seg)
            all_subjects.append(s_seg)
            tag = "FALL" if label == 1 else "ADL "
            print(f"  [{tag}] Sub{subject_id:02d}  {base:30s} → {X_seg.shape[0]:4d} windows")
        else:
            print(f"  [WARN] {base} produced 0 windows")
            skipped += 1
    except Exception as e:
        print(f"  [ERR]  {file_path}: {e}")
        skipped += 1

print(f"\nSkipped / errored: {skipped} file(s)")

if not all_X:
    raise RuntimeError("No usable windows produced.")

X        = np.concatenate(all_X,        axis=0)
y        = np.concatenate(all_y,        axis=0)
subjects = np.concatenate(all_subjects, axis=0)

n_falls = int(np.sum(y))
n_adl   = len(y) - n_falls
print(f"\nTotal windows : {len(y):>7,}")
print(f"  Falls  (1)  : {n_falls:>7,} ({100*n_falls/len(y):.1f}%)")
print(f"  ADL    (0)  : {n_adl:>7,} ({100*n_adl/len(y):.1f}%)")
print(f"  Channels    : {X.shape[2]}  (ax ay az gx gy gz acc_mag gyro_mag)")
print(f"  Window shape: {X.shape[1]} samples × {X.shape[2]} channels")

# ================================
# Subject-aware train / test split
# ================================
test_mask  = np.isin(subjects, TEST_SUBJECTS)
train_mask = ~test_mask

X_train_raw    = X[train_mask]
y_train_raw    = y[train_mask]
X_test_raw     = X[test_mask]
y_test_raw     = y[test_mask]
subjects_train = subjects[train_mask]
subjects_test  = subjects[test_mask]

print(f"\nTrain (raw) : {X_train_raw.shape}  |  falls {np.sum(y_train_raw)}")
print(f"Test  (raw) : {X_test_raw.shape}   |  falls {np.sum(y_test_raw)}")

# ================================
# Normalize using training statistics only
# Scaler operates on all 8 channels
# ================================
scaler = StandardScaler()
X_train_flat = X_train_raw.reshape(-1, N_CHANNELS)
scaler.fit(X_train_flat)

X_train_norm = scaler.transform(X_train_flat).reshape(X_train_raw.shape)
X_test_norm  = scaler.transform(X_test_raw.reshape(-1, N_CHANNELS)).reshape(X_test_raw.shape)

print(f"\nScaler means  : {np.round(scaler.mean_,  4).tolist()}")
print(f"Scaler scales : {np.round(scaler.scale_, 4).tolist()}")

# ================================
# 80/20 stratified split on training set → train / validation
# ================================
X_train, X_val, y_train, y_val = train_test_split(
    X_train_norm, y_train_raw,
    test_size=0.2,
    stratify=y_train_raw,
    random_state=42
)

print(f"\nAfter 80/20 split:")
print(f"  Train : {X_train.shape}, falls {np.sum(y_train)}")
print(f"  Val   : {X_val.shape},   falls {np.sum(y_val)}")

# ================================
# Save all required arrays
# ================================
OUTPUT_FILE = "sisfall_preprocessed.npz"
np.savez(OUTPUT_FILE,
         X_train=X_train,
         y_train=y_train,
         X_val=X_val,
         y_val=y_val,
         X_test=X_test_norm,
         y_test=y_test_raw,
         subjects_train=subjects_train,
         subjects_test=subjects_test,
         scaler_mean=scaler.mean_,
         scaler_scale=scaler.scale_)

print(f"\nPreprocessing complete → saved to '{OUTPUT_FILE}'")
print("Fields: X_train, y_train, X_val, y_val, X_test, y_test,")
print("        subjects_train, subjects_test, scaler_mean, scaler_scale")
print(f"Channel layout: ax ay az gx gy gz acc_mag gyro_mag  ({N_CHANNELS} channels)")
print(f"Each window: {WINDOW_SIZE} samples @ {TARGET_FS} Hz = {WINDOW_SEC}s")

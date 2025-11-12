import os
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

# === Path settings ===
INPUT_DIR = "/Users/kemingliu/Desktop/CoronaHack-Respiratory-Sound-Dataset/output/processed_audio"
OUTPUT_DIR = "/Users/kemingliu/Desktop/CoronaHack-Respiratory-Sound-Dataset/output/filtered_audio"
LOG_CSV = "/Users/kemingliu/Desktop/CoronaHack-Respiratory-Sound-Dataset/output/filter_log.csv"

# === Thresholds for filtering ===
MIN_DURATION = 0.5     # seconds
MIN_RMS = 0.01         # minimum loudness threshold
MIN_SNR = 3.0          # signal-to-noise ratio (estimated)
MAX_CLIP_FRAC = 0.01   # fraction of clipped samples allowed

# === Function to compute audio quality metrics ===
def compute_audio_metrics(y, sr):
    duration = len(y) / sr
    rms = np.mean(librosa.feature.rms(y=y))
    rms_all = librosa.feature.rms(y=y)[0]
    snr_est = np.max(rms_all) / (np.mean(rms_all) + 1e-6)  # approximate SNR
    clip_frac = np.sum(np.abs(y) > 0.99) / len(y)
    return duration, rms, snr_est, clip_frac


# === Main filtering function ===
def filter_audio_files(input_root, output_root, log_csv):
    all_files = []
    for root, _, files in os.walk(input_root):
        for file in files:
            if file.endswith(".wav"):
                all_files.append(os.path.join(root, file))

    print(f"🔍 Found {len(all_files)} audio files under {input_root}")

    logs = []
    kept, skipped = 0, 0

    for file_path in tqdm(all_files, desc="Evaluating audio quality"):
        try:
            y, sr = librosa.load(file_path, sr=None)
            if len(y) == 0:
                skipped += 1
                logs.append([file_path, 0, 0, 0, 0, "empty"])
                continue

            # Compute metrics
            duration, rms, snr_est, clip_frac = compute_audio_metrics(y, sr)

            # Determine if file passes all conditions
            pass_filters = (
                duration >= MIN_DURATION and
                rms >= MIN_RMS and
                snr_est >= MIN_SNR and
                clip_frac <= MAX_CLIP_FRAC
            )

            status = "kept" if pass_filters else "skipped"
            logs.append([file_path, duration, rms, snr_est, clip_frac, status])

            if pass_filters:
                rel_path = os.path.relpath(file_path, input_root)
                out_path = os.path.join(output_root, rel_path)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                sf.write(out_path, y, sr)
                kept += 1
            else:
                skipped += 1

        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            logs.append([file_path, None, None, None, None, "error"])
            skipped += 1

    # Save log
    df = pd.DataFrame(logs, columns=["file", "duration", "rms", "snr_est", "clip_frac", "status"])
    df.to_csv(log_csv, index=False)

    print(f"\n✅ Kept {kept} files")
    print(f"⚠️ Skipped {skipped} files (failed quality checks or error)")
    print(f"📘 Detailed log saved to {log_csv}")


if __name__ == "__main__":
    filter_audio_files(INPUT_DIR, OUTPUT_DIR, LOG_CSV)

import os
import librosa
import soundfile as sf
import numpy as np
import noisereduce as nr
from scipy.signal import resample_poly

INPUT_ROOT = "data"
OUTPUT_ROOT = "output/processed_audio"

TARGET_SR = 16000
SILENCE_TOP_DB = 20


def resample_audio(y, orig_sr, target_sr):
    if orig_sr == target_sr:
        return y
    # polyphase resampling
    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    return resample_poly(y, up, down)


def process_audio(input_path, output_path, target_sr=TARGET_SR):
    try:
        # Load
        y, orig_sr = librosa.load(input_path, sr=None)

        # Resample
        y = resample_audio(y, orig_sr, target_sr)

        # Noise Reduction
        y = nr.reduce_noise(y=y, sr=target_sr)

        # Trim Silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=SILENCE_TOP_DB)
        if len(y_trimmed) < target_sr * 0.1:
            y_trimmed = y

        # Save
        sf.write(output_path, y_trimmed, target_sr)

    except Exception as e:
        print(f"Error processing {input_path}: {e}")


def batch_process():
    for root, dirs, files in os.walk(INPUT_ROOT):
        for file in files:
            if file.lower().endswith(".wav"):
                in_path = os.path.join(root, file)

                # mirror folder structure
                rel = os.path.relpath(root, INPUT_ROOT)
                out_dir = os.path.join(OUTPUT_ROOT, rel)
                os.makedirs(out_dir, exist_ok=True)

                out_path = os.path.join(out_dir, file)

                process_audio(in_path, out_path)


if __name__ == "__main__":
    print("Starting audio cleaning ...")
    batch_process()
    print("Done!")

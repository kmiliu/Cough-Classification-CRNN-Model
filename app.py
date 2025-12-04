import streamlit as st
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

# Paths
BASE = Path(__file__).resolve().parent
TRAIN_CSV = BASE / "output" / "model_input_clean_train.csv"
MODEL_PATH_DEFAULT = BASE / "output" / "crnn_model_results_7" / "crnn_cough.pt"

st.set_page_config(page_title="CRNN V7 — Respiratory Sound Demo", layout="centered")


@st.cache_data
def load_train_stats():
    # feature columns expected in the CSV (same order used in training)
    feat_cols = [f"mfcc_{i+1}" for i in range(13)] + [f"chroma_{i+1}" for i in range(12)] + [
        "spec_centroid",
        "spec_bw",
        "rolloff",
        "zcr",
    ]
    if not TRAIN_CSV.exists():
        return feat_cols, None, None
    df = pd.read_csv(TRAIN_CSV, usecols=feat_cols)
    # imputer -> median, scaler -> mean/std
    med = df.median()
    mean = df.mean()
    std = df.std(ddof=0).replace(0, 1.0)
    return feat_cols, med.to_dict(), (mean.to_dict(), std.to_dict())


def extract_features_from_audio_file(path, sr=22050):
    try:
        y, sr = librosa.load(path, sr=sr)
        if len(y) < sr * 0.5:
            return None
        y = librosa.util.normalize(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        feats = []
        feats += [np.mean(mfcc[i]) for i in range(13)]
        feats += [np.mean(chroma[i]) for i in range(12)]
        feats += [spec_centroid, spec_bw, rolloff, zcr]
        return np.array(feats, dtype=float)
    except Exception as e:
        st.error(f"Feature extraction error: {e}")
        return None


class CRNNCoughClassifier(nn.Module):
    def __init__(self, input_length):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
        )
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True,
        )
        self.attn = nn.Linear(256, 1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        feats = self.conv(x)
        feats = feats.permute(0, 2, 1)
        outputs, _ = self.gru(feats)
        weights = torch.softmax(self.attn(outputs).squeeze(-1), dim=1).unsqueeze(-1)
        context = torch.sum(outputs * weights, dim=1)
        return self.classifier(context).squeeze(1)


@st.cache_resource
def load_model_from_path(model_path: Path, input_len: int):
    if not model_path.exists():
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNNCoughClassifier(input_len)
    state = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(state)
    except Exception:
        # maybe the state dict is nested or saved differently
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            # try loading directly
            model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def find_latest_model_in_output(pattern="crnn_cough.pt"):
    """Search `output/` for model files matching pattern and return the newest Path or None."""
    out_dir = BASE / "output"
    if not out_dir.exists():
        return None
    candidates = list(out_dir.rglob(pattern))
    if not candidates:
        # also try any .pt files under output
        candidates = list(out_dir.rglob("*.pt"))
    if not candidates:
        return None
    # pick newest by mtime
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def main():
    st.title("CRNN V7 — Respiratory Sound Demo")
    st.write("Upload a cough/breathing audio file (wav/mp3/webm). The app extracts features and runs the trained V7 model.")

    feat_cols, med, mean_std = load_train_stats()
    if med is None or mean_std is None:
        st.warning("Training statistics (imputer/scaler) not found: please ensure `output/model_input_clean_train.csv` exists in the repository for exact preprocessing.")

    uploaded = st.file_uploader("Audio file", type=["wav", "mp3", "webm"])

    model_file = st.file_uploader("(Optional) Upload PyTorch model file (.pt) to use for inference", type=["pt", "pth", "bin"])

    model_path = MODEL_PATH_DEFAULT
    if model_file is not None:
        # save temporarily and load
        tmp = BASE / "tmp_model_upload.pth"
        with open(tmp, "wb") as f:
            f.write(model_file.read())
        model_path = tmp

    # If user did not upload a model, try to auto-detect the newest model in output/
    if model_file is None:
        detected = find_latest_model_in_output()
        if detected is not None:
            model_path = detected
            st.info(f"Auto-detected model: `{model_path}`")

    model = None
    input_len = len(feat_cols)
    model = load_model_from_path(model_path, input_len)

    if model is None:
        st.info(f"Model not found at `{model_path}`. You can upload a `.pt`/.pth file above or place `crnn_cough.pt` under `output/crnn_model_results_7/`.")

    if uploaded is not None:
        # save uploaded to a temp file
        tmp_audio = BASE / "tmp_audio.wav"
        with open(tmp_audio, "wb") as f:
            f.write(uploaded.read())
        feats = extract_features_from_audio_file(tmp_audio)
        if feats is None:
            st.error("Unable to extract features from the uploaded file. Try a longer/wav file.")
            return

        # apply imputer (median) and scaler (mean/std) if available
        x = feats.copy()
        if med is not None and mean_std is not None:
            med_d = med
            mean_d, std_d = mean_std
            # impute
            for i, col in enumerate(feat_cols):
                if np.isnan(x[i]):
                    x[i] = med_d.get(col, 0.0)
            # scale
            for i, col in enumerate(feat_cols):
                x[i] = (x[i] - mean_d.get(col, 0.0)) / std_d.get(col, 1.0)
        else:
            # basic impute with zero
            x = np.nan_to_num(x, nan=0.0)

        # prepare tensor shaped (1, 1, features)
        xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        xt = xt.to(device)

        if model is None:
            st.warning("No model loaded — inference unavailable. Upload a model file or place one at the default path.")
            st.write("Extracted features (scaled):")
            df = pd.DataFrame([x], columns=feat_cols)
            st.dataframe(df)
            return

        with torch.no_grad():
            logits = model(xt)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            prob = float(probs[0])
            st.metric("Predicted probability (COVID) ", f"{prob:.3f}")
            st.progress(min(max(prob, 0.0), 1.0))
            st.write("Thresholded prediction (0.5):", "Positive" if prob >= 0.5 else "Negative")

        st.write("---")
        st.write("You can download the preprocessed features as CSV:")
        out_df = pd.DataFrame([x], columns=feat_cols)
        st.download_button("Download features CSV", out_df.to_csv(index=False), file_name="sample_features.csv", mime="text/csv")


if __name__ == "__main__":
    main()

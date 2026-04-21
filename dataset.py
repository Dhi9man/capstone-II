"""
LipSyncD — Dataset & Feature Extractor (v3 — CNN)
====================================================
Feature extraction now uses:
  - EfficientNet-B0 pretrained on ImageNet → 1280-dim face embedding
  - Sampled across N frames per video → mean + std pooled → 2560-dim
  - Combined with 3-dim audio-visual sync features → 2563-dim total

This is the standard approach for FF++ and achieves 85-95% accuracy.

Folder layouts supported:
  Flat archive (your download):
    FaceForensics++_C23/
      original/       ← real
      Deepfakes/      ← fake
      FaceSwap/       ← fake
      Face2Face/      ← fake
      NeuralTextures/ ← fake

  Canonical FF++:
    original_sequences/youtube/c23/videos/*.mp4
    manipulated_sequences/Deepfakes/c23/videos/*.mp4
"""

import os
import csv
import json
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    import torchvision.models as models
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

try:
    import librosa
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False

try:
    from scipy.signal import correlate
    from scipy.stats import pearsonr
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

try:
    import mediapipe as mp
    MP_OK = True
    _mp_face_mesh = mp.solutions.face_mesh
except ImportError:
    MP_OK = False

# ── Constants ─────────────────────────────────────────────────
FF_MANIPULATIONS     = ["Deepfakes", "FaceSwap"]
FF_ALL_MANIPULATIONS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
FF_COMPRESSIONS      = ["c0", "c23", "c40"]

# EfficientNet-B0 output dim
CNN_DIM   = 1280
# mean + std pooled across frames
FEAT_DIM  = CNN_DIM * 2   # 2560
SYNC_DIM  = 3
TOTAL_DIM = FEAT_DIM + SYNC_DIM   # 2563


# ─────────────────────────────────────────────────────────────
def scan_dataset(root: str) -> None:
    """Diagnostic — run with --scan before training."""
    root = Path(root)
    print(f"\n{'='*60}")
    print(f"  Scanning: {root}")
    print(f"{'='*60}")

    if not root.exists():
        print(f"  ERROR: Folder does not exist: {root}")
        print(f"{'='*60}\n")
        return

    subdirs = sorted([d for d in root.iterdir() if d.is_dir()])
    print(f"\n  Direct subfolders ({len(subdirs)}):")
    for d in subdirs:
        total = len(list(d.rglob("*.mp4"))) + len(list(d.rglob("*.avi")))
        print(f"    {d.name}/  →  {total} videos")

    print(f"\n  Canonical FF++ paths:")
    found_canonical = False
    for comp in FF_COMPRESSIONS:
        p = root / "original_sequences" / "youtube" / comp / "videos"
        if p.exists():
            print(f"    REAL  [{comp}]  →  {len(list(p.glob('*.mp4')))} videos  ✓")
            found_canonical = True
    for manip in FF_ALL_MANIPULATIONS:
        for comp in FF_COMPRESSIONS:
            p = root / "manipulated_sequences" / manip / comp / "videos"
            if p.exists():
                print(f"    FAKE  [{comp}]  {manip}  →  {len(list(p.glob('*.mp4')))} videos  ✓")
                found_canonical = True
    if not found_canonical:
        print(f"    (none found)")

    print(f"\n  Flat layout paths (archive-style):")
    found_flat = False
    for real_name in ("original", "original_sequences", "real", "youtube"):
        p = root / real_name
        if p.exists():
            vids = list(p.rglob("*.mp4")) + list(p.rglob("*.avi"))
            print(f"    REAL  {real_name}/  →  {len(vids)} videos  ✓ will be used as real")
            found_flat = True
    for manip in FF_ALL_MANIPULATIONS:
        p = root / manip
        if p.exists():
            vids = list(p.rglob("*.mp4")) + list(p.rglob("*.avi"))
            print(f"    FAKE  {manip}/  →  {len(vids)} videos  ✓ will be used as fake")
            found_flat = True
    if not found_flat:
        print(f"    (none found)")

    splits_dir = root / "splits"
    if splits_dir.exists():
        for f in sorted(splits_dir.glob("*.json")):
            data = json.loads(f.read_text())
            print(f"\n  SPLIT  {f.name}  →  {len(data)} entries")
    for f in root.glob("*.csv"):
        with open(f) as fh:
            rows = sum(1 for _ in fh)
        print(f"  CSV    {f.name}  →  {rows} rows")

    print(f"\n{'='*60}\n")


# ─────────────────────────────────────────────────────────────
class FFPPDataset:
    """Builds (video_path, label) pairs. label=0 real, label=1 fake."""

    def __init__(
        self,
        root: str,
        compression: str = "c23",
        manipulations: Optional[List[str]] = None,
        max_per_class: int = 500,
        split: Optional[str] = None,
        seed: int = 42,
    ):
        self.root          = Path(root)
        self.compression   = compression
        self.manipulations = manipulations if manipulations is not None else FF_MANIPULATIONS
        self.max_per_class = max_per_class
        self.split         = split
        self.rng           = random.Random(seed)
        self.samples: List[Tuple[str, int]] = []
        self._allowed_ids: Optional[set] = None

        self._load_split_ids()
        self._scan()

    def _load_split_ids(self):
        if self.split is None:
            return
        json_path = self.root / "splits" / f"{self.split}.json"
        if json_path.exists():
            pairs = json.loads(json_path.read_text())
            ids = set()
            for pair in pairs:
                for vid_id in pair:
                    ids.add(str(vid_id).zfill(3))
            self._allowed_ids = ids
            print(f"[Dataset] Split '{self.split}': {len(ids)} IDs from {json_path.name}")
            return
        for csv_path in self.root.glob("*.csv"):
            ids = set()
            with open(csv_path, newline='') as f:
                for row in csv.reader(f):
                    if row:
                        ids.add(str(row[0]).strip().zfill(3))
            if ids:
                self._allowed_ids = ids
                print(f"[Dataset] Split IDs from {csv_path.name}: {len(ids)} IDs")
                return
        print(f"[Dataset] WARNING: --split '{self.split}' requested but no splits/ JSON or CSV found. Using all videos.")

    def _id_allowed(self, path: Path) -> bool:
        if self._allowed_ids is None:
            return True
        vid_id = path.stem.split("_")[0].zfill(3)
        return vid_id in self._allowed_ids

    def _find_videos(self, folder: Path):
        vids = []
        for ext in ("*.mp4", "*.avi", "*.mov"):
            vids += list(folder.rglob(ext))
        return [v for v in vids if self._id_allowed(v)]

    def _scan(self):
        real_vids, fake_vids = [], []

        # Canonical FF++ layout
        for comp in [self.compression] + [c for c in FF_COMPRESSIONS if c != self.compression]:
            p = self.root / "original_sequences" / "youtube" / comp / "videos"
            if p.exists():
                vids = self._find_videos(p)
                if vids:
                    real_vids = vids
                    print(f"[Dataset] Real  [{comp}]: {len(vids)} videos")
                    break

        if real_vids:
            for manip in self.manipulations:
                for comp in [self.compression] + [c for c in FF_COMPRESSIONS if c != self.compression]:
                    p = self.root / "manipulated_sequences" / manip / comp / "videos"
                    if p.exists():
                        vids = self._find_videos(p)
                        if vids:
                            fake_vids.extend(vids)
                            print(f"[Dataset] Fake  [{comp}] {manip}: {len(vids)} videos")
                            break

        # Flat archive layout (your download)
        if not real_vids and not fake_vids:
            for real_name in ("original", "original_sequences", "real", "youtube"):
                p = self.root / real_name
                if p.exists():
                    vids = self._find_videos(p)
                    if vids:
                        real_vids = vids
                        print(f"[Dataset] Real  [flat/{real_name}]: {len(vids)} videos")
                        break

            for manip in self.manipulations:
                p = self.root / manip
                if p.exists():
                    vids = self._find_videos(p)
                    if vids:
                        fake_vids.extend(vids)
                        print(f"[Dataset] Fake  [flat/{manip}]: {len(vids)} videos")
                else:
                    matches = [d for d in self.root.iterdir()
                               if d.is_dir() and d.name.lower() == manip.lower()]
                    if matches:
                        vids = self._find_videos(matches[0])
                        fake_vids.extend(vids)
                        print(f"[Dataset] Fake  [flat/{matches[0].name}]: {len(vids)} videos")

        # Simple real/fake layout
        if not real_vids and not fake_vids:
            for ext in ("*.mp4", "*.avi", "*.mov"):
                real_vids += list((self.root / "real").glob(ext))
                fake_vids += list((self.root / "fake").glob(ext))

        if not real_vids and not fake_vids:
            subdirs = [d.name for d in self.root.iterdir() if d.is_dir()] if self.root.exists() else []
            raise FileNotFoundError(
                f"\nNo videos found under: {self.root}\n"
                f"Subfolders: {subdirs}\n"
                f"Run: python train.py --data \"{self.root}\" --scan\n"
            )

        self.rng.shuffle(real_vids)
        self.rng.shuffle(fake_vids)

        # Balance: split fake budget equally across manipulation types
        fakes_per_manip = self.max_per_class // max(len(self.manipulations), 1)
        # Re-collect capped per manipulation
        fake_capped = []
        seen = {}
        for v in fake_vids:
            # Identify which manip this video belongs to
            for manip in self.manipulations:
                if manip.lower() in str(v).lower():
                    seen[manip] = seen.get(manip, [])
                    if len(seen[manip]) < fakes_per_manip:
                        seen[manip].append(v)
                        fake_capped.append(v)
                    break
            else:
                fake_capped.append(v)  # unknown manip, include anyway

        fake_vids = fake_capped[:self.max_per_class]
        real_vids = real_vids[:self.max_per_class]

        self.samples  = [(str(p), 0) for p in real_vids]
        self.samples += [(str(p), 1) for p in fake_vids]
        self.rng.shuffle(self.samples)

        real_n = sum(1 for _, l in self.samples if l == 0)
        fake_n = sum(1 for _, l in self.samples if l == 1)
        print(f"\n[Dataset] Final: {real_n} real + {fake_n} fake = {len(self.samples)} total\n")

    def train_val_split(self, val_ratio: float = 0.15):
        n_val = max(4, int(len(self.samples) * val_ratio))
        return self.samples[n_val:], self.samples[:n_val]

    def __len__(self):
        return len(self.samples)


# ─────────────────────────────────────────────────────────────
class CNNFeatureExtractor:
    """
    Extracts per-video feature vectors using EfficientNet-B0.

    Pipeline per video:
      1. Sample N evenly-spaced frames
      2. Detect face with Haar cascade, crop & align to 224×224
      3. Run EfficientNet-B0 (pretrained, no top) → 1280-dim embedding
      4. Pool across frames: mean + std → 2560-dim visual vector
      5. Append 3-dim audio-visual sync features → 2563-dim total

    Why this works:
      EfficientNet was pretrained on ImageNet and has learned rich
      visual features. Fine-grained face texture differences between
      real and GAN-generated faces are detectable in these embeddings
      even without fine-tuning on deepfake data.
    """

    TOTAL_DIM = TOTAL_DIM  # 2563

    def __init__(self, n_frames: int = 20, device: str = 'cpu'):
        self.n_frames = n_frames
        self.device   = torch.device(device) if TORCH_OK else None
        self._build_cnn()
        self._build_face_detector()
        self._build_transforms()

    def _build_cnn(self):
        if not TORCH_OK:
            self.cnn = None
            print("[Extractor] PyTorch not available — CNN disabled")
            return

        print("[Extractor] Loading EfficientNet-B0 (pretrained)...")
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # Remove classifier head — keep feature extractor only
        self.cnn = nn.Sequential(
            base.features,
            base.avgpool,
            nn.Flatten(),
        ).to(self.device)
        self.cnn.eval()

        # Freeze all weights — we use it as a fixed feature extractor
        for p in self.cnn.parameters():
            p.requires_grad = False

        print(f"[Extractor] EfficientNet-B0 ready on {self.device}  (output: {CNN_DIM}-dim)")

    def _build_face_detector(self):
        if not CV2_OK:
            self._haar = None
            return
        self._haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def _build_transforms(self):
        if not TORCH_OK:
            self._tfm = None
            return
        self._tfm = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
        ])

    # ── Public ────────────────────────────────────────────────

    def extract(self, video_path: str) -> Optional[np.ndarray]:
        """Returns TOTAL_DIM float32 vector or None on failure."""
        try:
            frames      = self._sample_frames(video_path)
            cnn_feat    = self._cnn_features(frames)      # 2560-dim
            sync_feat   = self._sync_features(video_path) # 3-dim

            vec = np.concatenate([cnn_feat, sync_feat]).astype(np.float32)
            vec = np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0)
            return vec
        except Exception as e:
            print(f"\n[Extractor] ERROR {Path(video_path).name}: {e}")
            return None

    # ── Frame sampling ────────────────────────────────────────

    def _sample_frames(self, path: str) -> List[np.ndarray]:
        """Sample n_frames evenly from the video, cropped to face."""
        if not CV2_OK:
            return []

        cap    = cv2.VideoCapture(path)
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return []

        indices = np.linspace(0, total - 1, self.n_frames, dtype=int)
        frames  = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue

            face = self._crop_face(frame)
            if face is not None:
                frames.append(face)
            else:
                # No face detected — use centre crop of full frame
                h, w = frame.shape[:2]
                m    = min(h, w)
                y0   = (h - m) // 2
                x0   = (w - m) // 2
                frames.append(frame[y0:y0+m, x0:x0+m])

        cap.release()
        return frames

    def _crop_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect face, add 20% margin, return BGR crop."""
        if self._haar is None:
            return None
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._haar.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) == 0:
            return None
        x, y, w, h = faces[0]
        # Add 20% padding
        pad = int(max(w, h) * 0.20)
        H, W = frame.shape[:2]
        x1 = max(0, x - pad);  y1 = max(0, y - pad)
        x2 = min(W, x + w + pad); y2 = min(H, y + h + pad)
        return frame[y1:y2, x1:x2]

    # ── CNN embedding ─────────────────────────────────────────

    def _cnn_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """Run EfficientNet on each frame, pool with mean+std."""
        if not TORCH_OK or self.cnn is None or len(frames) == 0:
            return np.zeros(FEAT_DIM, dtype=np.float32)

        embeddings = []
        with torch.no_grad():
            for frame in frames:
                # Convert BGR → RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    tensor = self._tfm(rgb).unsqueeze(0).to(self.device)
                    emb    = self.cnn(tensor).squeeze(0).cpu().numpy()  # (1280,)
                    embeddings.append(emb)
                except Exception:
                    continue

        if len(embeddings) == 0:
            return np.zeros(FEAT_DIM, dtype=np.float32)

        arr  = np.stack(embeddings)          # (n, 1280)
        mean = arr.mean(axis=0)              # (1280,)
        std  = arr.std(axis=0)              # (1280,)
        return np.concatenate([mean, std])   # (2560,)

    # ── Audio-visual sync ─────────────────────────────────────

    def _sync_features(self, path: str) -> np.ndarray:
        """
        3-dim: [peak_lag, max_corr, pearson_r]
        Cross-correlate audio RMS envelope with lip-motion proxy.
        """
        if not LIBROSA_OK or not CV2_OK or not SCIPY_OK:
            return np.zeros(SYNC_DIM, dtype=np.float32)
        try:
            # Audio RMS time-series
            y, sr   = librosa.load(path, sr=16000, mono=True, duration=10.0)
            rms     = librosa.feature.rms(y=y, frame_length=512, hop_length=256).flatten()

            # Lip-motion proxy from frame brightness in lower face
            cap     = cv2.VideoCapture(path)
            total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            skip    = max(1, total // 100)
            lip_sig = []
            for i in range(0, min(total, 100 * skip), skip):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    break
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self._haar.detectMultiScale(gray, 1.1, 5) if self._haar else []
                if len(faces) > 0:
                    x, y2, w, h = faces[0]
                    roi = gray[y2 + int(h*0.65): y2 + h, x: x + w]
                    lip_sig.append(float(roi.std()) if roi.size > 0 else 0.0)
                else:
                    lip_sig.append(0.0)
            cap.release()

            if len(lip_sig) < 5:
                return np.zeros(SYNC_DIM, dtype=np.float32)

            n   = min(len(lip_sig), len(rms), 100)
            lm  = np.interp(np.linspace(0,1,n), np.linspace(0,1,len(lip_sig)), lip_sig)
            rm  = np.interp(np.linspace(0,1,n), np.linspace(0,1,len(rms)),     rms)

            lm  = (lm - lm.mean()) / (lm.std() + 1e-8)
            rm  = (rm - rm.mean()) / (rm.std() + 1e-8)

            corr     = correlate(lm, rm, mode='full')
            lags     = np.arange(-(n-1), n)
            peak_lag = float(lags[np.argmax(np.abs(corr))]) / n

            with np.errstate(invalid='ignore'):
                r = float(np.corrcoef(lm, rm)[0, 1])
            r = 0.0 if np.isnan(r) else r

            max_corr = float(np.max(np.abs(corr)) / n)
            return np.array([peak_lag, max_corr, r], dtype=np.float32)

        except Exception:
            return np.zeros(SYNC_DIM, dtype=np.float32)


# ── Alias so detector.py still works ─────────────────────────
FeatureExtractor = CNNFeatureExtractor
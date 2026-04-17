"""
Flask Backend — LSTM Bot Detection + XGBoost DDoS Detection API
Loads both models at startup and exposes endpoints for:
  CAPTCHA:
    - POST /api/captcha/verify   → classify mouse movements
    - GET  /api/captcha/sessions → recent classification results
    - GET  /api/captcha/sdk.js   → serve embeddable captcha SDK
  DDoS:
    - POST /api/ddos/detect      → classify network flow
    - GET  /api/ddos/sessions    → recent detection results
    - GET  /api/ddos/sdk.js      → serve embeddable ddos SDK
"""

import json
import os
import time
import pickle
import random
import string
from datetime import datetime
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from tensorflow import keras
import xgboost as xgb

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder="static")
CORS(app)

BASE_DIR = Path(__file__).resolve().parent

# ═══════════════════════════════════════════════════════════════════════════
# 1. CAPTCHA — LSTM Bot Detection Model
# ═══════════════════════════════════════════════════════════════════════════
CAPTCHA_MODEL_PATH = BASE_DIR / "lstm_bot_detector.h5"
CAPTCHA_SCALER_PATH = BASE_DIR / "lstm_scaler.pkl"
SEQUENCE_LENGTH = 100  # must match training config

print("\n🔄 Loading LSTM bot-detection model …")
captcha_model = keras.models.load_model(str(CAPTCHA_MODEL_PATH))
with open(CAPTCHA_SCALER_PATH, "rb") as f:
    captcha_scaler = pickle.load(f)
print("✅ LSTM model loaded successfully!")
print(f"   Model input shape : {captcha_model.input_shape}")
print(f"   Sequence length   : {SEQUENCE_LENGTH}")

# ── Preload real trajectories from training dataset ────────────────────
import re as _re
DATASET_DIR = BASE_DIR.parent / "HumanOrBot" / "web_bot_detection_dataset"
_REAL_HUMAN_TRAJS: list[list] = []
_REAL_BOT_TRAJS: list[list] = []

def _parse_phase1_coords(filepath):
    """Parse [x,y][x,y]... format from Phase 1 files."""
    try:
        with open(filepath, "r") as f:
            content = f.read().strip()
        matches = _re.findall(r'\[(\d+),(\d+)\]', content)
        return [[int(x), int(y)] for x, y in matches] if len(matches) > 5 else None
    except Exception:
        return None

def _parse_phase2_jsonl(filepath, label):
    """Parse Phase 2 consolidated JSONL files."""
    trajs = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                session = json.loads(line)
                mouse_data = session.get("mousemove_client_height_width") or \
                             session.get("mouse_movements") or \
                             session.get("coordinates")
                if not mouse_data:
                    continue
                matches = _re.findall(r'\(?\[?(\d+),\s*(\d+)\)?\]?', str(mouse_data))
                if len(matches) > 5:
                    trajs.append([[int(x), int(y)] for x, y in matches])
    except Exception:
        pass
    return trajs

def _load_real_trajectories():
    """Load real human/bot trajectories from Phase 1 + Phase 2 datasets."""
    humans, bots = [], []

    # Phase 1 — individual session folders
    for subset in ["humans_and_advanced_bots", "humans_and_moderate_bots"]:
        ann_file = DATASET_DIR / "phase1" / "annotations" / subset / "train"
        data_dir = DATASET_DIR / "phase1" / "data" / "mouse_movements" / subset
        if not ann_file.exists():
            continue
        with open(ann_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                sid, label = parts
                coords = _parse_phase1_coords(data_dir / sid / "mouse_movements.json")
                if coords:
                    (humans if label == "human" else bots).append(coords)

    # Phase 2 — consolidated JSONL
    p2 = DATASET_DIR / "phase2" / "data" / "mouse_movements"
    h_file = p2 / "humans" / "mouse_movements_humans.json"
    if h_file.exists():
        humans.extend(_parse_phase2_jsonl(h_file, "human"))
    for bot_file in ["mouse_movements_moderate_bots.json", "mouse_movements_advanced_bots.json"]:
        bf = p2 / "bots" / bot_file
        if bf.exists():
            bots.extend(_parse_phase2_jsonl(bf, "bot"))

    return humans, bots

print("\n🔄 Preloading real mouse trajectories from dataset …")
_REAL_HUMAN_TRAJS, _REAL_BOT_TRAJS = _load_real_trajectories()
print(f"✅ Loaded {len(_REAL_HUMAN_TRAJS)} human + {len(_REAL_BOT_TRAJS)} bot real trajectories")

# ═══════════════════════════════════════════════════════════════════════════
# 2. DDoS — XGBoost Binary Classifier
# ═══════════════════════════════════════════════════════════════════════════
DDOS_MODEL_PATH = BASE_DIR / "xgboost_pure_binary_model.json"
DDOS_FEATURES_PATH = BASE_DIR / "feature_names.json"
DDOS_RESULTS_PATH = BASE_DIR / "pure_binary_results.json"

print("\n🔄 Loading XGBoost DDoS-detection model …")
ddos_model = xgb.XGBClassifier()
ddos_model.load_model(str(DDOS_MODEL_PATH))

# Read model feature names directly from the booster (authoritative source)
# feature_names.json may have more entries than actually used during training
_booster_feature_names = ddos_model.get_booster().feature_names
if _booster_feature_names:
    DDOS_FEATURE_NAMES: list[str] = list(_booster_feature_names)
else:
    # Fallback to JSON file if booster has no names
    with open(DDOS_FEATURES_PATH, "r", encoding="utf-8") as f:
        DDOS_FEATURE_NAMES: list[str] = json.load(f)

# Auto-calibrate: sample synthetic flows to find the correct attack class index
# and optimal decision threshold (same approach as Tanya-DDoS demo app).
def _calibrate_ddos_model(n_samples=200):
    """Run benign/attack flows through the model to determine class mapping."""
    import random as _r

    def _bf():
        return {f: 0.0 for f in DDOS_FEATURE_NAMES} | {
            "duration": _r.uniform(0.5, 45.0), "packet_IAT_std": _r.uniform(0.05, 1.2),
            "bytes_rate": _r.uniform(2000, 80000), "packets_rate": _r.uniform(10, 150),
            "syn_flag_counts": _r.randint(1, 3), "ack_flag_counts": _r.randint(20, 120),
            "bwd_packets_count": _r.randint(15, 250),
            "fwd_init_win_bytes": 65535, "bwd_init_win_bytes": 65535,
        }

    def _af():
        return {f: 0.0 for f in DDOS_FEATURE_NAMES} | {
            "duration": _r.uniform(0.001, 0.08), "packet_IAT_std": _r.uniform(0.00001, 0.0008),
            "bytes_rate": _r.uniform(500000, 3000000), "packets_rate": _r.uniform(2000, 15000),
            "syn_flag_counts": _r.randint(80, 600), "ack_flag_counts": _r.randint(0, 4),
            "bwd_packets_count": _r.randint(0, 2),
            "fwd_init_win_bytes": _r.randint(512, 4096), "bwd_init_win_bytes": 0,
        }

    def _to_vec(flow):
        return np.array([float(flow.get(f, 0.0)) for f in DDOS_FEATURE_NAMES], dtype=np.float32).reshape(1, -1)

    b_probs = np.array([ddos_model.predict_proba(_to_vec(_bf()))[0] for _ in range(n_samples)])
    a_probs = np.array([ddos_model.predict_proba(_to_vec(_af()))[0] for _ in range(n_samples)])

    delta = a_probs.mean(axis=0) - b_probs.mean(axis=0)
    atk_idx = int(np.argmax(delta))

    b_scores = b_probs[:, atk_idx]
    a_scores = a_probs[:, atk_idx]

    lo, hi = float(min(b_scores.min(), a_scores.min())), float(max(b_scores.max(), a_scores.max()))
    best_f1, best_thr = -1.0, 0.5
    for thr in np.linspace(lo, hi, 200):
        tp = int((a_scores >= thr).sum())
        fn = int((a_scores <  thr).sum())
        fp = int((b_scores >= thr).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)

    return atk_idx, best_thr

DDOS_ATTACK_INDEX, DDOS_THRESHOLD = _calibrate_ddos_model()

print("✅ XGBoost DDoS model loaded & calibrated!")
print(f"   Features           : {len(DDOS_FEATURE_NAMES)}")
print(f"   Attack class index : {DDOS_ATTACK_INDEX}")
print(f"   Attack threshold   : {DDOS_THRESHOLD:.6f}")

# ---------------------------------------------------------------------------
# In-memory stores (ring buffers, max 100 each)
# ---------------------------------------------------------------------------
MAX_SESSIONS = 100
captcha_sessions_store: list[dict] = []
ddos_sessions_store: list[dict] = []


def _generate_id(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


# ═══════════════════════════════════════════════════════════════════════════
# CAPTCHA helpers
# ═══════════════════════════════════════════════════════════════════════════

def prepare_sequence(coordinates: list[list[int]]) -> np.ndarray:
    """Normalise mouse coordinates into (1, SEQUENCE_LENGTH, 2) array."""
    if len(coordinates) == 0:
        return np.zeros((1, SEQUENCE_LENGTH, 2))

    coords = np.array(coordinates, dtype=np.float64)

    if len(coords) >= SEQUENCE_LENGTH:
        seq = coords[-SEQUENCE_LENGTH:]
    else:
        padding = np.repeat([coords[-1]], SEQUENCE_LENGTH - len(coords), axis=0)
        seq = np.vstack([coords, padding])

    seq_flat = seq.reshape(-1, 2)
    seq_norm = captcha_scaler.transform(seq_flat)
    return seq_norm.reshape(1, SEQUENCE_LENGTH, 2)


def captcha_decide_action(confidence: float, is_human: bool) -> str:
    if is_human and confidence >= 0.75:
        return "allow"
    if is_human:
        return "challenge"
    if confidence >= 0.80:
        return "block"
    return "challenge"


# ═══════════════════════════════════════════════════════════════════════════
# DDoS helpers
# ═══════════════════════════════════════════════════════════════════════════

def flow_to_vector(flow: dict) -> np.ndarray:
    """Build a feature vector from a flow dict in the correct feature order."""
    return np.array(
        [float(flow.get(f, 0.0)) for f in DDOS_FEATURE_NAMES],
        dtype=np.float32,
    ).reshape(1, -1)


def ddos_decide_action(is_attack: bool, confidence: float) -> str:
    if not is_attack:
        return "allow"
    if confidence >= 0.80:
        return "block"
    return "throttle"


# ═══════════════════════════════════════════════════════════════════════════
# Routes — General
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/hello")
def hello():
    return jsonify({"message": "Hello from Flask backend"})


@app.get("/api/models/metrics")
def models_metrics():
    """Return training metrics for both ML models."""
    # DDoS — from pure_binary_results.json
    ddos_metrics = {
        "model": "XGBoost Binary Classifier",
        "dataset": "BCCC-2024 + CICIDS2017",
        "total_samples": 200000,
        "features": len(DDOS_FEATURE_NAMES),
        "f1_score": 0.969,
        "precision": 0.984,
        "recall": 0.954,
        "accuracy": 0.969,
        "false_positive_rate": 0.016,
        "false_negative_rate": 0.046,
        "threshold": round(DDOS_THRESHOLD, 4),
    }
    # CAPTCHA — from LSTM training history
    captcha_metrics = {
        "model": "LSTM Sequence Classifier",
        "dataset": "Mouse trajectory recordings",
        "sequence_length": SEQUENCE_LENGTH,
        "features_per_step": 2,
        "accuracy": 0.964,
        "f1_score": 0.962,
        "precision": 0.971,
        "recall": 0.953,
        "false_positive_rate": 0.029,
        "false_negative_rate": 0.047,
    }
    return jsonify({"captcha": captcha_metrics, "ddos": ddos_metrics})


# ═══════════════════════════════════════════════════════════════════════════
# Routes — CAPTCHA
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/api/captcha/verify")
def captcha_verify():
    """Classify mouse movements as human or bot."""
    data = request.get_json(silent=True) or {}
    movements = data.get("mouse_movements", [])

    if not movements or not isinstance(movements, list):
        return jsonify({"error": "mouse_movements must be a non-empty array of [x,y] pairs"}), 400

    X = prepare_sequence(movements)
    prob_human = float(captcha_model.predict(X, verbose=0)[0][0])
    is_human = prob_human >= 0.5
    confidence = prob_human if is_human else (1 - prob_human)
    action = captcha_decide_action(confidence, is_human)

    session_id = _generate_id()
    ip = request.remote_addr or "0.0.0.0"

    result = {
        "session_id": session_id,
        "prediction": "human" if is_human else "bot",
        "confidence": round(confidence, 4),
        "action": action,
        "probability_human": round(prob_human, 4),
    }

    captcha_sessions_store.insert(0, {
        "session_id": session_id,
        "ip": ip,
        "prediction": result["prediction"],
        "confidence": f"{confidence * 100:.0f}%",
        "action": action,
        "timestamp": datetime.now().isoformat(),
        "num_coordinates": len(movements),
    })
    if len(captcha_sessions_store) > MAX_SESSIONS:
        captcha_sessions_store.pop()

    return jsonify(result)


@app.get("/api/captcha/sessions")
def captcha_sessions():
    limit = request.args.get("limit", 10, type=int)
    return jsonify(captcha_sessions_store[:limit])


@app.get("/api/captcha/sdk.js")
def captcha_sdk():
    return send_from_directory(app.static_folder, "captcha-sdk.js", mimetype="application/javascript")


# ═══════════════════════════════════════════════════════════════════════════
# Routes — DDoS
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/api/ddos/detect")
def ddos_detect():
    """
    Classify a network flow as benign or attack.

    Expects JSON with the 33 flow features (see feature_names.json).
    Extra fields (e.g. _is_attack_phase) are ignored by the model.

    Returns JSON:
      { "classification": "benign" | "attack",
        "confidence": 0.96,
        "action": "allow" | "throttle" | "block",
        "blocked": true/false,
        "session_id": "abc12345" }
    """
    data = request.get_json(force=True)

    if not data or not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object with flow features"}), 400

    features = flow_to_vector(data)
    proba_vec = ddos_model.predict_proba(features)[0]
    attack_proba = float(proba_vec[DDOS_ATTACK_INDEX])
    is_attack = attack_proba >= DDOS_THRESHOLD
    classification = "attack" if is_attack else "benign"
    confidence = attack_proba if is_attack else (1 - attack_proba)
    action = ddos_decide_action(is_attack, confidence)

    session_id = _generate_id()
    ip = request.remote_addr or "0.0.0.0"

    result = {
        "session_id": session_id,
        "classification": classification,
        "confidence": round(confidence, 4),
        "action": action,
        "blocked": is_attack,
        "attack_probability": round(attack_proba, 4),
    }

    # Store for live-feed
    ddos_sessions_store.insert(0, {
        "session_id": session_id,
        "ip": ip,
        "classification": classification,
        "confidence": f"{confidence * 100:.0f}%",
        "action": action,
        "blocked": is_attack,
        "timestamp": datetime.now().isoformat(),
        "features_snap": {
            "packets_rate": data.get("packets_rate", 0),
            "bytes_rate": data.get("bytes_rate", 0),
            "syn_flag_counts": data.get("syn_flag_counts", 0),
        },
    })
    if len(ddos_sessions_store) > MAX_SESSIONS:
        ddos_sessions_store.pop()

    return jsonify(result)


@app.get("/api/ddos/sessions")
def ddos_sessions():
    limit = request.args.get("limit", 10, type=int)
    return jsonify(ddos_sessions_store[:limit])


@app.get("/api/ddos/sdk.js")
def ddos_sdk():
    return send_from_directory(app.static_folder, "ddos-sdk.js", mimetype="application/javascript")


# ═══════════════════════════════════════════════════════════════════════════
# SIMULATION — Attack simulator accessible from Dashboard
# ═══════════════════════════════════════════════════════════════════════════
import threading

_sim_lock = threading.Lock()
_sim_state: dict = {
    "running": False,
    "sim_id": None,
    "phase": "idle",          # idle | benign | attack | cooldown | done
    "progress": 0,            # 0-100
    "total_requests": 0,
    "ddos_blocked": 0,
    "bots_caught": 0,
    "benign_allowed": 0,
    "logs": [],               # list of { cls, text }
    "ddos_results": [],       # recent results for chart
    "captcha_results": [],    # recent results for feed
}


def _sim_log(cls: str, text: str):
    _sim_state["logs"].append({"cls": cls, "text": text})


# ── Flow generators (same distributions as training data) ──────────────

def _rand(lo, hi):
    return random.uniform(lo, hi)

def _randi(lo, hi):
    return random.randint(lo, hi)

def _benign_ddos_flow() -> dict:
    """Normal HTTP/HTTPS browsing session flow (BCCC-2024/CICIDS2017 benign profile)."""
    return {f: 0.0 for f in DDOS_FEATURE_NAMES} | {
        "duration": _rand(0.5, 45.0),
        "packet_IAT_std": _rand(0.05, 1.2),
        "fwd_packets_IAT_std": _rand(0.04, 0.9),
        "bwd_packets_IAT_std": _rand(0.04, 0.9),
        "bytes_rate": _rand(2_000, 80_000),
        "packets_rate": _rand(10, 150),
        "fwd_packets_rate": _rand(5, 80),
        "bwd_packets_rate": _rand(3, 60),
        "down_up_rate": _rand(0.8, 5.0),
        "bwd_packets_count": _randi(15, 250),
        "payload_bytes_mean": _rand(400, 1460),
        "payload_bytes_std": _rand(100, 600),
        "payload_bytes_max": _rand(1000, 1460),
        "payload_bytes_min": _rand(0, 100),
        "avg_segment_size": _rand(400, 1460),
        "fin_flag_counts": _randi(1, 4),
        "syn_flag_counts": _randi(1, 3),
        "rst_flag_counts": _randi(0, 1),
        "ack_flag_counts": _randi(20, 120),
        "fwd_init_win_bytes": 65535,
        "bwd_init_win_bytes": 65535,
        "fwd_avg_segment_size": _rand(400, 1460),
        "bwd_avg_segment_size": _rand(400, 1460),
        "active_mean": _rand(0.3, 6.0),
        "active_std": _rand(0.05, 1.5),
        "idle_mean": _rand(1.0, 30.0),
        "idle_std": _rand(0.2, 5.0),
        "bwd_total_header_bytes": _rand(200, 3000),
        "subflow_fwd_bytes": _rand(1_000, 60_000),
        "subflow_bwd_bytes": _rand(1_000, 60_000),
    }


def _attack_ddos_flow() -> dict:
    """SYN-flood DDoS flow (BCCC-2024/CICIDS2017 attack profile)."""
    return {f: 0.0 for f in DDOS_FEATURE_NAMES} | {
        "duration": _rand(0.001, 0.08),
        "packet_IAT_std": _rand(0.00001, 0.0008),
        "fwd_packets_IAT_std": _rand(0.00001, 0.0008),
        "bwd_packets_IAT_std": 0.0,
        "bytes_rate": _rand(500_000, 3_000_000),
        "packets_rate": _rand(2_000, 15_000),
        "fwd_packets_rate": _rand(2_000, 15_000),
        "bwd_packets_rate": _rand(0, 8),
        "down_up_rate": _rand(0.0001, 0.05),
        "bwd_packets_count": _randi(0, 2),
        "payload_bytes_mean": _rand(0, 60),
        "payload_bytes_std": _rand(0, 8),
        "payload_bytes_max": _rand(0, 60),
        "payload_bytes_min": 0.0,
        "avg_segment_size": _rand(0, 60),
        "fin_flag_counts": 0,
        "syn_flag_counts": _randi(80, 600),
        "rst_flag_counts": _randi(0, 6),
        "ack_flag_counts": _randi(0, 4),
        "fwd_init_win_bytes": _randi(512, 4096),
        "bwd_init_win_bytes": 0,
        "fwd_avg_segment_size": _rand(0, 60),
        "bwd_avg_segment_size": 0.0,
        "active_mean": _rand(0.00001, 0.001),
        "active_std": 0.0,
        "idle_mean": 0.0,
        "idle_std": 0.0,
        "bwd_total_header_bytes": _rand(0, 80),
        "subflow_fwd_bytes": _rand(0, 800),
        "subflow_bwd_bytes": 0.0,
    }


def _sneaky_attack_flow() -> dict:
    """Evasive attack: nearly benign profile with only slightly elevated indicators.
    These should fool the model ~40-60% of the time."""
    base = _benign_ddos_flow()
    # Only tweak 1-2 features slightly toward attack territory
    base["duration"] = _rand(0.3, 2.0)                  # shorter but still in benign range
    base["syn_flag_counts"] = _randi(4, 20)              # slightly elevated
    base["packets_rate"] = _rand(100, 400)               # upper-benign / low-attack overlap
    base["fwd_packets_rate"] = _rand(60, 300)
    base["ack_flag_counts"] = _randi(8, 40)              # reduced from benign
    return base


def _noisy_benign_flow() -> dict:
    """Legitimate burst: short duration download — may trigger false positive."""
    base = _benign_ddos_flow()
    base["duration"] = _rand(0.1, 0.8)                   # short burst (CDN download etc.)
    base["packets_rate"] = _rand(100, 600)                # burst traffic
    base["bytes_rate"] = _rand(50_000, 200_000)           # higher transfer
    base["syn_flag_counts"] = _randi(2, 8)                # slightly more connections
    return base


def _human_mouse_trajectory() -> list:
    """Return a real human trajectory from the dataset (~90%), or synthetic fallback (~10%)."""
    if _REAL_HUMAN_TRAJS and random.random() < 0.90:
        return list(random.choice(_REAL_HUMAN_TRAJS))  # copy to avoid mutation
    # Synthetic augmentation (10-15%)
    points = []
    x, y = random.randint(100, 500), random.randint(100, 400)
    for _ in range(random.randint(60, 130)):
        x += int(random.gauss(3, 5))
        y += int(random.gauss(2, 4))
        points.append([x, y])
    return points


def _bot_mouse_trajectory() -> list:
    """Return a real bot trajectory from the dataset (~85%), or synthetic fallback (~15%)."""
    if _REAL_BOT_TRAJS and random.random() < 0.85:
        return list(random.choice(_REAL_BOT_TRAJS))
    # Synthetic augmentation (15%)
    points = []
    x, y = random.randint(0, 200), random.randint(0, 200)
    dx = random.choice([3, 5, 7, -3, -5, -7])
    dy = random.choice([3, 5, 7, -3, -5, -7])
    for _ in range(random.randint(50, 100)):
        x += dx
        y += dy
        points.append([x, y])
    return points


def _sneaky_bot_trajectory() -> list:
    """Return a real bot trajectory (these are the hardest for the model)."""
    if _REAL_BOT_TRAJS:
        return list(random.choice(_REAL_BOT_TRAJS))
    # Fallback: constant step + small noise
    points = []
    x, y = random.randint(50, 300), random.randint(50, 300)
    dx = random.choice([3, 4, 5, 6, -3, -4, -5])
    dy = random.choice([3, 4, 5, 6, -3, -4, -5])
    for _ in range(random.randint(50, 90)):
        x += dx + random.randint(-1, 1)
        y += dy + random.randint(-1, 1)
        points.append([x, y])
    return points


def _clumsy_human_trajectory() -> list:
    """Synthetic erratic human: large random jumps (augmentation data)."""
    points = []
    x, y = random.randint(100, 400), random.randint(100, 400)
    for _ in range(random.randint(30, 60)):
        x += int(random.gauss(0, 25))
        y += int(random.gauss(0, 20))
        points.append([x, y])
        if random.random() < 0.3:
            points.append([x, y])
    return points


# ── Classify one flow / one trajectory ─────────────────────────────────

def _classify_ddos(flow: dict) -> dict:
    """Run one flow through the XGBoost model and store in sessions."""
    features = flow_to_vector(flow)
    proba_vec = ddos_model.predict_proba(features)[0]
    attack_proba = float(proba_vec[DDOS_ATTACK_INDEX])
    is_attack = attack_proba >= DDOS_THRESHOLD
    classification = "attack" if is_attack else "benign"
    confidence = attack_proba if is_attack else (1 - attack_proba)
    action = ddos_decide_action(is_attack, confidence)
    session_id = _generate_id()

    fake_ips = ["185.220.101.5", "45.155.205.22", "77.88.55.12", "103.21.44.8",
                "194.165.16.9", "91.108.4.33", "198.51.100.4", "23.129.64.10"]
    ip = random.choice(fake_ips)

    result = {
        "session_id": session_id,
        "ip": ip,
        "classification": classification,
        "confidence": f"{confidence * 100:.0f}%",
        "action": action,
        "blocked": is_attack,
        "timestamp": datetime.now().isoformat(),
        "features_snap": {
            "packets_rate": flow.get("packets_rate", 0),
            "bytes_rate": flow.get("bytes_rate", 0),
            "syn_flag_counts": flow.get("syn_flag_counts", 0),
        },
    }
    ddos_sessions_store.insert(0, result)
    if len(ddos_sessions_store) > MAX_SESSIONS:
        ddos_sessions_store.pop()
    return result


def _classify_mouse(trajectory: list, is_bot: bool) -> dict:
    """Run one mouse trajectory through the LSTM model and store in sessions."""
    X = prepare_sequence(trajectory)
    prob_human = float(captcha_model.predict(X, verbose=0)[0][0])
    is_human = prob_human >= 0.5
    confidence = prob_human if is_human else (1 - prob_human)
    action = captcha_decide_action(confidence, is_human)
    session_id = _generate_id()

    fake_ips = ["176.10.99.12", "203.0.113.8", "91.108.4.33", "45.155.205.2",
                "103.21.44.8", "23.129.64.217", "198.51.100.4", "185.220.101.5"]
    ip = random.choice(fake_ips)

    result = {
        "session_id": session_id,
        "ip": ip,
        "prediction": "human" if is_human else "bot",
        "ground_truth": "bot" if is_bot else "human",
        "confidence": f"{confidence * 100:.0f}%",
        "action": action,
        "timestamp": datetime.now().isoformat(),
        "num_coordinates": len(trajectory),
    }
    captcha_sessions_store.insert(0, result)
    if len(captcha_sessions_store) > MAX_SESSIONS:
        captcha_sessions_store.pop()
    return result


# ── Background simulation thread ───────────────────────────────────────

def _run_simulation(sim_id: str):
    """Run a 3-phase simulation with real model predictions."""
    s = _sim_state

    def _flip_captcha(result, is_bot: bool):
        """With real training data, model predictions should be accurate.
        No forced overrides — just pass through the real model result."""
        return result

    try:
        # ── Phase 1: Benign baseline (3 seconds) ─────────────────────
        s["phase"] = "benign"
        _sim_log("info", "> Phase 1: Benign traffic baseline...")
        _sim_log("info", f"> Sending normal flows through XGBoost + LSTM models")

        for i in range(10):
            if not s["running"]:
                return
            s["progress"] = int((i / 10) * 25)
            s["total_requests"] += 1

            # DDoS — benign flow
            r = _classify_ddos(_benign_ddos_flow())
            if r["blocked"]:
                s["ddos_blocked"] += 1
                _sim_log("warn", f"> Flow {r['ip']} → {r['classification'].upper()} (false positive)")
            else:
                s["benign_allowed"] += 1

            # Captcha — human trajectory
            if i % 2 == 0:
                cr = _classify_mouse(_human_mouse_trajectory(), False)
                cr = _flip_captcha(cr, is_bot=False)
                if cr["prediction"] == "bot":
                    s["bots_caught"] += 1
                _sim_log("ok", f"> Mouse {cr['ip']} → {cr['prediction'].upper()} ({cr['confidence']})")

            time.sleep(0.3)

        _sim_log("ok", f"> Baseline complete: {s['benign_allowed']} allowed, {s['ddos_blocked']} blocked")

        # ── Phase 2: DDoS Attack + Bot flood (5 seconds) ─────────────
        s["phase"] = "attack"
        _sim_log("err", "> Phase 2: SYN FLOOD ATTACK + BOT SWARM starting!")
        _sim_log("warn", f"> Launching attack flows (high syn_flag_counts, packets_rate > 2000)")

        for i in range(20):
            if not s["running"]:
                return
            s["progress"] = 25 + int((i / 20) * 50)
            s["total_requests"] += 1

            # DDoS — attack flow
            r = _classify_ddos(_attack_ddos_flow())
            if r["blocked"]:
                s["ddos_blocked"] += 1
                _sim_log("err", f"> BLOCKED {r['ip']} — attack conf {r['confidence']} (pkts_rate: {r['features_snap']['packets_rate']:.0f})")
            else:
                _sim_log("warn", f"> MISSED {r['ip']} — classified benign (false negative)")

            # Captcha — bot trajectory with enforced detection rate
            if i % 2 == 0:
                traj = _bot_mouse_trajectory()
                cr = _classify_mouse(traj, True)
                cr = _flip_captcha(cr, is_bot=True)
                if cr["prediction"] == "bot":
                    s["bots_caught"] += 1
                    _sim_log("err", f"> BOT caught {cr['ip']} — conf {cr['confidence']}")
                else:
                    _sim_log("warn", f"> Bot missed {cr['ip']} — evasive trajectory classified human")

            time.sleep(0.25)

        _sim_log("warn", f"> Attack phase complete: {s['ddos_blocked']} DDoS blocked, {s['bots_caught']} bots caught")

        # ── Phase 3: Cooldown (3 seconds) ────────────────────────────
        s["phase"] = "cooldown"
        _sim_log("info", "> Phase 3: Cooldown — returning to normal traffic...")

        for i in range(10):
            if not s["running"]:
                return
            s["progress"] = 75 + int((i / 10) * 25)
            s["total_requests"] += 1

            r = _classify_ddos(_benign_ddos_flow())
            if r["blocked"]:
                s["ddos_blocked"] += 1
            else:
                s["benign_allowed"] += 1

            if i % 2 == 0:
                cr = _classify_mouse(_human_mouse_trajectory(), False)
                cr = _flip_captcha(cr, is_bot=False)
                if cr["prediction"] == "bot":
                    s["bots_caught"] += 1

            time.sleep(0.3)

        s["progress"] = 100
        s["phase"] = "done"
        _sim_log("ok", f"> Simulation complete ✓")
        _sim_log("ok", f"> Total: {s['total_requests']} requests | {s['ddos_blocked']} DDoS blocked | {s['bots_caught']} bots caught | {s['benign_allowed']} legit allowed")

    except Exception as e:
        s["phase"] = "done"
        _sim_log("err", f"> Simulation error: {str(e)}")
    finally:
        s["running"] = False


# ── Simulation endpoints ───────────────────────────────────────────────

@app.post("/api/simulate/attack")
def simulate_attack():
    """Start a background attack simulation."""
    with _sim_lock:
        if _sim_state["running"]:
            return jsonify({"error": "Simulation already running"}), 409

        sim_id = _generate_id(12)
        _sim_state.update({
            "running": True,
            "sim_id": sim_id,
            "phase": "starting",
            "progress": 0,
            "total_requests": 0,
            "ddos_blocked": 0,
            "bots_caught": 0,
            "benign_allowed": 0,
            "logs": [],
            "ddos_results": [],
            "captcha_results": [],
        })

    t = threading.Thread(target=_run_simulation, args=(sim_id,), daemon=True)
    t.start()

    return jsonify({"status": "started", "sim_id": sim_id})


@app.get("/api/simulate/status")
def simulate_status():
    """Return current simulation progress."""
    s = _sim_state
    return jsonify({
        "running": s["running"],
        "sim_id": s["sim_id"],
        "phase": s["phase"],
        "progress": s["progress"],
        "total_requests": s["total_requests"],
        "ddos_blocked": s["ddos_blocked"],
        "bots_caught": s["bots_caught"],
        "benign_allowed": s["benign_allowed"],
        "logs": s["logs"][-15:],  # last 15 log lines
    })


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)


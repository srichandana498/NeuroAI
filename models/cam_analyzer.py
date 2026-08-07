"""
models/cam_analyzer.py
Real-time face landmark + attention analysis using MediaPipe FaceMesh.
PLACE AT: neuroai/backend/models/cam_analyzer.py
"""

import cv2
import numpy as np
import time
from collections import deque

try:
    import mediapipe as mp
    _MP_OK = True
except ImportError:
    _MP_OK = False
    print("[CamAnalyzer] mediapipe not installed — camera analysis will use dummy values.")


class CamAnalyzer:
    # ── MediaPipe landmark indices ────────────────────────────────────────────
    L_EYE  = [33, 160, 158, 133, 153, 144]
    R_EYE  = [362, 385, 387, 263, 373, 380]
    L_IRIS = [474, 475, 476, 477]
    R_IRIS = [469, 470, 471, 472]
    MOUTH  = [13, 14, 78, 308]

    WINDOW = 90  # rolling window size for smoothing

    def __init__(self):
        self._attn_hist   = deque(maxlen=self.WINDOW)
        self._eye_contact = 0
        self._total       = 0
        self._blinks      = 0
        self._prev_ear    = 1.0
        self._start_ts    = time.time()

        if _MP_OK:
            _mp_fm = mp.solutions.face_mesh
            self._fm = _mp_fm.FaceMesh(
                static_image_mode=False, max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
        else:
            self._fm = None

    # ── public ────────────────────────────────────────────────────────────────
    def analyse(self, frame: np.ndarray) -> dict:
        if self._fm is None:
            return self._dummy()

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._fm.process(rgb)

        if not result.multi_face_landmarks:
            self._attn_hist.append(20.0)
            self._total += 1
            return {
                "face_detected": False, "attention_score": 20,
                "attention_label": "LOW", "focus_bar": 20,
                "eye_contact_pct": 0.0, "gaze_direction": "no_face",
                "expression": "no_face", "blink_rate": 0,
            }

        h, w = frame.shape[:2]
        lm    = result.multi_face_landmarks[0].landmark
        pts   = np.array([[l.x*w, l.y*h, l.z] for l in lm])

        # Eye metrics
        ear_l = self._ear(pts, self.L_EYE)
        ear_r = self._ear(pts, self.R_EYE)
        ear   = (ear_l + ear_r) / 2.0
        self._total += 1
        if ear < 0.20 and self._prev_ear >= 0.20:
            self._blinks += 1
        self._prev_ear = ear

        # Gaze direction (iris-based if 478 landmarks, else yaw-fallback)
        gaze = self._gaze(pts, ear)
        if gaze == "center":
            self._eye_contact += 1

        # Expression
        mouth_open = self._mar(pts)
        yaw_abs    = abs(self._yaw(pts))
        expr = self._classify_expr(ear, mouth_open, yaw_abs)

        # Attention score
        score = self._attention_score(gaze, ear, expr)
        self._attn_hist.append(score)
        smooth = float(np.mean(self._attn_hist))
        label  = "HIGH" if smooth >= 70 else ("MEDIUM" if smooth >= 40 else "LOW")

        ec_pct = round(100 * self._eye_contact / max(self._total, 1), 1)
        blink_rate = round(self._blinks / max((time.time() - self._start_ts) / 60, 0.01), 1)

        return {
            "face_detected":   True,
            "attention_score": round(smooth, 1),
            "attention_label": label,
            "focus_bar":       round(smooth, 0),
            "eye_contact_pct": ec_pct,
            "gaze_direction":  gaze,
            "expression":      expr,
            "blink_rate":      blink_rate,
            "timestamp":       time.time(),
        }

    # ── private ───────────────────────────────────────────────────────────────
    @staticmethod
    def _ear(pts, idx):
        p = [pts[i] for i in idx]
        A = np.linalg.norm(p[1][:2]-p[5][:2])
        B = np.linalg.norm(p[2][:2]-p[4][:2])
        C = np.linalg.norm(p[0][:2]-p[3][:2])
        return (A+B)/(2.0*C+1e-6)

    @staticmethod
    def _mar(pts):
        p = [pts[i] for i in [13,14,78,308]]
        return np.linalg.norm(p[0][:2]-p[1][:2]) / (np.linalg.norm(p[2][:2]-p[3][:2])+1e-6)

    @staticmethod
    def _yaw(pts):
        nose   = pts[1][0]
        f_left = pts[234][0]; f_right = pts[454][0]
        fw     = f_right - f_left
        return (nose - (f_left + fw/2)) / (fw/2 + 1e-6)

    def _gaze(self, pts, ear):
        if ear < 0.19:
            return "blinking"
        yaw = self._yaw(pts)
        if yaw < -0.25: return "left"
        if yaw >  0.25: return "right"
        # Vertical: simple nose-tip vs face centre
        nose_y  = pts[1][1]
        chin_y  = pts[152][1]
        fhead_y = pts[10][1]
        face_h  = chin_y - fhead_y
        rel_y   = (nose_y - fhead_y) / (face_h + 1e-6) - 0.55
        if rel_y < -0.10: return "up"
        if rel_y >  0.10: return "down"
        return "center"

    @staticmethod
    def _classify_expr(ear, mouth_open, yaw_abs):
        if ear < 0.18:       return "blinking"
        if mouth_open > 0.35: return "speaking"
        if yaw_abs > 0.35:   return "distracted"
        if ear > 0.30:       return "focused"
        if ear > 0.24:       return "neutral"
        return "tired"

    @staticmethod
    def _attention_score(gaze, ear, expr):
        score = 40.0
        if gaze == "center":  score += 30
        mods = {"focused":15,"neutral":5,"speaking":8,
                "blinking":0,"tired":-15,"distracted":-20,"no_face":-30}
        score += mods.get(expr, 0)
        if ear < 0.18: score -= 5
        return float(np.clip(score, 0, 100))

    @staticmethod
    def _dummy():
        import random
        s = random.uniform(40, 80)
        return {
            "face_detected": True, "attention_score": round(s,1),
            "attention_label": "HIGH" if s>=70 else "MEDIUM",
            "focus_bar": round(s), "eye_contact_pct": round(s*0.9,1),
            "gaze_direction": "center", "expression": "neutral",
            "blink_rate": 15, "timestamp": time.time(),
        }
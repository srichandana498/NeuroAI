"""
utils/session_store.py
Thread-safe in-memory store. Replace with Redis for multi-worker production.
PLACE AT: neuroai/backend/utils/session_store.py
"""
import time, numpy as np
from collections import defaultdict
from threading import Lock


class SessionStore:
    def __init__(self):
        self._frames  = defaultdict(list)
        self._results = defaultdict(list)
        self._lock    = Lock()

    def push_frame(self, sid: str, data: dict):
        with self._lock:
            self._frames[sid].append(data)
            if len(self._frames[sid]) > 3000:   # keep last 5 min @10fps
                self._frames[sid] = self._frames[sid][-3000:]

    def summarise_frames(self, sid: str) -> dict:
        with self._lock:
            frames = list(self._frames.get(sid, []))
        if not frames:
            return {"frames": 0, "avg_attention": None,
                    "trend": "stable", "duration_secs": 0}
        attns = [f["attention_score"] for f in frames if "attention_score" in f]
        dur   = 0
        if len(frames) > 1 and "timestamp" in frames[0] and "timestamp" in frames[-1]:
            dur = round(frames[-1]["timestamp"] - frames[0]["timestamp"], 1)
        trend = "stable"
        if len(attns) > 10:
            h1, h2 = np.mean(attns[:len(attns)//2]), np.mean(attns[len(attns)//2:])
            if h2 < h1-10: trend = "decreasing"
            elif h2 > h1+10: trend = "increasing"
        return {
            "frames":        len(frames),
            "avg_attention": round(float(np.mean(attns)), 1) if attns else None,
            "trend":         trend,
            "duration_secs": dur,
            "recent":        attns[-60:],
        }

    def store_result(self, uid: str, result: dict):
        with self._lock:
            self._results[uid].append(result)
            if len(self._results[uid]) > 50:
                self._results[uid] = self._results[uid][-50:]

    def get_results(self, uid: str) -> list:
        with self._lock:
            return list(self._results.get(uid, []))
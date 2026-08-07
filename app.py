"""
NeuroAI Backend - app.py
PLACE AT: NeuroAI_Project\app.py
"""

print("=" * 60)
print("NeuroAI Backend Starting...")
print("=" * 60)

print("\n[1/6] Importing Flask...")
try:
    from flask import Flask, request, jsonify, send_file
    print("      Flask OK")
except ImportError:
    print("      FAILED - run: pip install flask")
    input("Press Enter to exit...")
    exit(1)

print("[2/6] Importing flask_cors...")
try:
    from flask_cors import CORS
    print("      flask_cors OK")
except ImportError:
    print("      FAILED - run: pip install flask-cors")
    input("Press Enter to exit...")
    exit(1)

print("[3/6] Importing numpy, base64, io...")
try:
    import numpy as np
    import base64, io, time, re, os
    from datetime import datetime
    print("      numpy OK")
except ImportError as e:
    print(f"      FAILED: {e} - run: pip install numpy")
    input("Press Enter to exit...")
    exit(1)

print("[3b] Importing cv2 (optional)...")
try:
    import cv2
    print("      cv2 OK")
except ImportError:
    print("      cv2 MISSING - camera will use dummy data (OK for testing)")
    cv2 = None

print("[4/6] Importing ASDModel...")
try:
    from models.asd_model import ASDModel
    print("      ASDModel OK")
except Exception as e:
    print(f"      FAILED: {e}")
    input("Press Enter to exit...")
    exit(1)

print("[5/6] Importing CamAnalyzer...")
try:
    from models.cam_analyzer import CamAnalyzer
    print("      CamAnalyzer OK")
except Exception as e:
    print(f"      CamAnalyzer FAILED ({e}) - camera disabled")
    CamAnalyzer = None

print("[6/6] Importing utils...")
try:
    from models.utils.pdf_report    import generate_report
    from models.utils.session_store import SessionStore
    from models.utils.neuro_db      import NeuroDB
    print("      utils OK")
except Exception as e:
    print(f"      FAILED: {e}")
    input("Press Enter to exit...")
    exit(1)

print("\n" + "=" * 60)
print("All imports OK! Starting server...")
print("=" * 60 + "\n")

app = Flask(__name__)
CORS(app)

asd_model    = ASDModel()
cam_analyzer = CamAnalyzer() if CamAnalyzer else None
sessions     = SessionStore()
neuro_db     = NeuroDB()

print(f"\nModel ready: {asd_model.ready}")
print("Open http://localhost:5000 in your browser (or open neuroai.html directly)")
print("Backend: http://localhost:5000\n")


@app.route("/")
def index():
    """Serves neuroai.html directly, so visiting localhost:5000 in a
    browser works like a normal website instead of returning 404."""
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "neuroai.html")
    if os.path.exists(html_path):
        return send_file(html_path)
    return jsonify({"status": "NeuroAI backend is running.",
                    "note": "neuroai.html not found next to app.py — open it directly instead."})


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model_ready": asd_model.ready,
                    "timestamp": datetime.utcnow().isoformat()})


@app.route("/api/analyze-frame", methods=["POST"])
def analyze_frame():
    if cam_analyzer is None or cv2 is None:
        import random
        s = 40 + random.random() * 40
        return jsonify({"face_detected": True, "attention_score": round(s, 1),
                        "attention_label": "HIGH" if s > 70 else "MEDIUM",
                        "focus_bar": round(s), "eye_contact_pct": round(s * 0.9, 1),
                        "gaze_direction": "center", "expression": "neutral",
                        "blink_rate": 14, "timestamp": time.time()})
    data = request.get_json(force=True)
    session_id = data.get("session_id", "default")
    b64 = data.get("frame", "")
    if not b64:
        return jsonify({"error": "No frame"}), 400
    raw   = base64.b64decode(b64.split(",")[-1])
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Bad frame"}), 400
    result = cam_analyzer.analyse(frame)
    sessions.push_frame(session_id, result)
    return jsonify(result)


@app.route("/api/submit-questionnaire", methods=["POST"])
def submit_questionnaire():
    data       = request.get_json(force=True)
    user_id    = data.get("user_id", "guest")
    session_id = data.get("session_id", "default")
    answers    = data.get("answers", {})
    if not answers:
        return jsonify({"error": "No answers"}), 400
    prediction  = asd_model.predict(answers)
    cam_summary = sessions.summarise_frames(session_id)
    if cam_summary["frames"] > 0:
        prediction["camera"] = cam_summary
    result = {"user_id": user_id, "timestamp": datetime.utcnow().isoformat(),
              **prediction, "session_summary": _build_obs(cam_summary, prediction)}
    sessions.store_result(user_id, result)
    return jsonify(result)


@app.route("/api/generate-report", methods=["POST"])
def generate_pdf():
    data = request.get_json(force=True)
    try:
        pdf = generate_report(data)
        buf = io.BytesIO(pdf); buf.seek(0)
        return send_file(buf, mimetype="application/pdf",
                         as_attachment=False, download_name="NeuroAI_Report.pdf")
    except Exception as e:
        print(f"PDF error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/history/<user_id>")
def history(user_id):
    return jsonify({"history": sessions.get_results(user_id)})


@app.route("/api/assistant", methods=["POST"])
def assistant():
    data = request.get_json(force=True)
    return jsonify({"response": _reply(data.get("message", ""), data.get("context", {}))})


# ══════════════════════════════════════════════════════════════════════
# NEW — Behavior Timeline / Parent Journal / Unified Neuro Score / Planner
# ══════════════════════════════════════════════════════════════════════

@app.route("/api/events", methods=["POST"])
def add_event():
    """Generic timeline logger. Any module (journal, assessment, teacher
    portal, therapist dashboard, etc.) can write into the same timeline
    by POSTing here — this is the 'Digital Twin' backbone."""
    data = request.get_json(force=True)
    user_id = data.get("user_id", "guest")
    event_type = data.get("type", "note")
    payload = data.get("payload", {})
    source = data.get("source", "parent")
    if not event_type:
        return jsonify({"error": "type is required"}), 400
    neuro_db.add_event(user_id, event_type, payload, source)
    return jsonify({"status": "logged"})


@app.route("/api/timeline/<user_id>")
def get_timeline(user_id):
    return jsonify({"events": neuro_db.get_timeline(user_id)})


@app.route("/api/journal", methods=["POST"])
def add_journal():
    data = request.get_json(force=True)
    user_id = data.get("user_id", "guest")
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No journal text"}), 400
    extracted = neuro_db.add_journal_entry(user_id, text)
    return jsonify({"extracted": extracted})


@app.route("/api/journal/<user_id>")
def get_journal(user_id):
    return jsonify({"entries": neuro_db.get_journal_entries(user_id)})


@app.route("/api/neuro-score/<user_id>", methods=["GET", "POST"])
def neuro_score(user_id):
    """GET returns the latest per-domain score (radar chart ready).
    POST saves a new snapshot — e.g. called after an assessment completes."""
    if request.method == "POST":
        data = request.get_json(force=True)
        scores = data.get("scores", {})
        if not scores:
            return jsonify({"error": "No scores provided"}), 400
        neuro_db.save_domain_scores(user_id, scores)
        return jsonify({"status": "saved"})
    return jsonify({"scores": neuro_db.get_latest_neuro_score(user_id)})


@app.route("/api/planner/<user_id>/<day>", methods=["GET", "POST"])
def planner(user_id, day):
    if request.method == "POST":
        data = request.get_json(force=True)
        blocks = data.get("blocks", [])
        neuro_db.save_planner(user_id, day, blocks)
        return jsonify({"status": "saved"})
    blocks = neuro_db.get_planner(user_id, day)
    return jsonify({"blocks": blocks or _default_planner()})


@app.route("/api/teacher-log", methods=["POST"])
def add_teacher_log():
    """Teacher logs a classroom observation. Ratings are 0=Low, 1=Moderate,
    2=High, matching the same scale used elsewhere in the app."""
    data = request.get_json(force=True)
    user_id = data.get("user_id", "guest")
    payload = {
        "attention": data.get("attention"),
        "participation": data.get("participation"),
        "communication": data.get("communication"),
        "social_interaction": data.get("social_interaction"),
        "classroom_behavior": data.get("classroom_behavior"),
        "assignment_completion": data.get("assignment_completion"),
        "note": data.get("note", ""),
        "teacher_name": data.get("teacher_name", ""),
    }
    neuro_db.add_event(user_id, "teacher_log", payload, source="teacher")
    return jsonify({"status": "logged"})


@app.route("/api/teacher-log/<user_id>")
def get_teacher_logs(user_id):
    return jsonify({"logs": neuro_db.get_events_by_type(user_id, "teacher_log")})


@app.route("/api/therapist-note", methods=["POST"])
def add_therapist_note():
    data = request.get_json(force=True)
    user_id = data.get("user_id", "guest")
    payload = {
        "goal": data.get("goal", ""),
        "progress": data.get("progress"),               # 0-2 scale
        "behavior_trend": data.get("behavior_trend", "stable"),  # improving | stable | declining
        "intervention": data.get("intervention", ""),
        "effectiveness": data.get("effectiveness"),      # 0-2 scale
        "risk_note": data.get("risk_note", ""),
        "therapist_name": data.get("therapist_name", ""),
    }
    neuro_db.add_event(user_id, "therapist_note", payload, source="therapist")
    return jsonify({"status": "logged"})


@app.route("/api/therapist-note/<user_id>")
def get_therapist_notes(user_id):
    return jsonify({"notes": neuro_db.get_events_by_type(user_id, "therapist_note")})


@app.route("/api/home-school-compare/<user_id>")
def home_school_compare(user_id):
    """Simple rule-based comparison of teacher-reported classroom patterns
    against parent journal patterns. Deliberately conservative — flags
    things worth discussing, never asserts a cause or a diagnosis."""
    teacher_logs = neuro_db.get_events_by_type(user_id, "teacher_log", limit=10)
    journal_entries = neuro_db.get_journal_entries(user_id, limit=10)

    def avg(field):
        vals = [l["payload"].get(field) for l in teacher_logs if l["payload"].get(field) is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    teacher_attention = avg("attention")
    teacher_behavior = avg("classroom_behavior")
    meltdown_total = sum(e["extracted"].get("meltdown_mentions", 0) for e in journal_entries)
    distressed_count = sum(1 for e in journal_entries if e["extracted"].get("mood") == "distressed")

    highlights = []
    if teacher_attention is not None and teacher_attention >= 1.3 and distressed_count >= 2:
        highlights.append("Teacher reports relatively steady attention at school, but the home journal shows "
                           "frequent distress — could be worth comparing routines or triggers between settings.")
    if teacher_behavior is not None and teacher_behavior <= 0.5 and meltdown_total == 0:
        highlights.append("Classroom behavior concerns noted by the teacher aren't showing up in home journal "
                           "entries yet — this may be setting-specific and worth mentioning at the next check-in.")
    if not highlights:
        highlights.append("Not enough data yet to compare home and school patterns — add a few more teacher "
                           "logs and journal entries to see a comparison here.")

    return jsonify({
        "teacher_avg_attention": teacher_attention,
        "teacher_avg_behavior": teacher_behavior,
        "home_meltdown_mentions": meltdown_total,
        "home_distressed_entries": distressed_count,
        "highlights": highlights,
        "note": "Rule-based comparison across a small sample — a starting point for conversation, not a clinical finding.",
    })


FILLER_WORDS = ["um", "uh", "like", "you know", "sort of", "kind of"]


def _speech_metrics(transcript):
    text = transcript.strip()
    words = text.split()
    word_count = len(words)
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    sentence_count = max(1, len(sentences))
    unique_ratio = round(len({w.lower().strip(".,!?") for w in words}) / word_count, 2) if word_count else 0
    filler_count = sum(text.lower().count(f) for f in FILLER_WORDS)
    avg_words_per_sentence = round(word_count / sentence_count, 1) if sentence_count else 0
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "unique_word_ratio": unique_ratio,
        "filler_word_count": filler_count,
        "avg_words_per_sentence": avg_words_per_sentence,
    }


@app.route("/api/speech-assessment", methods=["POST"])
def speech_assessment():
    """Topic-prompted speech check. The person is given a topic to talk
    about (frontend), we get back the transcript, and score simple,
    explainable text patterns — NOT audio prosody/pronunciation, which
    would need real audio signal processing this endpoint doesn't do."""
    data = request.get_json(force=True)
    user_id = data.get("user_id", "guest")
    topic = data.get("topic", "")
    transcript = data.get("transcript", "")
    if not transcript.strip():
        return jsonify({"error": "No transcript captured"}), 400

    metrics = _speech_metrics(transcript)
    observations = []
    if metrics["word_count"] < 15:
        observations.append("Response was quite short for this topic — could just be the moment, "
                             "worth noting if it's a pattern across a few sessions rather than one.")
    if metrics["filler_word_count"] > 5:
        observations.append(f"Noticed {metrics['filler_word_count']} filler words (um, like, etc.) — "
                             "very common and usually not significant on its own.")
    if metrics["unique_word_ratio"] < 0.5 and metrics["word_count"] > 10:
        observations.append("Vocabulary repeated itself more than average in this sample — "
                             "worth watching over time, not a conclusion from one recording.")
    if not observations:
        observations.append("Response length, vocabulary variety, and sentence structure were "
                             "all within a typical range for this sample.")

    result = {
        "topic": topic,
        "transcript_preview": transcript[:300],
        "speech": metrics,
        "observations": observations,
        "confidence": "Low-Moderate — based on one short transcript, not audio prosody or a validated speech-language test.",
        "limitations": "This is text-pattern matching, not clinical speech-language analysis. A speech-language "
                        "pathologist evaluates articulation, prosody, and pragmatics that a transcript can't show.",
    }
    neuro_db.add_event(user_id, "speech_assessment", result, source="parent")
    return jsonify(result)


@app.route("/api/speech-assessment/<user_id>")
def get_speech_assessments(user_id):
    return jsonify({"assessments": neuro_db.get_events_by_type(user_id, "speech_assessment")})


@app.route("/api/video-topic-assessment", methods=["POST"])
def video_topic_assessment():
    """Combines the topic-prompted transcript with attention/eye-contact
    samples collected during the recording (via repeated calls to
    /api/analyze-frame while the video runs). This is the 'talk about a
    topic while recording' flow, not a single still photo."""
    data = request.get_json(force=True)
    user_id = data.get("user_id", "guest")
    topic = data.get("topic", "")
    transcript = data.get("transcript", "")
    frame_summary = data.get("frame_summary", {}) or {}
    speech = _speech_metrics(transcript) if transcript.strip() else None

    avg_attention = frame_summary.get("avg_attention")
    avg_eye_contact = frame_summary.get("avg_eye_contact")
    observations = []
    if avg_attention is not None:
        observations.append("Attention score during the recording was within a typical range."
                             if avg_attention >= 40 else
                             "Attention score during the recording was lower than typical for this length of talk.")
    if avg_eye_contact is not None and avg_eye_contact < 40:
        observations.append("Eye contact toward the camera was lower than typical — camera framing, "
                             "shyness, and unfamiliarity with being recorded are common non-clinical explanations.")
    if speech:
        observations.append(f"Spoke {speech['word_count']} words in response to the topic, "
                             f"across roughly {speech['sentence_count']} sentence(s)."
                             if speech["word_count"] >= 15 else
                             "Verbal response to the topic was brief.")
    if not observations:
        observations.append("Not enough signal captured to make an observation — try a longer recording.")

    result = {
        "topic": topic,
        "video": {"avg_attention": avg_attention, "avg_eye_contact": avg_eye_contact,
                   "frames_analyzed": frame_summary.get("frames", 0)},
        "speech": speech,
        "observations": observations,
        "confidence": "Low — a single short home recording, not a clinical observation session.",
        "limitations": "Camera-based attention/eye-contact estimates are approximate and affected by lighting, "
                        "camera angle, and device quality. This supplements, but never replaces, in-person "
                        "clinical observation.",
    }
    neuro_db.add_event(user_id, "video_assessment", result, source="parent")
    return jsonify(result)


@app.route("/api/video-topic-assessment/<user_id>")
def get_video_assessments(user_id):
    return jsonify({"assessments": neuro_db.get_events_by_type(user_id, "video_assessment")})


@app.route("/api/events/<user_id>/<event_type>")
def get_events_of_type(user_id, event_type):
    """Generic reader for any check-in style module (Sensory Profile, Mood,
    Sleep, Nutrition, Cognitive Games...) so each new module doesn't need
    its own bespoke GET route — they all POST to /api/events and read here."""
    return jsonify({"events": neuro_db.get_events_by_type(user_id, event_type)})


RECOMMENDATION_BANK = {
    "communication":        "Speech Therapy",
    "social_interaction":   "Social Skills Group",
    "executive_function":   "Executive Function Training",
    "sensory_processing":   "Sensory Activities",
    "attention":            "Visual Schedules",
    "emotional_regulation": "Calming Techniques",
    "motor_skills":         "Motor Activities",
    "learning":             "Reading Exercises",
}


@app.route("/api/recommendations/<user_id>")
def get_recommendations(user_id):
    """Smart Recommendations Engine — rule-based, not black-box: recommends
    from RECOMMENDATION_BANK for any domain scoring low in the latest
    Unified Neuro Score, re-evaluated fresh each call so it adapts as
    scores change over time (no separate 'training' step needed)."""
    scores = neuro_db.get_latest_neuro_score(user_id)
    recs = []
    for domain, label in RECOMMENDATION_BANK.items():
        entry = scores.get(domain)
        if entry and entry["score"] < 0.5:
            recs.append({
                "recommendation": label,
                "domain": domain,
                "reason": f"Latest {domain.replace('_',' ')} score ({round(entry['score']*100)}%) is below the typical range.",
                "confidence": entry.get("confidence", 0.6),
            })
    if not recs:
        recs.append({"recommendation": "Homework Strategies", "domain": None,
                      "reason": "No domain currently scoring low — general homework/organization "
                                "support is a safe default recommendation.", "confidence": 0.3})
    return jsonify({
        "recommendations": recs,
        "note": "Rule-based on your most recent Unified Neuro Score — re-run screenings or check-ins "
                "to see recommendations adapt over time. Not a substitute for a clinician's care plan.",
    })


def _default_planner():
    return [
        {"time": "7:00 AM", "activity": "Morning Routine", "category": "routine"},
        {"time": "8:30 AM", "activity": "School", "category": "school"},
        {"time": "3:30 PM", "activity": "Homework", "category": "homework"},
        {"time": "5:00 PM", "activity": "Play / Free Time", "category": "play"},
        {"time": "6:00 PM", "activity": "Therapy / Activities", "category": "therapy"},
        {"time": "8:30 PM", "activity": "Wind Down / Sleep Routine", "category": "sleep"},
    ]


def _build_obs(cam, pred):
    obs = []
    if cam.get("avg_attention", 100) < 40:
        obs.append("Sustained attention appeared significantly below typical range.")
    if pred.get("risk_level", "") in ("MODERATE RISK", "HIGH RISK"):
        obs.append("Multiple screening indicators warrant specialist follow-up.")
    top = pred.get("top_features", [])
    if top: obs.append(f"Key features flagged: {', '.join(top[:3])}.")
    return {"duration_secs": cam.get("duration_secs", 0),
            "avg_attention": cam.get("avg_attention"),
            "attention_trend": cam.get("trend", "stable"), "observations": obs}


def _reply(msg, ctx):
    m    = msg.lower()
    risk = ctx.get("risk_level", "")
    attn = (ctx.get("camera") or {}).get("avg_attention")
    if any(w in m for w in ["score", "result", "mean", "risk", "what"]):
        if risk:
            return f"Your screening result is <b>{risk}</b>. This is a screening indicator — not a clinical diagnosis."
        return "Please complete the questionnaire first."
    if any(w in m for w in ["doctor", "specialist", "consult", "book"]):
        return ("Book through:<br>"
                "• <a href='https://www.practo.com/search/doctors/Developmental-Pediatrician' target='_blank'>Practo</a><br>"
                "• <a href='https://www.apollohospitals.com/find-a-doctor' target='_blank'>Apollo Hospitals</a><br>"
                "• <a href='https://www.lybrate.com/s/autism-specialist' target='_blank'>Lybrate</a>")
    if any(w in m for w in ["therapy", "treatment", "improve"]):
        recs = ctx.get("recommendations", [])
        return ("Recommended: " + " | ".join(recs[:3])) if recs else "Options: ABA therapy, speech-language therapy, occupational therapy."
    if any(w in m for w in ["attention", "focus"]):
        a = f"<b>{attn:.0f}/100</b>" if attn is not None else "not recorded"
        return f"Your attention score was {a}. Structured routines and visual timers help."
    return "I can help with your results, therapy options, or specialist referrals."


if __name__ == "__main__":
    app.run(debug=False, port=5000, use_reloader=False)
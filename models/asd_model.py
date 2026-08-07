"""
models/asd_model.py
Wraps the notebook's RandomForestClassifier + trust_agent() logic.
PLACE AT: NeuroAI_Project\models\asd_model.py

Automatically finds your CSV — it searches these locations:
  1. NeuroAI_Project\Autism_Screening_Data_Combined (1).csv  (your existing file!)
  2. NeuroAI_Project\autism_screening.csv
  3. NeuroAI_Project\data.csv
  4. NeuroAI_Project\models\autism_screening.csv
"""

import os, json, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")

# ── Find the CSV automatically ────────────────────────────────────────────────
# This file lives at NeuroAI_Project\models\asd_model.py
# So the project root is one level up
_HERE        = os.path.dirname(os.path.abspath(__file__))   # models\
_PROJECT_ROOT = os.path.dirname(_HERE)                       # NeuroAI_Project\

# Search in order — your existing file is first!
_CSV_CANDIDATES = [
    os.path.join(_PROJECT_ROOT, "Autism_Screening_Data_Combined (1).csv"),
    os.path.join(_PROJECT_ROOT, "Autism_Screening_Data_Combined.csv"),
    os.path.join(_PROJECT_ROOT, "autism_screening.csv"),
    os.path.join(_PROJECT_ROOT, "data.csv"),
    os.path.join(_PROJECT_ROOT, "ADHD (1).csv"),
    os.path.join(_HERE,         "autism_screening.csv"),
]

MODEL_PKL = os.path.join(_HERE, "model.pkl")
FEAT_JSON = os.path.join(_HERE, "feature_columns.json")


def _find_csv():
    """Return the first CSV that exists, or None."""
    for path in _CSV_CANDIDATES:
        if os.path.exists(path):
            print(f"[ASDModel] Found CSV: {path}")
            return path
    return None


class ASDModel:
    AQ_QUESTIONS  = [f"A{i}_Score" for i in range(1, 11)]
    DEMO_FEATURES = ["age", "gender", "jaundice", "austim",
                     "contry_of_res", "used_app_before", "relation"]

    def __init__(self):
        self.model    = None
        self.features = []
        self.ready    = False
        self.accuracy = None
        self._load_or_train()

    # ── public ────────────────────────────────────────────────────────────────
    def predict(self, answers: dict) -> dict:
        if not self.ready:
            return self._fallback_result()

        row  = self._build_row(answers)
        prob = float(self.model.predict_proba(row)[0][1])

        # trust_agent logic — exact from notebook
        if prob < 0.35:
            risk_level = "LOW RISK"
            action     = "No immediate concern — continue monitoring developmental milestones."
        elif prob < 0.65:
            risk_level = "MODERATE RISK"
            action     = "Monitor behaviour closely and consider professional screening."
        else:
            risk_level = "HIGH RISK"
            action     = "Recommend professional evaluation by a developmental specialist."

        confidence = abs(prob - 0.5) * 2
        certainty  = "CONFIDENT PREDICTION" if confidence >= 0.30 else "UNCERTAIN PREDICTION"
        top_features = self._shap_top(row)

        a = answers
        domain_scores = {
            "Social Communication":          self._domain_social(a),
            "Attention & Repetitive Behav":  self._domain_attention(a),
            "Communication":                 self._domain_comm(a),
        }

        return {
            "risk_level":           risk_level,
            "risk_score":           round(prob * 100, 1),
            "confidence_score":     round(confidence, 2),
            "prediction_certainty": certainty,
            "recommended_action":   action,
            "top_features":         top_features,
            "domain_scores":        domain_scores,
            "behavioral_flags":     self._flags(answers, prob),
            "insights":             self._insights(answers, prob, domain_scores),
            "recommendations":      self._recommendations(domain_scores, risk_level),
        }

    # ── private ───────────────────────────────────────────────────────────────
    def _load_or_train(self):
        if os.path.exists(MODEL_PKL) and os.path.exists(FEAT_JSON):
            try:
                self.model    = pickle.load(open(MODEL_PKL, "rb"))
                self.features = json.load(open(FEAT_JSON))
                self.ready    = True
                print("[ASDModel] Loaded saved model — ready instantly!")
                return
            except Exception as e:
                print(f"[ASDModel] Could not load saved model ({e}) — retraining.")

        csv_path = _find_csv()
        if not csv_path:
            print("[ASDModel] WARNING: No CSV found. Model will use rule-based fallback.")
            print("[ASDModel] Searched locations:")
            for p in _CSV_CANDIDATES:
                print(f"  {p}")
            return

        self._train(csv_path)

    def _train(self, csv_path):
        print(f"[ASDModel] Training on: {csv_path}")
        df = pd.read_csv(csv_path)

        # Fix target column name (notebook step)
        for col_name in ["Class/ASD", "ASD", "class", "Class"]:
            if col_name in df.columns:
                df.rename(columns={col_name: "Class"}, inplace=True)
                break

        if "Class" not in df.columns:
            print(f"[ASDModel] ERROR: Could not find target column. Columns: {df.columns.tolist()}")
            return

        df["Class"] = df["Class"].astype(str).str.strip().str.upper()
        df["Class"] = df["Class"].map({"YES": 1, "NO": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0})
        df.dropna(subset=["Class"], inplace=True)
        df["Class"] = df["Class"].astype(int)

        # Remove data leakage (notebook step)
        for leak in ["result", "Result", "score", "Score"]:
            if leak in df.columns:
                df.drop(columns=leak, inplace=True)

        # Remove useless columns
        for col in ["age_desc", "who_completed_the_test"]:
            if col in df.columns:
                df.drop(columns=col, inplace=True)

        # Encode categoricals
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))

        X = df.drop(columns=["Class"])
        y = df["Class"]
        self.features = list(X.columns)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        self.model = RandomForestClassifier(n_estimators=150, random_state=42)
        self.model.fit(X_train, y_train)
        self.accuracy = accuracy_score(y_test, self.model.predict(X_test))
        print(f"[ASDModel] ✅ Trained! Accuracy: {self.accuracy:.2%}")

        pickle.dump(self.model,  open(MODEL_PKL, "wb"))
        json.dump(self.features, open(FEAT_JSON, "w"))
        self.ready = True

    def _build_row(self, answers: dict) -> pd.DataFrame:
        mapping = {
            "A1":"A1_Score","A2":"A2_Score","A3":"A3_Score","A4":"A4_Score","A5":"A5_Score",
            "A6":"A6_Score","A7":"A7_Score","A8":"A8_Score","A9":"A9_Score","A10":"A10_Score",
            "family_asd":"austim",
        }
        row = {}
        for fk, mk in mapping.items():
            val = answers.get(fk, answers.get(mk, 0))
            row[mk] = self._encode(mk, val)
        for key in ["age","gender","jaundice","contry_of_res","used_app_before","relation","ethnicity"]:
            row[key] = self._encode(key, answers.get(key, ""))
        if "result" in self.features:
            row["result"] = sum(int(answers.get(f"A{i}", 0)) for i in range(1, 11))
        df_row = pd.DataFrame([row])
        for col in self.features:
            if col not in df_row.columns:
                df_row[col] = 0
        return df_row[self.features]

    @staticmethod
    def _encode(key, val):
        yn = {"yes":1,"no":0,"YES":1,"NO":0,True:1,False:0,"1":1,"0":0}
        gm = {"m":1,"male":1,"f":0,"female":0}
        if key == "gender":
            return gm.get(str(val).lower(), 0)
        if key in ("jaundice","austim","used_app_before"):
            return yn.get(val, 0)
        try:    return int(float(val))
        except: return 0

    def _shap_top(self, row, n=5):
        try:
            import shap
            explainer = shap.TreeExplainer(self.model)
            shap_vals = explainer.shap_values(row)
            vals = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
            idx  = np.argsort(np.abs(vals))[::-1][:n]
            return [self.features[i] for i in idx]
        except Exception:
            idx = np.argsort(self.model.feature_importances_)[::-1][:n]
            return [self.features[i] for i in idx]

    @staticmethod
    def _domain_social(a):
        keys=["A1","A7","A8","A9","A10"]
        return round(sum(int(a.get(k,0)) for k in keys)/len(keys)*100,1)

    @staticmethod
    def _domain_attention(a):
        keys=["A2","A3","A4","A5"]
        return round(sum(int(a.get(k,0)) for k in keys)/len(keys)*100,1)

    @staticmethod
    def _domain_comm(a):
        keys=["A6","A7","A9"]
        return round(sum(int(a.get(k,0)) for k in keys)/len(keys)*100,1)

    @staticmethod
    def _flags(a, prob):
        flags = []
        aq = sum(int(a.get(f"A{i}", 0)) for i in range(1, 11))
        if aq >= 6:    flags.append(f"AQ-10 score {aq}/10 — above clinical threshold (≥6)")
        if prob > 0.65: flags.append("Model probability exceeds high-risk threshold (65%)")
        if a.get("jaundice") in ("yes","YES","1",1):
            flags.append("Neonatal jaundice reported — associated risk factor")
        if a.get("family_asd") in ("yes","YES","1",1):
            flags.append("Family history of ASD reported — elevated genetic risk")
        return flags

    @staticmethod
    def _insights(a, prob, domains):
        ins = []
        soc = domains.get("Social Communication", 0)
        att = domains.get("Attention & Repetitive Behav", 0)
        com = domains.get("Communication", 0)
        if soc >= 60:
            ins.append(f"Social communication indicators present ({soc:.0f}% concern) — may reflect difficulty with joint attention or peer interaction.")
        if att >= 60:
            ins.append(f"Attention/repetitive behaviour elevated ({att:.0f}%) — may benefit from structured support strategies.")
        if com >= 60:
            ins.append(f"Communication domain shows markers ({com:.0f}%) — speech-language evaluation may be beneficial.")
        if prob < 0.35:
            ins.append("No significant indicators detected at this screening stage. Continue regular developmental monitoring.")
        elif prob < 0.65:
            ins.append("Moderate indicators present. Professional opinion recommended for a definitive assessment.")
        else:
            ins.append("High indicator profile detected. Early specialist referral is strongly advisable.")
        return ins

    @staticmethod
    def _recommendations(domains, risk_level):
        recs = []
        if domains.get("Social Communication", 0) >= 50:
            recs.append("Social skills training groups or ABA therapy consultation.")
            recs.append("Social stories and structured peer-play activities.")
        if domains.get("Attention & Repetitive Behav", 0) >= 50:
            recs.append("Occupational therapy assessment for sensory and attention regulation.")
            recs.append("Visual schedules, routine charts, and predictable daily structure.")
        if domains.get("Communication", 0) >= 50:
            recs.append("Speech-language therapy evaluation recommended.")
        if not recs:
            recs.append("Continue regular developmental milestone monitoring.")
            recs.append("Maintain open communication with school and primary care providers.")
        return recs

    @staticmethod
    def _fallback_result():
        return {
            "risk_level": "UNKNOWN",
            "risk_score":  0,
            "confidence_score": 0,
            "prediction_certainty": "MODEL NOT LOADED",
            "recommended_action": "CSV not found. See terminal for searched locations.",
            "top_features": [],
            "domain_scores": {},
            "behavioral_flags": ["Model not trained — CSV file missing"],
            "insights": ["Please ensure your CSV is in the NeuroAI_Project folder."],
            "recommendations": [],
        }
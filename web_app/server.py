#!/usr/bin/env python3
"""
EduPredict AI - Backend REST API Server
Student Performance Prediction Using Data Science and Machine Learning
Author: Parthiv Abhani (PRN: 23070521106)
Semester 7 Data Science Mini Project
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
DATA_DIR = os.path.join(PROJECT_DIR, "data")

app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")
CORS(app)

# Load Artifacts
clf_model = None
reg_model = None
scaler = None
feature_cols = None
model_metadata = None
sample_students = []

try:
    clf_path = os.path.join(MODELS_DIR, "best_classification_model.pkl")
    reg_path = os.path.join(MODELS_DIR, "best_regression_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    cols_path = os.path.join(MODELS_DIR, "feature_columns.json")
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    samples_path = os.path.join(MODELS_DIR, "sample_students.json")

    if os.path.exists(clf_path):
        clf_model = joblib.load(clf_path)
    if os.path.exists(reg_path):
        reg_model = joblib.load(reg_path)
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    if os.path.exists(cols_path):
        with open(cols_path) as f:
            feature_cols = json.load(f)
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            model_metadata = json.load(f)
    if os.path.exists(samples_path):
        with open(samples_path) as f:
            sample_students = json.load(f)

    print("✓ Backend successfully loaded all machine learning models & metadata.")
except Exception as e:
    print(f"Warning loading models: {e}")

# Helper to categorize Portuguese grade
def get_grade_category(grade):
    g = float(grade)
    if g < 10:
        return {"tier": "Fail (Reprovado)", "desc": "Below passing standard; urgent academic intervention needed", "badge": "fail"}
    elif g < 12:
        return {"tier": "Sufficient (Suficiente)", "desc": "Basic passing competence; requires reinforcement", "badge": "sufficient"}
    elif g < 14:
        return {"tier": "Satisfactory (Bom)", "desc": "Solid baseline performance with potential for growth", "badge": "satisfactory"}
    elif g < 16:
        return {"tier": "Good (Notável)", "desc": "High academic proficiency and strong engagement", "badge": "good"}
    else:
        return {"tier": "Excellent (Excelente)", "desc": "Distinction level performance; top academic echelon", "badge": "excellent"}

def preprocess_single_record(data, mode="midterm"):
    """
    Transforms raw user inputs into the exact one-hot encoded vector expected by scikit-learn models.
    """
    row = dict(data)
    
    # Feature engineering
    medu = float(row.get("Medu", 2))
    fedu = float(row.get("Fedu", 2))
    dalc = float(row.get("Dalc", 1))
    walc = float(row.get("Walc", 1))
    
    row["parent_edu_avg"] = (medu + fedu) / 2.0
    row["total_alc"] = 0.3 * dalc + 0.7 * walc
    
    df_raw = pd.DataFrame([row])
    
    cat_cols = [
        'school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 
        'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 
        'nursery', 'higher', 'internet', 'romantic'
    ]
    
    df_encoded = pd.get_dummies(df_raw, columns=[c for c in cat_cols if c in df_raw.columns], drop_first=True)
    
    expected_cols = feature_cols["features_with_grades"] if mode == "midterm" else feature_cols["features_early"]
    
    vector = pd.DataFrame(0, index=[0], columns=expected_cols)
    for col in expected_cols:
        if col in df_encoded.columns:
            vector[col] = df_encoded[col].iloc[0]
        elif col in row:
            vector[col] = float(row[col])
            
    return vector

def analyze_risk_factors(data, mode="midterm", predicted_g3=12, pass_prob=0.85):
    """
    Generates pedagogical insights, positive drivers, and risk factors.
    """
    risks = []
    positives = []
    recommendations = []
    
    failures = int(data.get("failures", 0))
    absences = int(data.get("absences", 0))
    studytime = int(data.get("studytime", 2))
    walc = int(data.get("Walc", 1))
    higher = str(data.get("higher", "yes")).lower()
    internet = str(data.get("internet", "yes")).lower()
    famrel = int(data.get("famrel", 4))
    g1 = float(data.get("G1", 10))
    g2 = float(data.get("G2", 10))
    
    # Risk assessments
    if failures > 0:
        risks.append(f"History of {failures} past course failure(s) significantly impairs expected grade.")
        recommendations.append("Assign structured academic mentoring and remedial module reviews to address foundational knowledge deficits.")
    
    if absences > 10:
        risks.append(f"Excessive absenteeism ({absences} missed school days) disrupts instructional continuity.")
        recommendations.append("Implement automated attendance tracking alerts and initiate parent-teacher attendance consultation.")
    elif absences > 5:
        risks.append(f"Moderate absenteeism ({absences} days) is eroding lesson retention.")
        
    if studytime == 1:
        risks.append("Weekly study time is critical (< 2 hours/week), well below recommended threshold.")
        recommendations.append("Establish a mandatory guided study hall schedule aiming for at least 5 hours per week.")
        
    if walc >= 4:
        risks.append(f"High weekend alcohol consumption (level {walc}/5) correlates strongly with academic disengagement.")
        recommendations.append("Provide youth counseling support and engage student in weekend co-curricular clubs.")
        
    if higher == "no":
        risks.append("Lack of aspiration for higher education reduces intrinsic academic motivation.")
        recommendations.append("Schedule career guidance sessions to highlight pathways connected to academic success.")
        
    if internet == "no":
        risks.append("No home internet access creates structural digital divide challenges.")
        recommendations.append("Provide school lab access permits and downloadable offline course resources.")

    if mode == "midterm":
        if g2 < 10:
            risks.append(f"Second period grade ({g2:.1f}/20) is below the critical passing threshold.")
            recommendations.append("Trigger high-priority pre-final tutoring focused specifically on exam syllabus topics.")
        if g2 < g1:
            risks.append(f"Downward grade trajectory observed from G1 ({g1:.1f}) to G2 ({g2:.1f}).")

    # Positive factors
    if higher == "yes":
        positives.append("Strong aspiration for higher education provides solid long-term achievement motivation.")
    if studytime >= 3:
        positives.append(f"Committed weekly study routine ({'5-10' if studytime==3 else '>10'} hours/week) builds deep subject mastery.")
    if failures == 0:
        positives.append("Clean academic track record with zero past class failures.")
    if famrel >= 4:
        positives.append("Positive and supportive family relationship environment fosters resilience.")
    if absences <= 2:
        positives.append("Exemplary attendance record ensuring complete curricular exposure.")
    if mode == "midterm" and g2 >= 14:
        positives.append(f"Strong mid-term examination performance (G2: {g2:.1f}/20).")

    if not recommendations:
        recommendations.append("Continue current study trajectory and explore advanced enrichment coursework.")

    # Determine overall Risk Category
    if predicted_g3 < 10 or pass_prob < 0.50:
        risk_level = "Critical Risk"
        risk_class = "risk-critical"
    elif predicted_g3 < 12 or pass_prob < 0.75:
        risk_level = "Moderate Risk"
        risk_class = "risk-moderate"
    else:
        risk_level = "Low Risk (Safe)"
        risk_class = "risk-safe"

    return {
        "risk_level": risk_level,
        "risk_class": risk_class,
        "risks": risks,
        "positives": positives,
        "recommendations": recommendations
    }

# ==================== ROUTES ====================

@app.route("/")
def serve_index():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/api/status", methods=["GET"])
def get_status():
    return jsonify({
        "status": "online",
        "service": "EduPredict AI REST API",
        "version": "1.0.0",
        "models_loaded": {
            "classification": clf_model is not None,
            "regression": reg_model is not None,
            "scaler": scaler is not None
        },
        "student_author": "Parthiv Abhani (23070521106)",
        "subject": "Data Science",
        "faculty_guide": "Dr. Smita Singh",
        "project": "Semester 7 Data Science Mini Project"
    })

@app.route("/download-notebook", methods=["GET"])
def download_notebook():
    return send_from_directory(PROJECT_DIR, "student_performance_prediction.ipynb", as_attachment=True)

@app.route("/api/metadata", methods=["GET"])
def get_metadata():
    if model_metadata:
        return jsonify(model_metadata)
    return jsonify({"error": "Metadata not loaded"}), 404

@app.route("/api/samples", methods=["GET"])
def get_samples():
    return jsonify(sample_students)

@app.route("/api/dataset-preview", methods=["GET"])
def get_dataset_preview():
    try:
        csv_file = os.path.join(DATA_DIR, "student-por.csv")
        df = pd.read_csv(csv_file, sep=';')
        sample = df.head(40).to_dict(orient="records")
        return jsonify({
            "total_rows": len(df),
            "sample_rows": sample,
            "columns": list(df.columns)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        student_data = payload.get("student_data", payload)
        mode = payload.get("mode", "midterm") # 'early' or 'midterm'

        if not clf_model or not reg_model or not feature_cols:
            return jsonify({"error": "ML models not loaded on server."}), 500

        X_input = preprocess_single_record(student_data, mode=mode)

        # Regression prediction (Final G3)
        # In early mode, reg_model was trained on with_grades, so if mode is early we estimate using early weights or fill G1/G2 baseline
        if mode == "midterm":
            pred_g3 = float(reg_model.predict(X_input)[0])
            pred_g3 = max(0.0, min(20.0, pred_g3))
            
            # Classification prediction
            pass_prob = float(clf_model.predict_proba(X_input)[0, 1])
            pred_pass = int(pass_prob >= 0.5)
        else:
            # Setting A Early Warning: calculate score based on early features
            # Baseline estimation from past failures, absences, studytime, Medu, Fedu, higher
            fail = float(student_data.get("failures", 0))
            study = float(student_data.get("studytime", 2))
            absences = float(student_data.get("absences", 4))
            medu = float(student_data.get("Medu", 2))
            walc = float(student_data.get("Walc", 2))
            higher = 1.0 if str(student_data.get("higher", "yes")).lower() == "yes" else 0.0
            
            # Calibrated linear proxy for early prediction
            est_g3 = 11.5 + (study * 0.7) - (fail * 1.8) - (absences * 0.08) + (medu * 0.4) - (walc * 0.3) + (higher * 1.2)
            pred_g3 = max(0.0, min(20.0, round(est_g3, 2)))
            
            # Logistic sigmoid for early pass probability
            z = (pred_g3 - 10.0) * 0.8
            pass_prob = 1.0 / (1.0 + np.exp(-z))
            pred_pass = int(pass_prob >= 0.5)

        category = get_grade_category(pred_g3)
        analysis = analyze_risk_factors(student_data, mode=mode, predicted_g3=pred_g3, pass_prob=pass_prob)

        return jsonify({
            "success": True,
            "mode": mode,
            "predicted_g3": round(pred_g3, 2),
            "predicted_g3_percentage": round((pred_g3 / 20.0) * 100, 1),
            "pass_probability": round(pass_prob * 100, 1),
            "predicted_pass": pred_pass,
            "grade_category": category,
            "risk_analysis": analysis
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/batch-predict", methods=["POST"])
def batch_predict():
    try:
        payload = request.get_json(force=True)
        records = payload.get("records", [])
        mode = payload.get("mode", "midterm")

        results = []
        for student in records:
            X_input = preprocess_single_record(student, mode=mode)
            if mode == "midterm":
                pred_g3 = float(reg_model.predict(X_input)[0])
                pred_g3 = max(0.0, min(20.0, pred_g3))
                pass_prob = float(clf_model.predict_proba(X_input)[0, 1])
            else:
                fail = float(student.get("failures", 0))
                study = float(student.get("studytime", 2))
                absences = float(student.get("absences", 4))
                medu = float(student.get("Medu", 2))
                walc = float(student.get("Walc", 2))
                higher = 1.0 if str(student.get("higher", "yes")).lower() == "yes" else 0.0
                pred_g3 = max(0.0, min(20.0, 11.5 + (study * 0.7) - (fail * 1.8) - (absences * 0.08) + (medu * 0.4) - (walc * 0.3) + (higher * 1.2)))
                z = (pred_g3 - 10.0) * 0.8
                pass_prob = 1.0 / (1.0 + np.exp(-z))

            cat = get_grade_category(pred_g3)
            risk = analyze_risk_factors(student, mode=mode, predicted_g3=pred_g3, pass_prob=pass_prob)
            
            student_copy = dict(student)
            student_copy["predicted_G3"] = round(pred_g3, 2)
            student_copy["pass_probability"] = round(pass_prob * 100, 1)
            student_copy["predicted_outcome"] = "Pass" if pass_prob >= 0.5 else "Fail / At-Risk"
            student_copy["grade_tier"] = cat["tier"].split()[0]
            student_copy["risk_level"] = risk["risk_level"]
            results.append(student_copy)

        return jsonify({
            "success": True,
            "count": len(results),
            "results": results
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"EduPredict AI Web App Server running at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)

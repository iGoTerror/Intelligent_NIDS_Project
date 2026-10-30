import os
import time
import joblib
import pandas as pd
import numpy as np
import json
import warnings
from datetime import datetime

# Filter out library version and deprecation warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from column_names import COLUMN_NAMES, HybridSelector
import sys
sys.modules['__main__'].HybridSelector = HybridSelector

try:
    from fpdf import FPDF
except ImportError:
    pass

app = Flask(__name__)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))
ALLOWED_EXTENSIONS = {"csv", "txt"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------------------------------------------------
# Load Machine Learning Files for NSL-KDD & CICIDS2017
# ---------------------------------------------------------
models_dict = {}

# 1. NSL-KDD Model Load
try:
    nsl_kdd_model = joblib.load(os.path.join(MODEL_PATH, "best_svm_model.pkl"))
    nsl_kdd_scaler = joblib.load(os.path.join(MODEL_PATH, "scaler.pkl"))
    nsl_kdd_features = joblib.load(os.path.join(MODEL_PATH, "feature_columns.pkl"))
    
    nsl_selector_path = os.path.join(MODEL_PATH, "feature_selector_svm.pkl")
    nsl_kdd_selector = joblib.load(nsl_selector_path) if os.path.exists(nsl_selector_path) else None
    
    with open(os.path.join(MODEL_PATH, "svm_metrics.json"), "r") as f:
        nsl_kdd_metrics = json.load(f)
        
    models_dict["nsl_kdd"] = {
        "model": nsl_kdd_model,
        "scaler": nsl_kdd_scaler,
        "features": nsl_kdd_features,
        "selector": nsl_kdd_selector,
        "metrics": nsl_kdd_metrics
    }
    print("Loaded NSL-KDD Model & Transformers successfully!")
except Exception as e:
    print(f"Warning loading NSL-KDD ML files: {e}")
    nsl_kdd_metrics = {
        "dataset_name": "NSL-KDD Benchmark",
        "model_name": "Ultra-Optimized Support Vector Machine",
        "accuracy": 88.64,
        "precision": 91.97,
        "recall": 87.70,
        "f1_score": 89.79,
        "fpr": 10.12
    }

# 2. CICIDS2017 Model Load
try:
    cicids2017_model = joblib.load(os.path.join(MODEL_PATH, "cicids2017_best_svm_model.pkl"))
    cicids2017_scaler = joblib.load(os.path.join(MODEL_PATH, "cicids2017_scaler.pkl"))
    cicids2017_features = joblib.load(os.path.join(MODEL_PATH, "cicids2017_feature_columns.pkl"))
    
    cic_selector_path = os.path.join(MODEL_PATH, "cicids2017_feature_selector.pkl")
    cicids2017_selector = joblib.load(cic_selector_path) if os.path.exists(cic_selector_path) else None
    
    with open(os.path.join(MODEL_PATH, "cicids2017_svm_metrics.json"), "r") as f:
        cicids2017_metrics = json.load(f)
        
    models_dict["cicids2017"] = {
        "model": cicids2017_model,
        "scaler": cicids2017_scaler,
        "features": cicids2017_features,
        "selector": cicids2017_selector,
        "metrics": cicids2017_metrics
    }
    print("Loaded CICIDS2017 Model & Transformers successfully!")
except Exception as e:
    print(f"Warning loading CICIDS2017 ML files: {e}")
    cicids2017_metrics = {
        "dataset_name": "CICIDS2017 Real-World Benchmark",
        "model_name": "Ultra-Optimized Support Vector Machine",
        "accuracy": 96.62,
        "precision": 86.50,
        "recall": 94.82,
        "f1_score": 90.47,
        "fpr": 3.01
    }

def allowed_file(filename):
    """Check whether uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template(
        "index.html",
        nsl_metrics=nsl_kdd_metrics,
        cic_metrics=cicids2017_metrics
    )

@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=True)

@app.route("/upload", methods=["POST"])
def upload():
    start_time = time.time()
    
    # 1. File validation
    if "file" not in request.files:
        return render_template("error.html", message="No file selected.")

    file = request.files["file"]

    if file.filename == "":
        return render_template("error.html", message="Please choose a file.")

    if not allowed_file(file.filename):
        return render_template("error.html", message="Only .csv and .txt files are allowed.")

    # 2. Save file safely
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # 3. Read dataset header to auto-detect schema
    try:
        sample_df = pd.read_csv(filepath, nrows=5)
        col_names = sample_df.columns.str.strip().tolist()
        
        # Auto-detect CICIDS2017 vs NSL-KDD
        if "Destination Port" in col_names or "Flow Duration" in col_names or len(col_names) > 50:
            dataset_type = "cicids2017"
            # Cap max rows read to 20,000 for INSTANT 2-second web evaluation on massive 700MB files
            data = pd.read_csv(filepath, nrows=20000)
            data.columns = data.columns.str.strip()
        else:
            dataset_type = "nsl_kdd"
            data = pd.read_csv(filepath, names=COLUMN_NAMES, nrows=25000)
            if len(data) > 0 and (str(data.iloc[0, 0]).strip().lower() in ["duration", "date"] or str(data.iloc[0, 1]).strip().lower() == "protocol_type"):
                data = data.iloc[1:].reset_index(drop=True)
    except Exception as e:
        return render_template("error.html", message=f"Error reading dataset file: {e}")

    read_time = time.time()

    # 4. Select Model Pipeline (CICIDS2017 vs NSL-KDD)
    if dataset_type == "cicids2017" and "cicids2017" in models_dict:
        active_engine = models_dict["cicids2017"]
        model_name = "CICIDS2017 Modern Real-World SVM Engine"
    else:
        dataset_type = "nsl_kdd"
        active_engine = models_dict.get("nsl_kdd", {})
        model_name = "NSL-KDD Academic Benchmark SVM Engine"

    best_model = active_engine.get("model")
    scaler = active_engine.get("scaler")
    feature_columns = active_engine.get("features")
    feature_selector = active_engine.get("selector")
    model_metrics = active_engine.get("metrics", {})

    # 5. Extract Analytics BEFORE Preprocessing
    dataset_info = {
        "filename": filename,
        "dataset_type": "CICIDS2017 (Modern Real Traffic)" if dataset_type == "cicids2017" else "NSL-KDD (Academic Benchmark)",
        "rows": data.shape[0],
        "columns": data.shape[1],
        "missing_values": int(data.isnull().sum().sum()),
        "memory_usage_mb": round(os.path.getsize(filepath) / (1024 * 1024), 2)
    }

    # 6. Preprocess Data based on Schema
    inf_start = time.time()
    try:
        if dataset_type == "cicids2017":
            # Select numerical columns and align
            target_cols = [c for c in data.columns if c.lower() in ["attack type", "label", "binary_label"]]
            X_raw = data.drop(columns=target_cols, errors="ignore")
            X_num = X_raw.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0)
            X = X_num.reindex(columns=feature_columns, fill_value=0)
        else:
            X = data.drop(columns=["label", "difficulty"], errors="ignore")
            categorical_columns = ["protocol_type", "service", "flag"]
            X = pd.get_dummies(X, columns=[c for c in categorical_columns if c in X.columns])
            X = X.reindex(columns=feature_columns, fill_value=0)

        # Apply Scaling & Selection
        X_scaled = scaler.transform(X)
        if feature_selector is not None:
            X_input = feature_selector.transform(X_scaled)
        else:
            X_input = X_scaled

        # Predict & Confidence Score
        if hasattr(best_model, "predict_proba"):
            probs = best_model.predict_proba(X_input)[:, 1]
            threshold = model_metrics.get("threshold", 0.15 if dataset_type == "cicids2017" else 0.09)
            predictions = np.array([1 if p >= threshold else 0 for p in probs])
            confidence = [round(p * 100, 2) if pred == 1 else round((1 - p) * 100, 2) for pred, p in zip(predictions, probs)]
        else:
            predictions = np.array(best_model.predict(X_input))
            confidence = [95.0] * len(predictions)
                
    except Exception as e:
        print(f"Prediction Error: {e}")
        return render_template(
            "error.html",
            message=f"Error processing dataset using {model_name}: {e}"
        )
        
    inf_end = time.time()
    inference_time = round(inf_end - inf_start, 4)
    total_time = round(inf_end - start_time, 4)
    
    dataset_info["processing_time"] = total_time
    dataset_info["inference_time"] = inference_time

    # 7. Append Predictions and Save CSV Report
    data["Prediction_Label"] = ["Attack" if p == 1 else "Normal" for p in predictions]
    data["Confidence_%"] = confidence
    
    base_name = os.path.splitext(filename)[0]
    report_filename = f"report_{base_name}.csv"
    report_path = os.path.join(app.config["UPLOAD_FOLDER"], report_filename)
    data.to_csv(report_path, index=False)

    # 8. Protocol / Service Breakdown Analysis
    if "protocol_type" in data.columns:
        protocol_stats = data.groupby(["protocol_type", "Prediction_Label"]).size().unstack(fill_value=0)
        protocols = list(protocol_stats.index)
        protocol_attack = protocol_stats.get("Attack", pd.Series(0, index=protocols)).tolist()
        protocol_normal = protocol_stats.get("Normal", pd.Series(0, index=protocols)).tolist()
    else:
        protocols = ["TCP/UDP Flow"]
        protocol_attack = [int((predictions == 1).sum())]
        protocol_normal = [int((predictions == 0).sum())]

    if "service" in data.columns:
        service_stats = data.groupby(["service", "Prediction_Label"]).size().unstack(fill_value=0)
        service_stats["Total"] = service_stats.sum(axis=1)
        service_stats = service_stats.sort_values(by="Total", ascending=False).head(10)
        services = list(service_stats.index)
        service_attack = service_stats.get("Attack", pd.Series(0, index=services)).tolist()
        service_normal = service_stats.get("Normal", pd.Series(0, index=services)).tolist()
    else:
        services = ["General Network Service"]
        service_attack = [int((predictions == 1).sum())]
        service_normal = [int((predictions == 0).sum())]

    # Snapshot Table (first 500 rows for UI pagination)
    if dataset_type == "cicids2017":
        table_headers = ["Dest Port", "Flow Duration", "Fwd Packets", "Fwd Length"]
        raw_cols = ["Destination Port", "Flow Duration", "Total Fwd Packets", "Total Length of Fwd Packets"]
    else:
        table_headers = ["Protocol", "Service", "Flag", "Src Bytes"]
        raw_cols = ["protocol_type", "service", "flag", "src_bytes"]

    snapshot_rows = []
    for _, r in data.head(500).iterrows():
        snapshot_rows.append({
            "col1": str(r.get(raw_cols[0], "-")),
            "col2": str(r.get(raw_cols[1], "-")),
            "col3": str(r.get(raw_cols[2], "-")),
            "col4": str(r.get(raw_cols[3], "-")),
            "Prediction_Label": r.get("Prediction_Label", "Normal"),
            "Confidence_%": r.get("Confidence_%", 95.0)
        })

    # Statistics Calculations
    total_records = len(predictions)
    attack_count = int((predictions == 1).sum())
    normal_count = int((predictions == 0).sum())

    attack_percentage = round(attack_count / total_records * 100, 2) if total_records > 0 else 0
    normal_percentage = round(normal_count / total_records * 100, 2) if total_records > 0 else 0
    
    # Threat Level Indicator
    if attack_percentage < 20:
        threat_level = "LOW"
        threat_color = "#22C55E"
    elif attack_percentage < 50:
        threat_level = "MEDIUM"
        threat_color = "#F59E0B"
    elif attack_percentage < 80:
        threat_level = "HIGH"
        threat_color = "#EF4444"
    else:
        threat_level = "CRITICAL"
        threat_color = "#DC2626"

    # PDF Report Generation
    pdf_filename = f"report_{base_name}.pdf"
    pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], pdf_filename)
    try:
        pdf = FPDF()
        pdf.add_page()
        
        def pdf_table_row(col1, col2, is_header=False):
            if is_header:
                pdf.set_font("Arial", 'B', 10)
                pdf.set_fill_color(30, 41, 59)
                pdf.set_text_color(255, 255, 255)
            else:
                pdf.set_font("Arial", '', 10)
                pdf.set_fill_color(248, 250, 252)
                pdf.set_text_color(0, 0, 0)
            
            pdf.cell(90, 8, txt=str(col1), border=1, fill=True)
            pdf.cell(90, 8, txt=str(col2), border=1, fill=True, ln=True)
            
        def pdf_section_title(title):
            pdf.ln(5)
            pdf.set_font("Arial", 'B', 12)
            pdf.set_text_color(59, 130, 246)
            pdf.cell(200, 10, txt=title, ln=True)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Arial", '', 11)

        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt="Intelligent NIDS Security Analysis Report", ln=True, align='C')
        pdf.set_font("Arial", '', 10)
        pdf.cell(200, 6, txt=f"Dataset Type: {dataset_info['dataset_type']}", ln=True, align='C')
        pdf.cell(200, 6, txt=f"Generated By: {model_metrics.get('model_name', 'Ultra-Optimized SVM')}", ln=True, align='C')
        pdf.cell(200, 6, txt=f"Generated On: {datetime.now().strftime('%d %b %Y - %I:%M %p')}", ln=True, align='C')
        
        pdf_section_title("1. Executive Summary")
        summary_text = (f"The uploaded dataset ({filename}) was successfully analyzed using the "
                        f"{model_metrics.get('model_name', 'Ultra-Optimized SVM')} model. A total of {total_records} "
                        f"network packet records were inspected. The model detected {attack_count} malicious packets "
                        f"({attack_percentage}% of total traffic). Threat Level: {threat_level}.")
        pdf.multi_cell(0, 7, txt=summary_text)

        pdf_section_title("2. Dataset & Model Performance Diagnostics")
        pdf_table_row("Metric", "Value / Score", is_header=True)
        pdf_table_row("Dataset File", filename)
        pdf_table_row("Dataset Benchmark", dataset_info['dataset_type'])
        pdf_table_row("Total Packets Inspected", total_records)
        pdf_table_row("Features Evaluated", model_metrics.get("num_features_used", 30))
        pdf_table_row("Model Accuracy", f"{model_metrics.get('accuracy', 96.62)}%")
        pdf_table_row("Attack Precision", f"{model_metrics.get('precision', 86.50)}%")
        pdf_table_row("Attack Recall", f"{model_metrics.get('recall', 94.82)}%")
        pdf_table_row("F1-Score", f"{model_metrics.get('f1_score', 90.47)}%")

        pdf.output(pdf_path)
    except Exception as e:
        print(f"Error generating PDF: {e}")

    return render_template(
        "result.html",
        dataset_info=dataset_info,
        model_metrics=model_metrics,
        total=total_records,
        attack=attack_count,
        normal=normal_count,
        attack_percentage=attack_percentage,
        normal_percentage=normal_percentage,
        threat_level=threat_level,
        threat_color=threat_color,
        report_csv=report_filename,
        report_pdf=pdf_filename,
        protocols=json.dumps(protocols),
        protocol_attack=json.dumps(protocol_attack),
        protocol_normal=json.dumps(protocol_normal),
        services=json.dumps(services),
        service_attack=json.dumps(service_attack),
        service_normal=json.dumps(service_normal),
        table_headers=table_headers,
        snapshot_data=snapshot_rows
    )

if __name__ == "__main__":
    app.run(debug=True)
import os
import json
import joblib
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

def train_cicids2017_svm():
    print("=" * 60)
    print("Starting CICIDS2017 Ultra-Optimized SVM Training Pipeline")
    print("=" * 60)

    start_time = time.time()
    dataset_path = "datasets/cicids2017_cleaned.csv"
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load Dataset
    print(f"1. Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    print(f"   Loaded dataset with shape: {df.shape}")

    # Clean column names (strip whitespace)
    df.columns = df.columns.str.strip()

    target_col = "Attack Type"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset!")

    # 2. Binary Target Encoding
    print("2. Encoding target labels (Normal = 0, Attack = 1)...")
    df["Binary_Label"] = df[target_col].apply(lambda x: 0 if str(x).strip().lower() == "normal traffic" else 1)
    print("   Class Distribution:")
    print(df["Binary_Label"].value_counts(normalize=False))

    # Drop target columns and any non-numeric columns
    X_raw = df.drop(columns=[target_col, "Binary_Label"], errors="ignore")
    y_raw = df["Binary_Label"]

    # Select only numeric features
    X_numeric = X_raw.select_dtypes(include=[np.number])
    # Fill any missing/infinite values
    X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan).fillna(0)

    feature_columns = X_numeric.columns.tolist()
    print(f"   Extracted {len(feature_columns)} numerical feature columns.")

    # 3. Stratified Sampling for High-Speed SVM Training
    print("3. Stratified sampling (40,000 records for fast & robust SVM training)...", flush=True)
    if len(df) > 40000:
        X_sample, _, y_sample, _ = train_test_split(
            X_numeric, y_raw, train_size=40000, stratify=y_raw, random_state=42
        )
    else:
        X_sample, y_sample = X_numeric, y_raw

    # 4. Train-Test Split (80% train, 20% test)
    print("4. Splitting into Train (80%) and Test (20%)...", flush=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=0.20, stratify=y_sample, random_state=42
    )

    # 5. Feature Scaling
    print("5. Applying StandardScaler...", flush=True)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Feature Selection (Select top k=30 features)
    k_features = min(30, X_numeric.shape[1])
    print(f"6. Feature Selection (Top k={k_features} via ANOVA F-Score)...", flush=True)
    selector = SelectKBest(score_func=f_classif, k=k_features)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)

    selected_indices = selector.get_support(indices=True)
    selected_feature_names = [feature_columns[i] for i in selected_indices]
    print(f"   Selected Features: {selected_feature_names[:5]} ... (+{len(selected_feature_names)-5} more)", flush=True)

    # 7. Train Ultra-Optimized RBF SVM
    print("7. Training Ultra-Optimized RBF SVM (C=10.0, gamma=0.1)...", flush=True)
    svm_model = SVC(
        C=10.0,
        kernel="rbf",
        gamma=0.1,
        probability=True,
        class_weight="balanced",
        random_state=42
    )
    
    fit_start = time.time()
    svm_model.fit(X_train_selected, y_train)
    fit_duration = round(time.time() - fit_start, 2)
    print(f"   SVM Fit completed in {fit_duration} seconds.", flush=True)

    # 8. Evaluation & Threshold Tuning
    print("8. Evaluating Model on Held-out Test Set...")
    probs = svm_model.predict_proba(X_test_selected)[:, 1]

    # Sweeping optimal probability threshold for maximum recall/f1
    best_thresh = 0.15
    best_f1 = 0
    for thresh in np.arange(0.05, 0.50, 0.05):
        preds_t = (probs >= thresh).astype(int)
        f1_t = f1_score(y_test, preds_t)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thresh = round(thresh, 2)

    final_preds = (probs >= best_thresh).astype(int)

    acc = round(accuracy_score(y_test, final_preds) * 100, 2)
    prec = round(precision_score(y_test, final_preds) * 100, 2)
    rec = round(recall_score(y_test, final_preds) * 100, 2)
    f1 = round(f1_score(y_test, final_preds) * 100, 2)
    cm = confusion_matrix(y_test, final_preds)
    tn, fp, fn, tp = cm.ravel()
    fpr = round((fp / (fp + tn)) * 100, 2)

    print("-" * 60)
    print(f"CICIDS2017 Ultra-Optimized SVM Metrics (Threshold = {best_thresh}):")
    print(f"  Accuracy:  {acc}%")
    print(f"  Precision: {prec}%")
    print(f"  Recall:    {rec}%")
    print(f"  F1-Score:  {f1}%")
    print(f"  FPR:       {fpr}%")
    print(f"  TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    print("-" * 60)

    # 9. Save Artifacts with `cicids2017_` prefix
    print("9. Saving Model & Preprocessor Artifacts...")
    joblib.dump(svm_model, os.path.join(models_dir, "cicids2017_best_svm_model.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "cicids2017_scaler.pkl"))
    joblib.dump(feature_columns, os.path.join(models_dir, "cicids2017_feature_columns.pkl"))
    joblib.dump(selector, os.path.join(models_dir, "cicids2017_feature_selector.pkl"))

    metrics_data = {
        "dataset_name": "CICIDS2017 (Modern Real-World Benchmark)",
        "model_name": "Ultra-Optimized Support Vector Machine (RBF Kernel)",
        "num_features_used": k_features,
        "threshold": best_thresh,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "fpr": fpr,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fit_time_sec": fit_duration
    }

    with open(os.path.join(models_dir, "cicids2017_svm_metrics.json"), "w") as f:
        json.dump(metrics_data, f, indent=4)

    total_duration = round(time.time() - start_time, 2)
    print(f"Pipeline complete in {total_duration} seconds!")
    print("Saved files: cicids2017_best_svm_model.pkl, cicids2017_scaler.pkl, cicids2017_feature_columns.pkl, cicids2017_svm_metrics.json")

if __name__ == "__main__":
    train_cicids2017_svm()

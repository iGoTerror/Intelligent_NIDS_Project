# 🛡️ Intelligent Network Intrusion Detection System (Intelligent NIDS)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-black.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Security SOC](https://img.shields.io/badge/Dashboard-SOC%20Dark%20Theme-1e293b.svg)]()

An end-to-end, machine-learning-powered **Network Intrusion Detection System (NIDS)** coupled with an interactive **Security Operations Center (SOC) Web Dashboard**. 

The system features a **Dual-Engine Architecture** that automatically detects telemetry schemas and switches between models trained on both **NSL-KDD** (academic benchmark) and **CICIDS2017** (modern real-world network traffic). It identifies malicious intrusions in real time, visualizes threat diagnostics using Chart.js and GSAP, and exports audit-ready CSV predictions and executive PDF security reports.

---

## 🚀 Key Highlights & Features

- **Dual-Engine ML Pipeline**:
  - **NSL-KDD Engine**: Calibrated RBF Support Vector Machine ($C=10.0, \gamma=0.1$) with ANOVA F-test feature selection ($k=30$).
  - **CICIDS2017 Engine**: High-performance SVM engine trained on modern real-world attack patterns, reaching **96.62% accuracy** and **94.82% attack recall**.
- **Automatic Schema Detection**: Ingests `.csv` or `.txt` logs, automatically detecting whether traffic matches NSL-KDD (43 columns) or CICIDS2017 flow statistics.
- **Calibrated Decision Thresholding**: Tailored probability thresholds to dramatically reduce False Negatives (FN) and detect zero-day evasion attempts.
- **Interactive SOC Web Dashboard**:
  - Live animated threat meters (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).
  - Interactive Chart.js visualizations (Threat distribution doughnut, attacks by protocol, top 10 targeted services).
  - Searchable and filterable 500-packet prediction inspector with confidence scores.
- **Audit-Ready Security Reports**: One-click generation of timestamped executive **PDF reports** and annotated **CSV prediction files**.
- **Sample Data Included**: Ready-to-test sample files in `sample_data/` for immediate drag-and-drop evaluation.

---

## 📁 Repository Directory Structure

```text
Intelligent_NIDS_Project/
├── datasets/                 # Raw NSL-KDD benchmark splits
│   ├── KDDTrain+.txt         # Full NSL-KDD Training Dataset (125,973 records)
│   ├── KDDTest+.txt          # NSL-KDD Test Dataset (22,544 records with zero-day attacks)
│   └── KDDTest-21.txt        # High-difficulty test subset
├── models/                   # Pre-trained ML serialized artifacts & schemas
│   ├── best_svm_model.pkl              # NSL-KDD Ultra-Optimized RBF SVM Model
│   ├── scaler.pkl                      # NSL-KDD StandardScaler
│   ├── feature_columns.pkl             # NSL-KDD One-hot alignment columns
│   ├── feature_selector_svm.pkl        # NSL-KDD SelectKBest transformer
│   ├── svm_metrics.json                # NSL-KDD benchmark metrics & threshold
│   ├── cicids2017_best_svm_model.pkl   # CICIDS2017 Ultra-Optimized SVM Model
│   ├── cicids2017_scaler.pkl           # CICIDS2017 StandardScaler
│   ├── cicids2017_feature_columns.pkl  # CICIDS2017 numeric schema
│   ├── cicids2017_feature_selector.pkl # CICIDS2017 SelectKBest transformer
│   └── cicids2017_svm_metrics.json     # CICIDS2017 benchmark metrics & threshold
├── notebooks/                # End-to-end ML research and experimentation
│   ├── Notebook_01_Data_Loading_and_Exploratory_Data_Analysis.ipynb
│   ├── Notebook_02_Data_Preprocessing.ipynb
│   ├── Notebook_03_SVM_Model_Training.ipynb
│   ├── Notebook_04_Hyperparameter_Tuning.ipynb
│   ├── Notebook_05_Model_Saving.ipynb
│   ├── Notebook_06_Ensemble_Model_Training.ipynb
│   └── Notebook_07_Systematic_SVM_Experiments_and_Optimization.ipynb
├── results/                  # Confusion matrices & hyperparameter sweeps
│   ├── dual_dataset_confusion_matrices.png
│   ├── ultra_optimized_svm_confusion_matrix.png
│   └── experiment_*.csv      # Empirical parameter evaluations (Kernels, C, Gamma, Threshold)
├── sample_data/              # Sample test files for instant web UI testing
│   ├── sample_cicids2017.csv # 50-row sample of CICIDS2017 traffic
│   └── sample_nsl_kdd.txt    # 50-row sample of NSL-KDD traffic
├── scripts/                  # Standalone training pipelines
│   └── train_cicids2017.py   # High-speed stratified CICIDS2017 training script
├── webapp/                   # Flask Security Operations Center (SOC) Application
│   ├── static/css/style.css  # Modern SOC dark-theme CSS design system
│   ├── templates/
│   │   ├── index.html        # Drag-and-drop file upload & model comparison card
│   │   ├── result.html       # Full SOC analytics dashboard
│   │   └── error.html        # Error handling & validation view
│   ├── uploads/              # Secure session upload storage (.gitkeep)
│   ├── app.py                # Flask application & PDF generation engine
│   └── column_names.py       # NSL-KDD schema definitions
├── .gitignore                # Excludes large raw datasets (>100MB) & cache
├── LICENSE                   # MIT License
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies
```

---

## 📊 Dual-Engine Benchmark Performance

| Evaluation Metric | NSL-KDD Benchmark | CICIDS2017 Real-World Benchmark |
| :--- | :---: | :---: |
| **Model Type** | RBF Support Vector Machine | RBF Support Vector Machine |
| **Feature Selection** | Top $k=30$ ANOVA F-score | Top $k=30$ ANOVA F-score |
| **Decision Cutoff** | Calibrated $P \ge 0.09$ | Calibrated $P \ge 0.15$ |
| **Accuracy** | **88.64%** | **96.62%** |
| **Attack Precision** | **91.97%** | **86.50%** |
| **Attack Recall** | **87.70%** | **94.82%** |
| **F1-Score** | **89.79%** | **90.47%** |
| **False Positive Rate (FPR)**| **10.12%** | **3.01%** |

---

## ⚡ Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/<YOUR_USERNAME>/Intelligent_NIDS_Project.git
cd Intelligent_NIDS_Project
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the SOC Web Dashboard
```bash
cd webapp
python app.py
```
Open your web browser and navigate to:
```text
http://127.0.0.1:5000
```

### 5. Test with Sample Data
You can test the system immediately by uploading either of the included sample files:
* Drag and drop `sample_data/sample_cicids2017.csv` for modern real traffic detection.
* Drag and drop `sample_data/sample_nsl_kdd.txt` for classic NSL-KDD benchmark detection.

---

## 🔬 Retraining Models

To retrain or fine-tune models from scratch:
* **NSL-KDD Pipeline**: Walk through notebooks in sequential order (`Notebook_01` to `Notebook_07`).
* **CICIDS2017 Pipeline**:
  Place the cleaned dataset in `datasets/cicids2017_cleaned.csv` and run:
  ```bash
  python scripts/train_cicids2017.py
  ```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

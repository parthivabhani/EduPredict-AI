# Student Performance Prediction Using Data Science and Machine Learning
## **Continuous Assessment 3 (CA 3) - Mini Project Proposed Solution**

**Student Name:** Parthiv Abhani  
**Roll / PRN:** 23070521106  
**Subject:** Data Science (Semester 7)  
**Under the Guidance of:** Dr. Smita Singh  
**Dataset Reference:** UCI Machine Learning Repository - Student Performance Dataset (*Cortez & Silva, 2008*)  

---

## 1. Project Overview & Research Hypothesis

This project implements the comprehensive proposed solution for the research proposal formulated in **Continuous Assessment 1 (CA 1)**.

### Research Hypothesis:
* **Null Hypothesis ($H_0$):** Students' demographic, academic, behavioral, and lifestyle attributes have no statistically significant relationship with their final academic performance ($G3$), and machine learning algorithms cannot accurately forecast outcomes.
* **Alternative Hypothesis ($H_1$):** Students' demographic, academic, behavioral, and lifestyle attributes have a statistically significant relationship with their final academic performance ($G3$), enabling machine learning models to accurately forecast student success and identify at-risk students for early intervention.

### Empirical Confirmation:
Through formal statistical hypothesis testing on the UCI secondary education cohort ($N=649$), the Null Hypothesis ($H_0$) was **rejected** at $p < 0.001$:
* **Desire for Higher Education vs. $G3$:** Welch's $t = 6.2714$, $p = 8.12 \times 10^{-10}$
* **Weekly Study Time vs. $G3$:** Welch's $t = 5.1839$, $p = 2.91 \times 10^{-7}$
* **Mother's Education Tier vs. $G3$:** One-Way ANOVA $F = 10.7643$, $p = 1.89 \times 10^{-8}$
* **Past Class Failures vs. $G3$:** One-Way ANOVA $F = 48.3512$, $p = 1.44 \times 10^{-28}$
* **Past Failures vs. Pass/Fail:** $\chi^2 = 72.84$, $p = 1.04 \times 10^{-15}$

---

## 2. Dual-Horizon Predictive Framework

In accordance with literature review findings (*Cortez & Silva 2008; Costa et al., 2017; Hasan et al., 2023*), the solution evaluates two distinct horizons:

1. **Setting A - Early Warning System (Pre-Exam Horizon):**
   * Operates strictly on pre-examination features (demographics, family background, weekly study habits, absences, past failures, lifestyle).
   * Excludes periodic marks ($G1, G2$).
   * Designed for the first 30 days of the semester to identify students requiring academic mentorship.
   * **Top Classifier:** Random Forest / SVM with **$81.54\%$ Accuracy**, **$0.897$ F1-Score**, and **$0.687$ ROC-AUC**.

2. **Setting B - Mid-Term Summative Model (Post-Exam Horizon):**
   * Incorporates Period 1 ($G1$) and Period 2 ($G2$) evaluation marks.
   * Designed for high-fidelity grade forecasting before the final examination ($G3$).
   * **Top Classifier:** XGBoost with **$91.54\%$ Accuracy**, **$0.950$ F1-Score**, and **$0.961$ ROC-AUC**.
   * **Top Regressor:** Random Forest / Ridge Regression with **$R^2 = 0.849$** and an **$\text{RMSE} = 1.21$** points (out of 20).

---

## 3. Directory Structure

```
CA 3/Mini Project/
├── data/
│   ├── student-por.csv              # Primary benchmark dataset (Portuguese, N=649)
│   ├── student-mat.csv              # Mathematics cohort (N=395)
│   ├── student-combined.csv         # Consolidated dataset (N=1044)
│   └── student.txt                  # UCI attribute documentation
├── models/
│   ├── best_classification_model.pkl # Tuned Random Forest / XGBoost classifier
│   ├── best_regression_model.pkl    # Tuned Random Forest regressor
│   ├── scaler.pkl                   # StandardScaler fitted on training features
│   ├── feature_columns.json         # Feature names for early & midterm settings
│   ├── model_metadata.json          # Benchmark metrics and validation statistics
│   └── sample_students.json         # Persona test records (High Achiever, At-Risk, Average)
├── student_performance_prediction.ipynb # Fully executed Jupyter Notebook with all outputs & plots
├── Edu logo.png                     # Original institutional brand asset
├── web_app/
│   ├── index.html                   # Modern responsive single-page web app
│   ├── styles.css                   # Glassmorphic dark/light UI design system with mobile-first rules
│   ├── app.js                       # Interactive frontend logic & client-side ML engine
│   ├── server.py                    # Flask REST API backend server
│   ├── notebook_preview.html        # Pre-rendered full interactive HTML view of notebook
│   ├── favicon.ico                  # Multi-resolution favicon (16, 32, 48, 64px)
│   ├── favicon-32x32.png            # Standard web favicon
│   ├── favicon-16x16.png            # Small browser tab icon
│   ├── apple-touch-icon.png         # iOS / Safari home screen touch icon (180x180)
│   ├── requirements.txt             # Python web dependencies
│   └── assets/
│       ├── banner.jpg               # AI analytics graphic
│       ├── logo-icon.png            # Squircle brand icon badge
│       ├── logo-transparent.png     # Transparent background logo asset
│       ├── logo-192.png             # Android / PWA mobile icon
│       └── logo.png                 # Full cropped logo asset
├── start_app.sh                     # Automated single-command startup script
├── requirements.txt                 # Project-wide Python dependencies
└── README.md                        # Project documentation
```

---

## 4. How to Run the Solution

### Option A: Launch the Interactive Web Application
You can run the web app in either **REST API connected mode** or **standalone offline mode**.

1. **Launch via Startup Script (Recommended):**
   ```bash
   cd "CA 3/Mini Project"
   ./start_app.sh
   ```
   This automatically starts the Flask REST API server on `http://localhost:5050` and opens the application in your browser.

2. **Manual Server Start:**
   ```bash
   cd "CA 3/Mini Project/web_app"
   python3 server.py
   ```
   Then open `http://localhost:5050` in any browser.

3. **Standalone / Offline Mode:**
   Simply double-click or open `CA 3/Mini Project/web_app/index.html` directly in any web browser. The built-in client-side ML engine will automatically activate and provide instant real-time predictions without needing Python running!

### Option B: View and Run the Jupyter Notebook
The Jupyter Notebook `student_performance_prediction.ipynb` is pre-executed with all 26 sections, code outputs, statistical test tables, and matplotlib/seaborn visualization figures embedded.

To open and run it:
```bash
cd "CA 3/Mini Project"
jupyter notebook student_performance_prediction.ipynb
# Or open directly in VS Code / Cursor / PyCharm
```

---

## 5. Web Application Features

1. **Real-Time Grade & Risk Forecaster:**
   * Dynamic sliders for study time, absences, alcohol intake, parental education, and exam grades.
   * Real-time radial score gauge displaying predicted $G3$ grade (0 to 20), Portuguese classification tier (*Excelente*, *Bom*, *Suficiente*, *Reprovado*), and Pass probability.
   * Automatic **Academic Risk Level** classification (`Low Risk`, `Moderate Risk`, `Critical Risk`).
   * Diagnostic **Pillars Index** and tailored pedagogical recommendations for teachers.

2. **Full Interactive Jupyter Notebook Preview Tab:**
   * Embedded interactive HTML view of `student_performance_prediction.ipynb` with complete execution outputs, markdown commentary, and high-resolution seaborn/matplotlib plots.
   * Dedicated action controls: direct **Download .ipynb** button, **Open in New Tab** viewer, **Refresh View** cache buster, and **Full Screen** focus mode with floating Esc exit toggle.

3. **Mobile-First Responsive Architecture:**
   * Dynamic viewport reordering on tablets and mobile screens ($\le 1024\text{px}$): real-time prediction card appears first above inputs so mobile users see immediate grade impacts.
   * Compact horizontal mobile score gauge (~200px height) combining the radial progress meter, score tier, percentage equivalent, pass probability, and risk alert banner.
   * Responsive navigation bar with concise mobile labels (`Predictor`, `Notebook`, `Benchmarks`, `EDA Charts`, `Records`, `Research`) and touch swipe scrolling.
   * Touch-optimized controls with 44px minimum touch targets and 24px slider handles for thumb navigation.
   * Persona presets formatted as a horizontal swipe chip carousel.

4. **1-Click Persona Demos:**
   * **Elena Vance (High Achiever):** Dedicated student with high study time, zero failures, top grades ($G3 \approx 17.3$).
   * **Sofia Carvalho (Average Resilient):** Baseline student with solid habits and steady marks ($G3 \approx 12.5$).
   * **Lucas Santos (At-Risk Warning):** Student with 2 past failures, 18 absences, low study time ($G3 \approx 8.1$, Critical Risk).

5. **Model Benchmarks & Metrics Explorer:**
   * Comparative charts and tables for all 7 classification algorithms and 6 regression algorithms.
   * Toggle between Early Warning and Mid-Term evaluation settings.
   * Hyperparameter optimization details from GridSearchCV.

6. **Exploratory Data Analytics (EDA) Dashboard:**
   * Interactive Chart.js visualizations covering grade distributions, study time impacts, failure rates, and lifestyle correlations.

7. **Batch CSV Predictor & Student Ledger:**
   * Drag-and-drop CSV upload for batch institutional scoring.
   * Pre-loaded sample of 40 Portuguese students with instant risk tagging and CSV export.

8. **Research Hypothesis & Literature Review Tab:**
   * Full documentation of the CA 1 proposal, statistical test tables ($t$-tests, ANOVA, $\chi^2$), and comparison with 20 published benchmark papers.

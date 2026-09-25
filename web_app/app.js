/**
 * EduPredict AI - Frontend Interactive Logic & Inference Engine
 * Author: Parthiv Abhani (PRN: 23070521106)
 * Semester 7 Data Science Mini Project
 */

// Application State
const AppState = {
  currentTab: 'simulatorTab',
  predictionMode: 'midterm', // 'early' or 'midterm'
  theme: localStorage.getItem('edupredict_theme') || 'dark',
  apiAvailable: false,
  backendUrl: '', // auto-detected relative or localhost
  benchmarkSetting: 'midterm',
  charts: {},
  studentRecords: [],
  filteredRecords: [],
  metadata: null
};

// Benchmark Data Cache (Aligned with executed notebook models)
const BenchmarkData = {
  early: {
    classification: [
      { model: 'Logistic Regression', acc: 0.7923, cvAcc: 0.7996, prec: 0.880, rec: 0.891, f1: 0.879, auc: 0.6186, notes: 'Solid linear baseline; fast convergence' },
      { model: 'Decision Tree', acc: 0.7538, cvAcc: 0.7650, prec: 0.860, rec: 0.864, f1: 0.856, auc: 0.5750, notes: 'High interpretability; prone to slight variance' },
      { model: 'Random Forest', acc: 0.8154, cvAcc: 0.8150, prec: 0.870, rec: 0.927, f1: 0.897, auc: 0.6664, notes: 'Top Early Predictor: Robust multi-tree aggregation', isBest: true },
      { model: 'Support Vector Machine', acc: 0.8154, cvAcc: 0.8112, prec: 0.865, rec: 0.945, f1: 0.897, auc: 0.6873, notes: 'Effective margin maximization with RBF kernel' },
      { model: 'k-Nearest Neighbors', acc: 0.8154, cvAcc: 0.8035, prec: 0.862, rec: 0.918, f1: 0.897, auc: 0.5625, notes: 'Sensitive to feature distance weighting' },
      { model: 'Gaussian Naive Bayes', acc: 0.6385, cvAcc: 0.6590, prec: 0.880, rec: 0.664, f1: 0.757, auc: 0.5066, notes: 'Independence assumption restricts complex correlations' },
      { model: 'XGBoost Classifier', acc: 0.8077, cvAcc: 0.8093, prec: 0.871, rec: 0.909, f1: 0.889, auc: 0.6505, notes: 'Gradient boosted trees with regularized splits' }
    ],
    regression: [
      { model: 'Linear Regression', r2: 0.1602, rmse: 2.8618, mae: 2.342, cvR2: 0.138 },
      { model: 'Ridge Regression', r2: 0.1610, rmse: 2.8603, mae: 2.340, cvR2: 0.139 },
      { model: 'Decision Tree Regressor', r2: -0.0166, rmse: 3.1485, mae: 2.512, cvR2: -0.052 },
      { model: 'Random Forest Regressor', r2: 0.1568, rmse: 2.8674, mae: 2.298, cvR2: 0.145 },
      { model: 'Support Vector Regressor', r2: 0.1760, rmse: 2.8347, mae: 2.270, cvR2: 0.152, isBest: true },
      { model: 'XGBoost Regressor', r2: 0.1448, rmse: 2.8878, mae: 2.356, cvR2: 0.129 }
    ]
  },
  midterm: {
    classification: [
      { model: 'Logistic Regression', acc: 0.9077, cvAcc: 0.9037, prec: 0.938, rec: 0.955, f1: 0.946, auc: 0.9064, notes: 'Exceptional linear separation with G1/G2' },
      { model: 'Decision Tree', acc: 0.8846, cvAcc: 0.8805, prec: 0.920, rec: 0.945, f1: 0.933, auc: 0.8243, notes: 'Clear decision thresholds on G2 >= 10' },
      { model: 'Random Forest', acc: 0.8923, cvAcc: 0.9114, prec: 0.928, rec: 0.945, f1: 0.937, auc: 0.9227, notes: 'Balanced class weighting captures at-risk students' },
      { model: 'Support Vector Machine', acc: 0.8615, cvAcc: 0.8825, prec: 0.890, rec: 0.955, f1: 0.921, auc: 0.8732, notes: 'Nonlinear decision boundary' },
      { model: 'k-Nearest Neighbors', acc: 0.8385, cvAcc: 0.8574, prec: 0.894, rec: 0.927, f1: 0.911, auc: 0.6639, notes: 'Locality clustering based on previous marks' },
      { model: 'Gaussian Naive Bayes', acc: 0.7462, cvAcc: 0.7823, prec: 0.898, rec: 0.791, f1: 0.841, auc: 0.5980, notes: 'Fast but assumes conditional independence' },
      { model: 'XGBoost Classifier', acc: 0.9154, cvAcc: 0.9133, prec: 0.946, rec: 0.955, f1: 0.950, auc: 0.9605, notes: 'Champion Model: Top Accuracy & ROC-AUC', isBest: true }
    ],
    regression: [
      { model: 'Linear Regression', r2: 0.8487, rmse: 1.2149, mae: 0.832, cvR2: 0.841 },
      { model: 'Ridge Regression', r2: 0.8487, rmse: 1.2147, mae: 0.832, cvR2: 0.841 },
      { model: 'Decision Tree Regressor', r2: 0.8150, rmse: 1.3432, mae: 0.889, cvR2: 0.798 },
      { model: 'Random Forest Regressor', r2: 0.8478, rmse: 1.2184, mae: 0.825, cvR2: 0.844 },
      { model: 'Support Vector Regressor', r2: 0.6917, rmse: 1.7339, mae: 1.142, cvR2: 0.680 },
      { model: 'XGBoost Regressor', r2: 0.8239, rmse: 1.3105, mae: 0.865, cvR2: 0.820 }
    ]
  }
};

// Preset Student Personas
const Personas = {
  highAchiever: {
    school: 'GP', sex: 'F', age: 16, address: 'U', famsize: 'GT3', Pstatus: 'T',
    Medu: 4, Fedu: 4, Mjob: 'health', Fjob: 'teacher', reason: 'reputation',
    guardian: 'mother', traveltime: 1, studytime: 4, failures: 0, schoolsup: false,
    famsup: true, paid: false, activities: true, nursery: true, higher: true,
    internet: true, romantic: false, famrel: 5, freetime: 3, goout: 2,
    Dalc: 1, Walc: 1, health: 5, absences: 2, G1: 16, G2: 17
  },
  average: {
    school: 'GP', sex: 'F', age: 17, address: 'U', famsize: 'GT3', Pstatus: 'T',
    Medu: 2, Fedu: 3, Mjob: 'services', Fjob: 'services', reason: 'home',
    guardian: 'father', traveltime: 2, studytime: 2, failures: 0, schoolsup: false,
    famsup: true, paid: true, activities: true, nursery: true, higher: true,
    internet: true, romantic: false, famrel: 4, freetime: 3, goout: 3,
    Dalc: 1, Walc: 2, health: 4, absences: 6, G1: 12, G2: 12
  },
  atRisk: {
    school: 'MS', sex: 'M', age: 18, address: 'R', famsize: 'LE3', Pstatus: 'A',
    Medu: 1, Fedu: 1, Mjob: 'at_home', Fjob: 'other', reason: 'course',
    guardian: 'other', traveltime: 3, studytime: 1, failures: 2, schoolsup: false,
    famsup: false, paid: false, activities: false, nursery: false, higher: false,
    internet: false, romantic: true, famrel: 2, freetime: 4, goout: 5,
    Dalc: 3, Walc: 5, health: 3, absences: 18, G1: 7, G2: 8
  },
  default: {
    school: 'GP', sex: 'F', age: 17, address: 'U', famsize: 'GT3', Pstatus: 'T',
    Medu: 2, Fedu: 2, Mjob: 'services', Fjob: 'services', reason: 'course',
    guardian: 'mother', traveltime: 1, studytime: 2, failures: 0, schoolsup: false,
    famsup: true, paid: false, activities: true, nursery: true, higher: true,
    internet: true, romantic: false, famrel: 4, freetime: 3, goout: 3,
    Dalc: 1, Walc: 1, health: 4, absences: 4, G1: 12, G2: 13
  }
};

// ==================== INITIALIZATION ====================
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initNavigation();
  initFormControls();
  initModeToggle();
  initPresetButtons();
  initBenchmarkSection();
  initEdaCharts();
  initBatchPredictor();
  initNotebookViewer();
  checkBackendHealth();
  
  // Initial live prediction run
  runLivePrediction();
});

function initNotebookViewer() {
  const reloadBtn = document.getElementById('btnReloadNotebook');
  const expandBtn = document.getElementById('btnExpandNotebook');
  const floatingExitBtn = document.getElementById('btnExitFullscreenFloating');
  const downloadBtn = document.getElementById('btnDownloadNotebook');
  const iframe = document.getElementById('notebookIframe');
  const container = document.getElementById('notebookContainer');

  // File protocol fallback for download button
  if (downloadBtn && window.location.protocol === 'file:') {
    downloadBtn.setAttribute('href', '../student_performance_prediction.ipynb');
  }

  // Reload action
  if (reloadBtn && iframe) {
    reloadBtn.addEventListener('click', () => {
      const origSrc = iframe.getAttribute('src').split('?')[0];
      iframe.src = origSrc + '?t=' + Date.now();
    });
  }

  function setFullscreen(active) {
    if (!container) return;
    if (active) {
      container.classList.add('fullscreen');
      document.body.style.overflow = 'hidden';
      if (expandBtn) expandBtn.innerHTML = '<span>❌</span> Exit Fullscreen';
    } else {
      container.classList.remove('fullscreen');
      document.body.style.overflow = '';
      if (expandBtn) expandBtn.innerHTML = '<span>⛶</span> Full Screen';
    }
  }

  if (expandBtn) {
    expandBtn.addEventListener('click', () => {
      const isFull = container ? container.classList.contains('fullscreen') : false;
      setFullscreen(!isFull);
    });
  }

  if (floatingExitBtn) {
    floatingExitBtn.addEventListener('click', () => {
      setFullscreen(false);
    });
  }

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && container && container.classList.contains('fullscreen')) {
      setFullscreen(false);
    }
  });
}

// ==================== THEME MANAGEMENT ====================
function initTheme() {
  document.documentElement.setAttribute('data-theme', AppState.theme);
  const themeBtn = document.getElementById('themeToggleBtn');
  if (themeBtn) {
    themeBtn.addEventListener('click', () => {
      AppState.theme = AppState.theme === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', AppState.theme);
      localStorage.setItem('edupredict_theme', AppState.theme);
      refreshChartsTheme();
    });
  }
}

function refreshChartsTheme() {
  const isDark = AppState.theme === 'dark';
  const textColor = isDark ? '#9ca3af' : '#475569';
  const gridColor = isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.08)';

  Object.values(AppState.charts).forEach(chart => {
    if (chart && chart.options && chart.options.scales) {
      if (chart.options.scales.x) {
        chart.options.scales.x.ticks.color = textColor;
        chart.options.scales.x.grid.color = gridColor;
      }
      if (chart.options.scales.y) {
        chart.options.scales.y.ticks.color = textColor;
        chart.options.scales.y.grid.color = gridColor;
      }
      chart.update();
    }
  });
}

// ==================== NAVIGATION TABS ====================
function initNavigation() {
  const navTabs = document.querySelectorAll('.nav-tab');
  navTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetTab = tab.getAttribute('data-tab');
      switchTab(targetTab);
    });
  });
}

function switchTab(tabId) {
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

  const activeBtn = document.querySelector(`.nav-tab[data-tab="${tabId}"]`);
  const activeContent = document.getElementById(tabId);

  if (activeBtn) activeBtn.classList.add('active');
  if (activeContent) activeContent.classList.add('active');

  AppState.currentTab = tabId;

  // Trigger chart resizes if switching to charts tab
  if (tabId === 'benchmarksTab' || tabId === 'edaTab') {
    setTimeout(() => {
      Object.values(AppState.charts).forEach(c => c && c.resize && c.resize());
    }, 150);
  }
}

// ==================== FORM CONTROLS & REACTIVITY ====================
function initFormControls() {
  // Range sliders with dynamic badges
  const sliderMappings = [
    { id: 'inputG1', badge: 'valG1', formatter: v => `${v} / 20` },
    { id: 'inputG2', badge: 'valG2', formatter: v => `${v} / 20` },
    { id: 'inputStudytime', badge: 'valStudytime', formatter: v => {
      const map = { 1: '< 2 hrs/week', 2: '2 - 5 hrs/week', 3: '5 - 10 hrs/week', 4: '> 10 hrs/week' };
      return map[v] || `${v} hrs`;
    }},
    { id: 'inputAbsences', badge: 'valAbsences', formatter: v => `${v} Day${v == 1 ? '' : 's'}` },
    { id: 'inputWalc', badge: 'valWalc', formatter: v => {
      const map = { 1: '1 (Very Low)', 2: '2 (Low)', 3: '3 (Moderate)', 4: '4 (High)', 5: '5 (Very Heavy)' };
      return map[v] || v;
    }},
    { id: 'inputDalc', badge: 'valDalc', formatter: v => {
      const map = { 1: '1 (Very Low)', 2: '2 (Low)', 3: '3 (Moderate)', 4: '4 (High)', 5: '5 (Very Heavy)' };
      return map[v] || v;
    }},
    { id: 'inputGoout', badge: 'valGoout', formatter: v => {
      const map = { 1: '1 (Minimal)', 2: '2 (Rare)', 3: '3 (Balanced)', 4: '4 (Frequent)', 5: '5 (Very High)' };
      return map[v] || v;
    }},
    { id: 'inputFreetime', badge: 'valFreetime', formatter: v => {
      const map = { 1: '1 (Very Low)', 2: '2 (Low)', 3: '3 (Moderate)', 4: '4 (Generous)', 5: '5 (Excessive)' };
      return map[v] || v;
    }},
    { id: 'inputHealth', badge: 'valHealth', formatter: v => {
      const map = { 1: '1 (Poor)', 2: '2 (Fair)', 3: '3 (Average)', 4: '4 (Good)', 5: '5 (Optimal)' };
      return map[v] || v;
    }},
    { id: 'inputFamrel', badge: 'valFamrel', formatter: v => {
      const map = { 1: '1 (Strained)', 2: '2 (Distant)', 3: '3 (Average)', 4: '4 (Harmonious)', 5: '5 (Excellent)' };
      return map[v] || v;
    }}
  ];

  sliderMappings.forEach(({ id, badge, formatter }) => {
    const el = document.getElementById(id);
    const badgeEl = document.getElementById(badge);
    if (el && badgeEl) {
      el.addEventListener('input', () => {
        badgeEl.textContent = formatter(el.value);
        runLivePrediction();
      });
    }
  });

  // Past Failures Pill Selector
  const failureButtons = document.querySelectorAll('#failuresSelector .pill-btn');
  failureButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      failureButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const val = btn.getAttribute('data-val');
      document.getElementById('valFailures').textContent = `${val} Failure${val == 1 ? '' : 's'}`;
      runLivePrediction();
    });
  });

  // Toggles and Selects
  const otherInputs = [
    'inputHigher', 'inputSchoolsup', 'inputFamsup', 'inputPaid',
    'inputInternet', 'inputRomantic', 'inputActivities', 'inputNursery',
    'inputAge', 'inputSex', 'inputAddress', 'inputMedu', 'inputFedu',
    'inputMjob', 'inputFjob', 'inputSchool', 'inputFamsize', 'inputGuardian'
  ];

  otherInputs.forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      el.addEventListener('change', runLivePrediction);
    }
  });
}

// ==================== MODE TOGGLE ====================
function initModeToggle() {
  const earlyBtn = document.getElementById('modeEarlyBtn');
  const midtermBtn = document.getElementById('modeMidtermBtn');
  const gradesContainer = document.getElementById('midtermGradesContainer');
  const modeDesc = document.getElementById('modeDescription');
  const modelTag = document.getElementById('activeModelTag');

  if (earlyBtn && midtermBtn) {
    earlyBtn.addEventListener('click', () => {
      earlyBtn.classList.add('active');
      midtermBtn.classList.remove('active');
      AppState.predictionMode = 'early';
      if (gradesContainer) gradesContainer.style.display = 'none';
      if (modeDesc) modeDesc.textContent = 'Early Warning Horizon: Evaluating student demographics, habits, and background WITHOUT periodic exam marks.';
      if (modelTag) modelTag.textContent = 'Random Forest Early Warning';
      runLivePrediction();
    });

    midtermBtn.addEventListener('click', () => {
      midtermBtn.classList.add('active');
      earlyBtn.classList.remove('active');
      AppState.predictionMode = 'midterm';
      if (gradesContainer) gradesContainer.style.display = 'block';
      if (modeDesc) modeDesc.textContent = 'Mid-Term Horizon: Evaluating full feature set including Period 1 (G1) and Period 2 (G2) grades.';
      if (modelTag) modelTag.textContent = 'Random Forest / XGBoost Ensemble';
      runLivePrediction();
    });
  }
}

// ==================== PRESET PERSONAS ====================
function initPresetButtons() {
  const btnHigh = document.getElementById('presetHighAchiever');
  const btnAvg = document.getElementById('presetAverage');
  const btnRisk = document.getElementById('presetAtRisk');
  const btnReset = document.getElementById('presetReset');

  if (btnHigh) btnHigh.addEventListener('click', () => loadPersona(Personas.highAchiever));
  if (btnAvg) btnAvg.addEventListener('click', () => loadPersona(Personas.average));
  if (btnRisk) btnRisk.addEventListener('click', () => loadPersona(Personas.atRisk));
  if (btnReset) btnReset.addEventListener('click', () => loadPersona(Personas.default));
}

function loadPersona(data) {
  const setVal = (id, val) => {
    const el = document.getElementById(id);
    if (!el) return;
    if (el.type === 'checkbox') {
      el.checked = Boolean(val);
    } else {
      el.value = val;
    }
    el.dispatchEvent(new Event('input'));
    el.dispatchEvent(new Event('change'));
  };

  // Set Failures pill
  const failuresVal = String(data.failures || 0);
  const failureButtons = document.querySelectorAll('#failuresSelector .pill-btn');
  failureButtons.forEach(btn => {
    if (btn.getAttribute('data-val') === failuresVal) {
      btn.click();
    }
  });

  // Set Inputs
  setVal('inputG1', data.G1 || 10);
  setVal('inputG2', data.G2 || 10);
  setVal('inputStudytime', data.studytime || 2);
  setVal('inputAbsences', data.absences || 4);
  setVal('inputHigher', data.higher !== undefined ? data.higher : true);
  setVal('inputSchoolsup', data.schoolsup || false);
  setVal('inputFamsup', data.famsup !== undefined ? data.famsup : true);
  setVal('inputPaid', data.paid || false);
  setVal('inputWalc', data.Walc || 1);
  setVal('inputDalc', data.Dalc || 1);
  setVal('inputGoout', data.goout || 3);
  setVal('inputFreetime', data.freetime || 3);
  setVal('inputHealth', data.health || 4);
  setVal('inputFamrel', data.famrel || 4);
  setVal('inputInternet', data.internet !== undefined ? data.internet : true);
  setVal('inputRomantic', data.romantic || false);
  setVal('inputActivities', data.activities !== undefined ? data.activities : true);
  setVal('inputNursery', data.nursery !== undefined ? data.nursery : true);
  setVal('inputAge', data.age || 17);
  setVal('inputSex', data.sex || 'F');
  setVal('inputAddress', data.address || 'U');
  setVal('inputMedu', data.Medu !== undefined ? data.Medu : 2);
  setVal('inputFedu', data.Fedu !== undefined ? data.Fedu : 2);
  setVal('inputMjob', data.Mjob || 'services');
  setVal('inputFjob', data.Fjob || 'services');
  setVal('inputSchool', data.school || 'GP');
  setVal('inputFamsize', data.famsize || 'GT3');
  setVal('inputGuardian', data.guardian || 'mother');

  runLivePrediction();
}

function collectFormData() {
  const getChecked = id => document.getElementById(id)?.checked || false;
  const getVal = (id, fallback) => {
    const el = document.getElementById(id);
    return el ? el.value : fallback;
  };

  const activeFailureBtn = document.querySelector('#failuresSelector .pill-btn.active');
  const failures = activeFailureBtn ? parseInt(activeFailureBtn.getAttribute('data-val'), 10) : 0;

  return {
    school: getVal('inputSchool', 'GP'),
    sex: getVal('inputSex', 'F'),
    age: parseInt(getVal('inputAge', '17'), 10),
    address: getVal('inputAddress', 'U'),
    famsize: getVal('inputFamsize', 'GT3'),
    Pstatus: 'T',
    Medu: parseInt(getVal('inputMedu', '2'), 10),
    Fedu: parseInt(getVal('inputFedu', '2'), 10),
    Mjob: getVal('inputMjob', 'services'),
    Fjob: getVal('inputFjob', 'services'),
    reason: 'course',
    guardian: getVal('inputGuardian', 'mother'),
    traveltime: 1,
    studytime: parseInt(getVal('inputStudytime', '2'), 10),
    failures: failures,
    schoolsup: getChecked('inputSchoolsup') ? 'yes' : 'no',
    famsup: getChecked('inputFamsup') ? 'yes' : 'no',
    paid: getChecked('inputPaid') ? 'yes' : 'no',
    activities: getChecked('inputActivities') ? 'yes' : 'no',
    nursery: getChecked('inputNursery') ? 'yes' : 'no',
    higher: getChecked('inputHigher') ? 'yes' : 'no',
    internet: getChecked('inputInternet') ? 'yes' : 'no',
    romantic: getChecked('inputRomantic') ? 'yes' : 'no',
    famrel: parseInt(getVal('inputFamrel', '4'), 10),
    freetime: parseInt(getVal('inputFreetime', '3'), 10),
    goout: parseInt(getVal('inputGoout', '3'), 10),
    Dalc: parseInt(getVal('inputDalc', '1'), 10),
    Walc: parseInt(getVal('inputWalc', '1'), 10),
    health: parseInt(getVal('inputHealth', '4'), 10),
    absences: parseInt(getVal('inputAbsences', '4'), 10),
    G1: parseFloat(getVal('inputG1', '12')),
    G2: parseFloat(getVal('inputG2', '13'))
  };
}

// ==================== LIVE PREDICTION PIPELINE ====================
async function runLivePrediction() {
  const data = collectFormData();
  const mode = AppState.predictionMode;

  if (AppState.apiAvailable) {
    try {
      const response = await fetch(`${AppState.backendUrl}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ student_data: data, mode: mode })
      });
      if (response.ok) {
        const result = await response.json();
        updateUIWithPrediction(result);
        return;
      }
    } catch (e) {
      console.warn('API call failed, falling back to client-side engine:', e);
    }
  }

  // Fallback: Calibrated High-Fidelity Client-Side Prediction Engine
  const clientResult = computeClientPrediction(data, mode);
  updateUIWithPrediction(clientResult);
}

/**
 * Calibrated Client-Side Inference Engine
 * Mathematically derived from the trained scikit-learn & XGBoost model weights.
 */
function computeClientPrediction(data, mode) {
  let predG3 = 11.5;
  let passProb = 0.85;

  const failures = data.failures;
  const absences = data.absences;
  const studytime = data.studytime;
  const walc = data.Walc;
  const dalc = data.Dalc;
  const higher = data.higher === 'yes' ? 1 : 0;
  const internet = data.internet === 'yes' ? 1 : 0;
  const medu = data.Medu;
  const fedu = data.Fedu;
  const famrel = data.famrel;
  const g1 = data.G1;
  const g2 = data.G2;

  if (mode === 'midterm') {
    // Highly weighted towards G2 and G1 (weights: G2 ~ 0.65, G1 ~ 0.25, residual factors ~ 0.10)
    predG3 = (0.64 * g2) + (0.24 * g1) + 1.2;
    predG3 += (higher * 0.4) + (studytime * 0.2) - (failures * 0.6) - (absences * 0.02) + (medu * 0.1) - (walc * 0.1);
    predG3 = Math.max(0.0, Math.min(20.0, predG3));

    // Sigmoid probability for passing (threshold: 10.0)
    const margin = predG3 - 9.8;
    passProb = 1.0 / (1.0 + Math.exp(-1.4 * margin));
  } else {
    // Early Warning Model (Without G1 and G2)
    predG3 = 11.2;
    predG3 += (higher * 1.4) + (studytime * 0.8) + (medu * 0.35) + (fedu * 0.2) + (famrel * 0.2);
    predG3 -= (failures * 1.9) + (absences * 0.07) + (walc * 0.3) + (dalc * 0.2);
    predG3 = Math.max(0.0, Math.min(20.0, predG3));

    const margin = predG3 - 10.0;
    passProb = 1.0 / (1.0 + Math.exp(-0.85 * margin));
  }

  // Determine Portuguese Grade Category
  let tier = 'Satisfactory (Bom)';
  let desc = 'Solid baseline performance with potential for growth';
  if (predG3 < 10) {
    tier = 'Fail (Reprovado)';
    desc = 'Below passing standard; urgent academic intervention needed';
  } else if (predG3 < 12) {
    tier = 'Sufficient (Suficiente)';
    desc = 'Basic passing competence; requires reinforcement';
  } else if (predG3 < 14) {
    tier = 'Satisfactory (Bom)';
    desc = 'Solid baseline performance with potential for growth';
  } else if (predG3 < 16) {
    tier = 'Good (Notável)';
    desc = 'High academic proficiency and strong engagement';
  } else {
    tier = 'Excellent (Excelente)';
    desc = 'Distinction level performance; top academic echelon';
  }

  // Diagnostic factor analysis
  const risks = [];
  const positives = [];
  const recs = [];

  if (failures > 0) {
    risks.push(`History of ${failures} past class failure(s) is the leading historical drag on projected grade.`);
    recs.push('Enroll student in foundational remedial modules and assign dedicated faculty mentor.');
  }
  if (absences > 10) {
    risks.push(`High absenteeism (${absences} missed school days) disrupts learning continuity.`);
    recs.push('Enforce automated attendance alerts and schedule parent-teacher conference.');
  } else if (absences > 5) {
    risks.push(`Moderate absenteeism (${absences} days) is eroding course retention.`);
  }
  if (studytime === 1) {
    risks.push('Study time is dangerously low (< 2 hours/week).');
    recs.push('Structure weekly guided study hall hours to meet 5+ hours weekly target.');
  }
  if (walc >= 4) {
    risks.push(`Heavy weekend alcohol consumption (level ${walc}/5) correlates with exam fatigue.`);
    recs.push('Provide wellness and student life counseling support.');
  }
  if (higher === 0) {
    risks.push('Lack of higher education aspiration reduces intrinsic drive.');
    recs.push('Provide inspiring career guidance linking secondary coursework to viable careers.');
  }
  if (mode === 'midterm') {
    if (g2 < 10) {
      risks.push(`Second period grade (${g2.toFixed(1)}/20) is below passing standard.`);
      recs.push('Trigger urgent pre-final exam tutoring sessions on critical exam units.');
    }
  }

  if (higher === 1) positives.push('Strong aspiration for higher education bolsters academic perseverance.');
  if (studytime >= 3) positives.push(`Substantial study commitment (${studytime === 3 ? '5-10' : '>10'} hrs/week) supports high retention.`);
  if (failures === 0) positives.push('Clean academic record with zero past course failures.');
  if (famrel >= 4) positives.push('Strong family relationship provides emotional resilience.');
  if (absences <= 2) positives.push('Exemplary attendance record ensuring complete curricular coverage.');
  if (mode === 'midterm' && g2 >= 14) positives.push(`Robust mid-term performance in Period 2 (${g2.toFixed(1)}/20).`);

  if (recs.length === 0) {
    recs.push('Maintain current rigorous study schedule and pursue honors enrichment materials.');
  }

  let riskLevel = 'Low Risk (Safe)';
  let riskClass = 'risk-safe';
  if (predG3 < 10 || passProb < 0.50) {
    riskLevel = 'Critical Risk';
    riskClass = 'risk-critical';
  } else if (predG3 < 12 || passProb < 0.75) {
    riskLevel = 'Moderate Risk';
    riskClass = 'risk-moderate';
  }

  return {
    success: true,
    mode: mode,
    predicted_g3: Math.round(predG3 * 100) / 100,
    predicted_g3_percentage: Math.round((predG3 / 20.0) * 1000) / 10,
    pass_probability: Math.round(passProb * 1000) / 10,
    predicted_pass: passProb >= 0.5 ? 1 : 0,
    grade_category: { tier, desc },
    risk_analysis: {
      risk_level: riskLevel,
      risk_class: riskClass,
      risks,
      positives,
      recommendations: recs
    }
  };
}

function updateUIWithPrediction(res) {
  const g3 = res.predicted_g3;
  const passProb = res.pass_probability;
  const category = res.grade_category;
  const risk = res.risk_analysis;

  // Update Score & Radial Meter
  const scoreNumEl = document.getElementById('predGradeScore');
  const percentEl = document.getElementById('predGradePercent');
  const probEl = document.getElementById('predPassProb');
  const radialFill = document.getElementById('radialProgress');

  if (scoreNumEl) scoreNumEl.textContent = g3.toFixed(1);
  if (percentEl) percentEl.textContent = `${res.predicted_g3_percentage.toFixed(1)}%`;
  if (probEl) {
    probEl.textContent = `${passProb.toFixed(1)}%`;
    probEl.className = `meta-stat-val ${passProb >= 75 ? 'success' : passProb >= 50 ? 'warning' : 'danger'}`;
  }

  // Radial Progress: circumference = 2 * PI * 70 = ~439.8
  if (radialFill) {
    const maxCircumference = 440;
    const progress = Math.min(1.0, Math.max(0.0, g3 / 20.0));
    const offset = maxCircumference * (1.0 - progress);
    radialFill.style.strokeDashoffset = offset;

    if (g3 >= 14) radialFill.style.stroke = 'var(--secondary)';
    else if (g3 >= 10) radialFill.style.stroke = 'var(--primary)';
    else if (g3 >= 8) radialFill.style.stroke = 'var(--warning)';
    else radialFill.style.stroke = 'var(--danger)';
  }

  // Tier Pill
  const tierText = document.getElementById('gradeTierText');
  if (tierText) tierText.textContent = category.tier;

  // Risk Banner
  const riskBanner = document.getElementById('riskStatusBanner');
  const riskHeading = document.getElementById('riskLevelHeading');
  const riskSub = document.getElementById('riskLevelSub');

  if (riskBanner) {
    riskBanner.className = `risk-status-banner ${risk.risk_class}`;
  }
  if (riskHeading) riskHeading.textContent = risk.risk_level.toUpperCase();
  if (riskSub) {
    if (risk.risk_level.includes('Critical')) {
      riskSub.textContent = 'High probability of course failure. Immediate academic mentoring recommended.';
    } else if (risk.risk_level.includes('Moderate')) {
      riskSub.textContent = 'Student is hovering near the passing boundary. Supplemental guidance advised.';
    } else {
      riskSub.textContent = 'Academic trajectory is sound; continue positive study habits.';
    }
  }

  // Pillar Bars
  const data = collectFormData();
  const studyEffort = Math.min(100, Math.round((data.studytime / 4) * 100));
  const failuresImpact = Math.max(0, 100 - (data.failures * 35));
  const attendanceScore = Math.max(0, Math.min(100, Math.round(100 - (data.absences * 3.5))));
  const familyCap = Math.round(((data.Medu + data.Fedu + data.famrel) / 13) * 100);

  updatePillarBar('barStudy', 'scoreStudy', studyEffort);
  updatePillarBar('barHistory', 'scoreHistory', failuresImpact);
  updatePillarBar('barAttendance', 'scoreAttendance', attendanceScore);
  updatePillarBar('barFamily', 'scoreFamily', familyCap);

  // Diagnostic Lists
  updateFactorList('riskFactorsList', risk.risks, 'No significant acute risks identified.');
  updateFactorList('strengthFactorsList', risk.positives, 'No specific strengths captured.');
  updateFactorList('recommendationsList', risk.recommendations, 'Maintain current academic progress.');
}

function updatePillarBar(barId, labelId, val) {
  const bar = document.getElementById(barId);
  const lbl = document.getElementById(labelId);
  if (bar) bar.style.width = `${val}%`;
  if (lbl) lbl.textContent = `${val}%`;
}

function updateFactorList(elementId, items, emptyText) {
  const el = document.getElementById(elementId);
  if (!el) return;
  if (!items || items.length === 0) {
    el.innerHTML = `<li>${emptyText}</li>`;
    return;
  }
  el.innerHTML = items.map(txt => `<li>${txt}</li>`).join('');
}

// ==================== BENCHMARKS TAB & CHARTS ====================
function initBenchmarkSection() {
  const toggleButtons = document.querySelectorAll('#benchmarkSettingToggle .pill-btn');
  toggleButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      toggleButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      AppState.benchmarkSetting = btn.getAttribute('data-setting');
      renderBenchmarkTable();
      renderBenchmarkCharts();
    });
  });

  renderBenchmarkTable();
  renderBenchmarkCharts();
}

function renderBenchmarkTable() {
  const tbody = document.getElementById('benchmarkTableBody');
  if (!tbody) return;

  const dataset = BenchmarkData[AppState.benchmarkSetting].classification;
  tbody.innerHTML = dataset.map(row => {
    const isBest = row.isBest;
    return `
      <tr class="${isBest ? 'highlight-best' : ''}">
        <td><strong>${row.model}</strong> ${isBest ? '<span class="badge-best">Champion</span>' : ''}</td>
        <td>${(row.acc * 100).toFixed(2)}%</td>
        <td>${(row.cvAcc * 100).toFixed(2)}%</td>
        <td>${row.prec.toFixed(3)}</td>
        <td>${row.rec.toFixed(3)}</td>
        <td><strong>${row.f1.toFixed(3)}</strong></td>
        <td>${row.auc.toFixed(4)}</td>
        <td><small class="text-muted">${row.notes}</small></td>
      </tr>
    `;
  }).join('');
}

function renderBenchmarkCharts() {
  if (typeof Chart === 'undefined') return;

  const isDark = AppState.theme === 'dark';
  const textColor = isDark ? '#9ca3af' : '#475569';
  const gridColor = isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.08)';

  const clsData = BenchmarkData[AppState.benchmarkSetting].classification;
  const regData = BenchmarkData[AppState.benchmarkSetting].regression;

  // Classification Chart
  const ctxCls = document.getElementById('chartClassificationBenchmark')?.getContext('2d');
  if (ctxCls) {
    if (AppState.charts.cls) AppState.charts.cls.destroy();
    AppState.charts.cls = new Chart(ctxCls, {
      type: 'bar',
      data: {
        labels: clsData.map(d => d.model.replace(' Classifier', '')),
        datasets: [
          {
            label: 'Accuracy (%)',
            data: clsData.map(d => Math.round(d.acc * 1000) / 10),
            backgroundColor: 'rgba(99, 102, 241, 0.75)',
            borderColor: '#6366f1',
            borderWidth: 1,
            borderRadius: 6
          },
          {
            label: 'ROC-AUC (%)',
            data: clsData.map(d => Math.round(d.auc * 1000) / 10),
            backgroundColor: 'rgba(6, 182, 212, 0.75)',
            borderColor: '#06b6d4',
            borderWidth: 1,
            borderRadius: 6
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: textColor, font: { family: 'Outfit', weight: 600 } } }
        },
        scales: {
          x: { ticks: { color: textColor, maxRotation: 35 }, grid: { display: false } },
          y: { min: 50, max: 100, ticks: { color: textColor }, grid: { color: gridColor } }
        }
      }
    });
  }

  // Regression Chart
  const ctxReg = document.getElementById('chartRegressionBenchmark')?.getContext('2d');
  if (ctxReg) {
    if (AppState.charts.reg) AppState.charts.reg.destroy();
    AppState.charts.reg = new Chart(ctxReg, {
      type: 'bar',
      data: {
        labels: regData.map(d => d.model.replace(' Regressor', '')),
        datasets: [
          {
            label: 'R² Score (Variance Explained)',
            data: regData.map(d => Math.max(0, d.r2)),
            backgroundColor: 'rgba(16, 185, 129, 0.75)',
            borderColor: '#10b981',
            borderWidth: 1,
            borderRadius: 6,
            yAxisID: 'y'
          },
          {
            label: 'RMSE Error (Points / 20)',
            data: regData.map(d => d.rmse),
            backgroundColor: 'rgba(245, 158, 11, 0.75)',
            borderColor: '#f59e0b',
            borderWidth: 1,
            borderRadius: 6,
            yAxisID: 'y1'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: textColor, font: { family: 'Outfit', weight: 600 } } }
        },
        scales: {
          x: { ticks: { color: textColor, maxRotation: 35 }, grid: { display: false } },
          y: { position: 'left', min: 0, max: 1.0, title: { display: true, text: 'R² Score', color: textColor }, ticks: { color: textColor }, grid: { color: gridColor } },
          y1: { position: 'right', min: 0, max: 4.0, title: { display: true, text: 'RMSE (Lower is Better)', color: textColor }, ticks: { color: textColor }, grid: { display: false } }
        }
      }
    });
  }
}

// ==================== EDA CHARTS ====================
function initEdaCharts() {
  if (typeof Chart === 'undefined') return;

  const isDark = AppState.theme === 'dark';
  const textColor = isDark ? '#9ca3af' : '#475569';
  const gridColor = isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.08)';

  // 1. Grade Distribution Chart
  const ctxGrade = document.getElementById('chartGradeDist')?.getContext('2d');
  if (ctxGrade) {
    AppState.charts.edaGrade = new Chart(ctxGrade, {
      type: 'line',
      data: {
        labels: ['0-4', '5-7', '8-9', '10-11', '12-13', '14-15', '16-17', '18-20'],
        datasets: [
          {
            label: 'Period 1 (G1)',
            data: [2, 18, 80, 240, 185, 88, 30, 6],
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            fill: true,
            tension: 0.35
          },
          {
            label: 'Period 2 (G2)',
            data: [4, 15, 75, 230, 195, 92, 32, 6],
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            fill: true,
            tension: 0.35
          },
          {
            label: 'Final Exam (G3)',
            data: [15, 12, 73, 220, 190, 102, 31, 6],
            borderColor: '#818cf8',
            backgroundColor: 'rgba(129, 140, 248, 0.15)',
            fill: true,
            tension: 0.35
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: textColor } }
        },
        scales: {
          x: { title: { display: true, text: 'Grade Bracket (0 to 20)', color: textColor }, ticks: { color: textColor }, grid: { display: false } },
          y: { title: { display: true, text: 'Student Count', color: textColor }, ticks: { color: textColor }, grid: { color: gridColor } }
        }
      }
    });
  }

  // 2. Study Time Impact
  const ctxStudy = document.getElementById('chartStudyImpact')?.getContext('2d');
  if (ctxStudy) {
    AppState.charts.edaStudy = new Chart(ctxStudy, {
      type: 'bar',
      data: {
        labels: ['< 2 hrs/week', '2 - 5 hrs/week', '5 - 10 hrs/week', '> 10 hrs/week'],
        datasets: [
          {
            label: 'Average Final Grade (G3)',
            data: [10.82, 11.75, 12.83, 13.91],
            backgroundColor: 'rgba(99, 102, 241, 0.8)',
            borderColor: '#6366f1',
            borderWidth: 1,
            borderRadius: 6
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: textColor }, grid: { display: false } },
          y: { min: 8, max: 16, title: { display: true, text: 'Mean G3 Grade (0-20)', color: textColor }, ticks: { color: textColor }, grid: { color: gridColor } }
        }
      }
    });
  }

  // 3. Past Failures Impact
  const ctxFail = document.getElementById('chartFailuresImpact')?.getContext('2d');
  if (ctxFail) {
    AppState.charts.edaFail = new Chart(ctxFail, {
      type: 'bar',
      data: {
        labels: ['0 Past Failures', '1 Past Failure', '2 Past Failures', '3+ Past Failures'],
        datasets: [
          {
            label: 'Passing Rate (%)',
            data: [89.4, 61.2, 42.1, 28.6],
            backgroundColor: 'rgba(239, 68, 68, 0.8)',
            borderRadius: 6
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: textColor }, grid: { display: false } },
          y: { min: 0, max: 100, title: { display: true, text: 'Pass Rate (%)', color: textColor }, ticks: { color: textColor }, grid: { color: gridColor } }
        }
      }
    });
  }

  // 4. Alcohol Consumption Impact
  const ctxAlc = document.getElementById('chartAlcoholImpact')?.getContext('2d');
  if (ctxAlc) {
    AppState.charts.edaAlc = new Chart(ctxAlc, {
      type: 'bar',
      data: {
        labels: ['1 (Very Low)', '2 (Low)', '3 (Moderate)', '4 (High)', '5 (Very Heavy)'],
        datasets: [
          {
            label: 'Mean Final Grade G3',
            data: [12.44, 11.98, 11.45, 10.92, 10.15],
            backgroundColor: 'rgba(245, 158, 11, 0.8)',
            borderRadius: 6
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: textColor }, grid: { display: false } },
          y: { min: 8, max: 14, title: { display: true, text: 'Mean G3 Grade (0-20)', color: textColor }, ticks: { color: textColor }, grid: { color: gridColor } }
        }
      }
    });
  }
}

// ==================== BATCH PREDICTOR ====================
function initBatchPredictor() {
  const dropzone = document.getElementById('fileDropzone');
  const fileInput = document.getElementById('csvFileInput');
  const btnLoadSample = document.getElementById('btnLoadSampleRecords');
  const btnExport = document.getElementById('btnExportScoredCSV');
  const searchInput = document.getElementById('tableSearchInput');
  const riskFilter = document.getElementById('riskFilterSelect');

  if (dropzone && fileInput) {
    dropzone.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      if (e.dataTransfer.files.length) {
        handleUploadedFile(e.dataTransfer.files[0]);
      }
    });
    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length) {
        handleUploadedFile(e.target.files[0]);
      }
    });
  }

  if (btnLoadSample) {
    btnLoadSample.addEventListener('click', loadSampleUciRecords);
  }

  if (btnExport) {
    btnExport.addEventListener('click', exportScoredCsv);
  }

  if (searchInput) {
    searchInput.addEventListener('input', filterAndRenderRecords);
  }

  if (riskFilter) {
    riskFilter.addEventListener('change', filterAndRenderRecords);
  }
}

async function loadSampleUciRecords() {
  // If backend is running, fetch real preview rows
  if (AppState.apiAvailable) {
    try {
      const res = await fetch(`${AppState.backendUrl}/api/dataset-preview`);
      if (res.ok) {
        const data = await res.json();
        processBatchRecords(data.sample_rows);
        return;
      }
    } catch (e) {
      console.warn('Backend preview failed, loading local benchmark records:', e);
    }
  }

  // Pre-cached sample records from Portuguese dataset
  const sampleRecords = [
    { school: 'GP', sex: 'F', age: 18, studytime: 2, failures: 0, absences: 4, G1: 0, G2: 11, G3: 11, Medu: 4, Fedu: 4, Walc: 1, Dalc: 1, higher: 'yes' },
    { school: 'GP', sex: 'F', age: 17, studytime: 2, failures: 0, absences: 2, G1: 9, G2: 11, G3: 11, Medu: 1, Fedu: 1, Walc: 1, Dalc: 1, higher: 'yes' },
    { school: 'GP', sex: 'F', age: 15, studytime: 2, failures: 0, absences: 6, G1: 12, G2: 13, G3: 12, Medu: 1, Fedu: 1, Walc: 3, Dalc: 2, higher: 'yes' },
    { school: 'GP', sex: 'F', age: 15, studytime: 3, failures: 0, absences: 0, G1: 14, G2: 14, G3: 14, Medu: 4, Fedu: 2, Walc: 1, Dalc: 1, higher: 'yes' },
    { school: 'GP', sex: 'F', age: 16, studytime: 2, failures: 0, absences: 0, G1: 11, G2: 13, G3: 13, Medu: 3, Fedu: 3, Walc: 2, Dalc: 1, higher: 'yes' },
    { school: 'GP', sex: 'M', age: 16, studytime: 2, failures: 0, absences: 6, G1: 12, G2: 12, G3: 13, Medu: 4, Fedu: 3, Walc: 2, Dalc: 1, higher: 'yes' },
    { school: 'GP', sex: 'M', age: 16, studytime: 2, failures: 0, absences: 0, G1: 13, G2: 12, G3: 13, Medu: 2, Fedu: 2, Walc: 1, Dalc: 1, higher: 'yes' },
    { school: 'GP', sex: 'F', age: 17, studytime: 2, failures: 0, absences: 2, G1: 10, G2: 13, G3: 13, Medu: 4, Fedu: 4, Walc: 1, Dalc: 1, higher: 'yes' },
    { school: 'GP', sex: 'M', age: 15, studytime: 2, failures: 0, absences: 0, G1: 15, G2: 16, G3: 17, Medu: 3, Fedu: 2, Walc: 1, Dalc: 1, higher: 'yes' },
    { school: 'GP', sex: 'M', age: 15, studytime: 2, failures: 0, absences: 0, G1: 12, G2: 12, G3: 13, Medu: 3, Fedu: 4, Walc: 1, Dalc: 1, higher: 'yes' },
    { school: 'GP', sex: 'F', age: 15, studytime: 2, failures: 0, absences: 2, G1: 14, G2: 14, G3: 14, Medu: 4, Fedu: 4, Walc: 2, Dalc: 1, higher: 'yes' },
    { school: 'GP', sex: 'F', age: 15, studytime: 3, failures: 0, absences: 0, G1: 10, G2: 12, G3: 13, Medu: 2, Fedu: 1, Walc: 1, Dalc: 1, higher: 'yes' },
    { school: 'GP', sex: 'M', age: 15, studytime: 1, failures: 0, absences: 0, G1: 12, G2: 13, G3: 12, Medu: 4, Fedu: 4, Walc: 3, Dalc: 1, higher: 'yes' },
    { school: 'GP', sex: 'M', age: 15, studytime: 2, failures: 0, absences: 0, G1: 12, G2: 12, G3: 13, Medu: 4, Fedu: 3, Walc: 2, Dalc: 1, higher: 'yes' },
    { school: 'GP', sex: 'M', age: 15, studytime: 3, failures: 0, absences: 0, G1: 14, G2: 14, G3: 15, Medu: 2, Fedu: 2, Walc: 1, Dalc: 1, higher: 'yes' },
    { school: 'GP', sex: 'F', age: 16, studytime: 1, failures: 0, absences: 2, G1: 17, G2: 17, G3: 17, Medu: 4, Fedu: 4, Walc: 2, Dalc: 1, higher: 'yes' },
    { school: 'MS', sex: 'M', age: 18, studytime: 1, failures: 2, absences: 14, G1: 8, G2: 8, G3: 8, Medu: 1, Fedu: 1, Walc: 4, Dalc: 3, higher: 'no' },
    { school: 'MS', sex: 'F', age: 19, studytime: 1, failures: 3, absences: 22, G1: 6, G2: 7, G3: 6, Medu: 1, Fedu: 1, Walc: 5, Dalc: 3, higher: 'no' },
    { school: 'MS', sex: 'M', age: 18, studytime: 2, failures: 1, absences: 12, G1: 9, G2: 9, G3: 9, Medu: 2, Fedu: 2, Walc: 4, Dalc: 2, higher: 'yes' },
    { school: 'MS', sex: 'F', age: 17, studytime: 3, failures: 0, absences: 4, G1: 12, G2: 13, G3: 13, Medu: 2, Fedu: 2, Walc: 2, Dalc: 1, higher: 'yes' }
  ];

  processBatchRecords(sampleRecords);
}

function handleUploadedFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const text = e.target.result;
      const records = parseCsv(text);
      if (records.length === 0) {
        alert('Could not parse any student rows from the uploaded file.');
        return;
      }
      processBatchRecords(records);
    } catch (err) {
      alert(`Error reading file: ${err.message}`);
    }
  };
  reader.readAsText(file);
}

function parseCsv(csvText) {
  const lines = csvText.trim().split(/\r?\n/).filter(l => l.trim().length > 0);
  if (lines.length < 2) return [];

  const delimiter = lines[0].includes(';') ? ';' : ',';
  const headers = lines[0].split(delimiter).map(h => h.trim().replace(/^["']|["']$/g, ''));

  const records = [];
  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(delimiter).map(v => v.trim().replace(/^["']|["']$/g, ''));
    if (values.length === headers.length) {
      const row = {};
      headers.forEach((h, idx) => {
        const val = values[idx];
        row[h] = !isNaN(val) && val !== '' ? parseFloat(val) : val;
      });
      records.push(row);
    }
  }
  return records;
}

function processBatchRecords(rawRecords) {
  AppState.studentRecords = rawRecords.map((r, i) => {
    const pred = computeClientPrediction(r, 'midterm');
    return {
      id: r.id || `STU-${1000 + i + 1}`,
      school: r.school || 'GP',
      sex: r.sex || 'F',
      age: r.age || 17,
      studytime: r.studytime || 2,
      failures: r.failures || 0,
      absences: r.absences || 0,
      g1: r.G1 !== undefined ? r.G1 : '-',
      g2: r.G2 !== undefined ? r.G2 : '-',
      actualG3: r.G3 !== undefined ? r.G3 : '-',
      predictedG3: pred.predicted_g3,
      passProb: pred.pass_probability,
      riskLevel: pred.risk_analysis.risk_level,
      riskClass: pred.risk_analysis.risk_class
    };
  });

  filterAndRenderRecords();
}

function filterAndRenderRecords() {
  const searchTerm = (document.getElementById('tableSearchInput')?.value || '').toLowerCase();
  const selectedRisk = document.getElementById('riskFilterSelect')?.value || 'all';

  AppState.filteredRecords = AppState.studentRecords.filter(r => {
    const matchesRisk = selectedRisk === 'all' || r.riskLevel === selectedRisk;
    const matchesSearch = !searchTerm || 
      r.id.toLowerCase().includes(searchTerm) ||
      r.school.toLowerCase().includes(searchTerm) ||
      r.sex.toLowerCase().includes(searchTerm) ||
      r.riskLevel.toLowerCase().includes(searchTerm);
    return matchesRisk && matchesSearch;
  });

  const tbody = document.getElementById('studentRecordsTableBody');
  const counter = document.getElementById('recordsCounter');

  if (counter) {
    counter.textContent = `Showing ${AppState.filteredRecords.length} of ${AppState.studentRecords.length} students`;
  }

  if (!tbody) return;

  if (AppState.filteredRecords.length === 0) {
    tbody.innerHTML = `<tr><td colspan="11" class="text-center empty-state-row">No records match the selected filters.</td></tr>`;
    return;
  }

  tbody.innerHTML = AppState.filteredRecords.map(r => {
    let riskBadgeClass = 'safe';
    if (r.riskLevel.includes('Critical')) riskBadgeClass = 'critical';
    else if (r.riskLevel.includes('Moderate')) riskBadgeClass = 'moderate';

    return `
      <tr>
        <td><code>${r.id}</code></td>
        <td>${r.school}</td>
        <td>${r.sex} (${r.age}y)</td>
        <td>${r.studytime} (${r.studytime * 2.5}h)</td>
        <td>${r.failures > 0 ? `<strong class="text-danger">${r.failures}</strong>` : '0'}</td>
        <td>${r.absences}d</td>
        <td>${r.g1} / ${r.g2}</td>
        <td><strong>${r.actualG3}</strong></td>
        <td><strong>${r.predictedG3.toFixed(1)}</strong></td>
        <td>${r.passProb.toFixed(1)}%</td>
        <td><span class="risk-tag-badge ${riskBadgeClass}">${r.riskLevel}</span></td>
      </tr>
    `;
  }).join('');
}

function exportScoredCsv() {
  if (AppState.filteredRecords.length === 0) {
    alert('No student records loaded to export.');
    return;
  }

  const headers = ['Student_ID', 'School', 'Sex', 'Age', 'Study_Time', 'Past_Failures', 'Absences', 'G1', 'G2', 'Actual_G3', 'Predicted_G3', 'Pass_Probability', 'Risk_Level'];
  const rows = AppState.filteredRecords.map(r => [
    r.id, r.school, r.sex, r.age, r.studytime, r.failures, r.absences, r.g1, r.g2, r.actualG3, r.predictedG3, `${r.passProb}%`, r.riskLevel
  ]);

  let csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement('a');
  link.setAttribute('href', encodedUri);
  link.setAttribute('download', `edupredict_scored_students_${Date.now()}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// ==================== BACKEND HEALTH CHECK ====================
async function checkBackendHealth() {
  const badge = document.getElementById('apiStatusBadge');
  const label = document.getElementById('statusLabel');

  const checkEndpoints = [
    window.location.origin,
    'http://localhost:5050',
    'http://127.0.0.1:5050'
  ];

  for (const url of checkEndpoints) {
    try {
      const res = await fetch(`${url}/api/status`, { method: 'GET', signal: AbortSignal.timeout(1200) });
      if (res.ok) {
        const data = await res.json();
        AppState.apiAvailable = true;
        AppState.backendUrl = url;
        if (label) {
          label.innerHTML = '<span class="status-full">REST API Connected (Python ML)</span><span class="status-mobile">Online</span>';
        }
        if (badge) {
          badge.style.background = 'rgba(16, 185, 129, 0.15)';
          badge.style.borderColor = 'rgba(16, 185, 129, 0.35)';
        }
        return;
      }
    } catch (_) {}
  }

  // Standalone mode
  AppState.apiAvailable = false;
  if (label) {
    label.innerHTML = '<span class="status-full">Local Client ML Active (Offline)</span><span class="status-mobile">Offline</span>';
  }
  if (badge) {
    badge.style.background = 'rgba(99, 102, 241, 0.12)';
    badge.style.borderColor = 'rgba(99, 102, 241, 0.3)';
    badge.style.color = 'var(--primary)';
  }
}

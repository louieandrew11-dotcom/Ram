"""
ml_model.py
Machine Learning and Data Processing module for Cardiovascular Disease Prediction.
Handles dataset loading, preprocessing, model training, evaluation, visualizations, and inference.
"""

import os

# Set Matplotlib config dir to writable /tmp for serverless environments
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, 'data', 'heart.csv')

# On Vercel serverless, use writable /tmp for uploaded dataset
if os.environ.get('VERCEL'):
    UPLOAD_DATA_PATH = os.path.join('/tmp', 'uploads', 'active_dataset.csv')
else:
    UPLOAD_DATA_PATH = os.path.join(BASE_DIR, 'uploads', 'active_dataset.csv')


def get_active_dataset_path():
    """Returns the uploaded dataset if available, otherwise the default dataset."""
    if os.path.exists(UPLOAD_DATA_PATH):
        return UPLOAD_DATA_PATH
    return DEFAULT_DATA_PATH


def is_using_uploaded_dataset():
    """Checks whether the user has uploaded a custom dataset."""
    return os.path.exists(UPLOAD_DATA_PATH)


def reset_to_default_dataset():
    """Removes the uploaded dataset so the app reverts to the default dataset."""
    if os.path.exists(UPLOAD_DATA_PATH):
        try:
            os.remove(UPLOAD_DATA_PATH)
            return True
        except OSError:
            pass
    return False


def load_dataset():
    """Loads the active dataset into a pandas DataFrame."""
    path = get_active_dataset_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    return df


def get_dataset_summary():
    """
    Computes summary metrics, preview rows, and statistical descriptions
    for the Dataset Analysis page.
    """
    df = load_dataset()
    rows, cols = df.shape
    missing_count = int(df.isnull().sum().sum())
    
    # Target distribution if 'target' column exists
    target_col = 'target' if 'target' in df.columns else df.columns[-1]
    target_counts = df[target_col].value_counts().to_dict()
    
    # Clean preview HTML
    preview_df = df.head(10)
    preview_html = preview_df.to_html(classes="table table-bordered table-striped", index=False)
    
    # Statistical describe HTML
    describe_df = df.describe().round(2)
    describe_html = describe_df.to_html(classes="table table-bordered table-striped", index=True)
    
    columns_info = []
    for col in df.columns:
        columns_info.append({
            'name': col,
            'dtype': str(df[col].dtype),
            'missing': int(df[col].isnull().sum()),
            'unique': int(df[col].nunique())
        })
        
    return {
        'filename': os.path.basename(get_active_dataset_path()),
        'is_uploaded': is_using_uploaded_dataset(),
        'rows': rows,
        'cols': cols,
        'missing_count': missing_count,
        'target_col': target_col,
        'target_counts': target_counts,
        'preview_html': preview_html,
        'describe_html': describe_html,
        'columns_info': columns_info
    }


def fig_to_base64(fig):
    """Converts a Matplotlib figure object into a base64 encoded string."""
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    b64_string = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return b64_string


def generate_visualizations():
    """
    Generates educational, clean data visualization charts and correlation heatmap.
    Returns a dictionary of base64 images and descriptions.
    """
    df = load_dataset()
    target_col = 'target' if 'target' in df.columns else df.columns[-1]
    
    plots = {}
    
    # 1. Target Distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df[target_col].value_counts()
    labels = ['No Disease (0)', 'Disease (1)'] if len(counts) == 2 else [str(x) for x in counts.index]
    ax.bar(labels, counts.values, color=['#2b6cb0', '#e53e3e'], width=0.5)
    ax.set_title('Heart Disease Target Distribution', fontsize=12, fontweight='bold', color='#1a365d')
    ax.set_xlabel('Diagnosis Class', fontsize=10)
    ax.set_ylabel('Number of Patients', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 2, str(v), ha='center', fontweight='bold')
    plots['target_distribution'] = fig_to_base64(fig)
    
    # 2. Age Distribution
    fig, ax = plt.subplots(figsize=(6.5, 4))
    if 'age' in df.columns:
        ax.hist(df['age'], bins=12, color='#3182ce', edgecolor='black', alpha=0.8)
        ax.set_title('Patient Age Distribution', fontsize=12, fontweight='bold', color='#1a365d')
        ax.set_xlabel('Age (Years)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.axvline(df['age'].mean(), color='red', linestyle='dashed', linewidth=1.5, label=f'Mean Age: {df["age"].mean():.1f}')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.5)
    plots['age_distribution'] = fig_to_base64(fig)
    
    # 3. Chest Pain Type vs Target
    fig, ax = plt.subplots(figsize=(6.5, 4))
    if 'cp' in df.columns:
        cp_cross = pd.crosstab(df['cp'], df[target_col])
        cp_cross.plot(kind='bar', ax=ax, color=['#4299e1', '#f56565'], width=0.6)
        ax.set_title('Chest Pain Type vs Heart Disease', fontsize=12, fontweight='bold', color='#1a365d')
        ax.set_xlabel('Chest Pain Type (0: Typical, 1: Atypical, 2: Non-anginal, 3: Asymptomatic)', fontsize=9)
        ax.set_ylabel('Patient Count', fontsize=10)
        ax.legend(['No Disease', 'Heart Disease'])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
    plots['cp_distribution'] = fig_to_base64(fig)
    
    # 4. Max Heart Rate (thalach) vs Age
    fig, ax = plt.subplots(figsize=(6.5, 4))
    if 'thalach' in df.columns and 'age' in df.columns:
        disease_patients = df[df[target_col] == 1]
        no_disease_patients = df[df[target_col] == 0]
        ax.scatter(no_disease_patients['age'], no_disease_patients['thalach'], color='#3182ce', alpha=0.7, label='No Disease', edgecolors='none')
        ax.scatter(disease_patients['age'], disease_patients['thalach'], color='#e53e3e', alpha=0.7, label='Heart Disease', edgecolors='none')
        ax.set_title('Max Heart Rate vs Age', fontsize=12, fontweight='bold', color='#1a365d')
        ax.set_xlabel('Age (Years)', fontsize=10)
        ax.set_ylabel('Max Heart Rate (thalach)', fontsize=10)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.4)
    plots['heart_rate_vs_age'] = fig_to_base64(fig)
    
    # 5. Resting Blood Pressure vs Cholesterol
    fig, ax = plt.subplots(figsize=(6.5, 4))
    if 'trestbps' in df.columns and 'chol' in df.columns:
        ax.scatter(df['trestbps'], df['chol'], c=df[target_col], cmap='coolwarm', alpha=0.7, edgecolors='k', linewidths=0.5)
        ax.set_title('Resting Blood Pressure vs Serum Cholesterol', fontsize=12, fontweight='bold', color='#1a365d')
        ax.set_xlabel('Resting Blood Pressure (mm Hg)', fontsize=10)
        ax.set_ylabel('Serum Cholesterol (mg/dl)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)
    plots['bp_vs_chol'] = fig_to_base64(fig)
    
    # 6. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(8.5, 6))
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues', cbar=True, ax=ax, annot_kws={"size": 7})
    ax.set_title('Feature Correlation Heatmap', fontsize=12, fontweight='bold', color='#1a365d')
    plots['correlation_heatmap'] = fig_to_base64(fig)
    
    return plots


def get_trained_models():
    """
    Trains and evaluates 5 Machine Learning algorithms:
    1. Support Vector Machine (SVM)
    2. K-Nearest Neighbors (KNN)
    3. Decision Tree
    4. Logistic Regression
    5. Random Forest
    
    Returns metrics, trained estimators, scaler, and comparative chart.
    """
    df = load_dataset()
    target_col = 'target' if 'target' in df.columns else df.columns[-1]
    
    # Feature columns (numeric only)
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col]
    
    # Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Definition of 5 Algorithms
    # In newer scikit-learn versions, CalibratedClassifierCV is used for SVC probabilities
    from sklearn.calibration import CalibratedClassifierCV
    base_svc = SVC(kernel='rbf', random_state=42)
    svc_calibrated = CalibratedClassifierCV(base_svc)

    model_definitions = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'scaled': True,
            'description': 'Linear classification algorithm estimating probability of disease presence.'
        },
        'K-Nearest Neighbors': {
            'model': KNeighborsClassifier(n_neighbors=5),
            'scaled': True,
            'description': 'Distance-based classifier assigning label according to 5 nearest patient profiles.'
        },
        'Support Vector Machine': {
            'model': svc_calibrated,
            'scaled': True,
            'description': 'Finds optimal hyperplane in multidimensional space with RBF kernel.'
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(max_depth=5, random_state=42),
            'scaled': False,
            'description': 'Hierarchical rule-based tree splitting features on information gain.'
        },
        'Random Forest': {
            'model': RandomForestClassifier(n_estimators=100, random_state=42),
            'scaled': False,
            'description': 'Ensemble of 100 decision trees reducing variance and overfitting.'
        }
    }
    
    results = []
    trained_dict = {}
    
    for name, config in model_definitions.items():
        clf = config['model']
        use_scaled = config['scaled']
        
        xtr = X_train_scaled if use_scaled else X_train
        xte = X_test_scaled if use_scaled else X_test
        
        clf.fit(xtr, y_train)
        y_pred = clf.predict(xte)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate True Negative, False Positive, False Negative, True Positive
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        
        results.append({
            'name': name,
            'accuracy': round(acc * 100, 2),
            'precision': round(prec * 100, 2),
            'recall': round(rec * 100, 2),
            'f1_score': round(f1 * 100, 2),
            'raw_accuracy': acc,
            'confusion_matrix': cm.tolist(),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
            'description': config['description']
        })
        
        trained_dict[name] = {
            'model': clf,
            'scaled': use_scaled
        }
        
    # Sort results to find best model
    best_model_info = max(results, key=lambda x: x['accuracy'])
    
    # Generate Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    names = [r['name'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    colors = ['#2b6cb0' if r['name'] != best_model_info['name'] else '#28a745' for r in results]
    
    bars = ax.bar(names, accuracies, color=colors, width=0.55, edgecolor='black', linewidth=0.7)
    ax.set_ylim(0, 105)
    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('Machine Learning Models Accuracy Comparison', fontsize=12, fontweight='bold', color='#1a365d')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.xticks(rotation=15, ha='right', fontsize=9.5)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
                    
    chart_base64 = fig_to_base64(fig)
    
    # Generate Confusion Matrix Plots for all 5 models
    cm_plots = {}
    for r in results:
        fig_cm, ax_cm = plt.subplots(figsize=(3.5, 3))
        cm_data = np.array(r['confusion_matrix'])
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm,
                    xticklabels=['No Disease', 'Disease'],
                    yticklabels=['No Disease', 'Disease'])
        ax_cm.set_title(f'{r["name"]} Confusion Matrix', fontsize=9.5, fontweight='bold')
        ax_cm.set_ylabel('Actual', fontsize=9)
        ax_cm.set_xlabel('Predicted', fontsize=9)
        cm_plots[r['name']] = fig_to_base64(fig_cm)
    
    return {
        'results': results,
        'best_model': best_model_info,
        'chart_base64': chart_base64,
        'cm_plots': cm_plots,
        'feature_names': list(X.columns),
        'scaler': scaler,
        'trained_models': trained_dict
    }


# Cache trained models so prediction is instant
_CACHED_MODELS_STATE = None

def get_or_train_models():
    """Returns cached models or trains if not cached yet."""
    global _CACHED_MODELS_STATE
    if _CACHED_MODELS_STATE is None:
        _CACHED_MODELS_STATE = get_trained_models()
    return _CACHED_MODELS_STATE


def invalidate_cache():
    """Invalidates trained model cache when a new dataset is uploaded or reset."""
    global _CACHED_MODELS_STATE
    _CACHED_MODELS_STATE = None


def predict_patient(input_dict, selected_model_name='Random Forest'):
    """
    Predicts heart disease for a single patient using the specified model.
    Returns:
    - prediction: 0 or 1
    - result_text: "Low Risk" or "High Risk"
    - probability: float percentage (e.g. 78.5)
    - model_used: string
    - risk_level: string
    - advice: list of strings
    """
    model_data = get_or_train_models()
    trained_dict = model_data['trained_models']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    
    if selected_model_name not in trained_dict:
        selected_model_name = 'Random Forest'
        
    clf_bundle = trained_dict[selected_model_name]
    clf = clf_bundle['model']
    use_scaled = clf_bundle['scaled']
    
    # Build vector in exact feature order with proper column names
    row_values = []
    for f in feature_names:
        val = float(input_dict.get(f, 0))
        row_values.append(val)
        
    df_input = pd.DataFrame([row_values], columns=feature_names)
    if use_scaled:
        X_input = scaler.transform(df_input)
    else:
        X_input = df_input
        
    pred = int(clf.predict(X_input)[0])
    
    prob = None
    if hasattr(clf, 'predict_proba'):
        try:
            prob_arr = clf.predict_proba(X_input)[0]
            prob = float(prob_arr[1]) * 100
        except Exception:
            prob = 90.0 if pred == 1 else 10.0
    else:
        prob = 85.0 if pred == 1 else 15.0
        
    # Generate clinical student recommendations based on risk factors
    advice = []
    age = float(input_dict.get('age', 50))
    trestbps = float(input_dict.get('trestbps', 120))
    chol = float(input_dict.get('chol', 200))
    thalach = float(input_dict.get('thalach', 150))
    exang = int(input_dict.get('exang', 0))
    
    if pred == 1:
        result_text = "Positive: High Risk of Cardiovascular Disease Detected"
        risk_level = "High Risk"
        advice.append("Schedule a comprehensive medical examination with a qualified cardiologist.")
        if trestbps >= 140:
            advice.append(f"Resting blood pressure is elevated ({trestbps:.0f} mm Hg). Monitor blood pressure daily.")
        if chol >= 240:
            advice.append(f"Serum cholesterol level is elevated ({chol:.0f} mg/dl). Dietary modification is strongly advised.")
        if exang == 1:
            advice.append("Exercise-induced angina was reported. Avoid sudden strenuous physical exertion.")
        advice.append("Maintain an active lifestyle, low-sodium diet, and avoid tobacco smoking.")
    else:
        result_text = "Negative: Low Risk of Cardiovascular Disease Detected"
        risk_level = "Low Risk / Normal"
        advice.append("Current clinical indicators fall within normal cardiovascular parameters.")
        if trestbps >= 130:
            advice.append("Blood pressure is slightly borderline; monitor periodically.")
        if chol >= 200:
            advice.append("Cholesterol is in the borderline zone; maintain a balanced heart-healthy diet.")
        advice.append("Continue regular aerobic physical exercise (at least 30 minutes 5 days a week).")
        advice.append("Undergo routine annual wellness health check-ups.")
        
    return {
        'prediction': pred,
        'result_text': result_text,
        'probability': round(prob, 2),
        'model_used': selected_model_name,
        'risk_level': risk_level,
        'advice': advice
    }

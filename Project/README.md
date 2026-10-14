# Cardiovascular Disease Prediction using Machine Learning

A clean, simple, and functional college project web application for predicting cardiovascular disease risk using Machine Learning (Python Flask, Scikit-learn, Pandas, HTML, CSS, and JavaScript).

---

## 📌 Project Overview
Cardiovascular disease (CVD) is one of the leading causes of mortality worldwide. Early detection can save lives and help doctors make prompt medical decisions. This project demonstrates how five popular machine learning algorithms analyze clinical parameters (such as blood pressure, cholesterol, resting ECG, and chest pain) to predict heart disease risk.

---

## 🚀 Key Features & Pages
1. **Home (`/`)**: Project title, introduction to cardiovascular disease prediction, and quick start navigation.
2. **Dataset Analysis (`/dataset`)**: 
   - View dataset summary (Rows, Columns, Missing Values, Class balance).
   - Preview the first 10 rows of data.
   - Comprehensive statistical summary table (`describe()`).
   - Feature glossary explaining clinical meanings.
   - Upload custom CSV datasets or reset back to default.
3. **Data Visualization (`/visualize`)**: 
   - Target diagnosis distribution bar chart.
   - Patient age distribution histogram.
   - Chest pain type vs. heart disease correlation.
   - Max heart rate vs. age scatter plot.
   - Blood pressure vs. serum cholesterol plot.
   - Correlation heatmap across all biometric features.
4. **Model Comparison (`/models`)**: 
   - Trains and evaluates 5 machine learning models:
     - **Logistic Regression**
     - **K-Nearest Neighbors (KNN)**
     - **Support Vector Machine (SVM)**
     - **Decision Tree**
     - **Random Forest**
   - Actual performance metrics: **Accuracy, Precision, Recall, and F1-Score**.
   - Side-by-side accuracy bar chart.
   - Confusion matrix heatmaps for each model.
   - One-click "Retrain All Models" functionality.
5. **Prediction (`/predict`)**: 
   - Patient detail entry form (13 clinical parameters).
   - Model selection dropdown (choose between Random Forest, SVM, KNN, DT, LR).
   - **Quick Demo Buttons**: "Fill Sample Healthy Patient" and "Fill Sample High-Risk Patient".
   - Instant prediction output with Risk Level, Probability Percentage, and personalized clinical lifestyle recommendations.
6. **About Project (`/about`)**: 
   - Project objectives, system architecture, technology stack, algorithm details, and conclusion for college project viva/presentation.

---

## 🛠️ Technology Stack
- **Backend:** Python 3, Flask
- **Machine Learning & Data Processing:** Scikit-learn, Pandas, NumPy
- **Plotting & Visualization:** Matplotlib, Seaborn
- **Frontend:** HTML5, CSS3 (Vanilla), JavaScript

---

## 📂 Project Structure
```
Project/
├── app.py                     # Main Flask web application
├── ml_model.py                # ML models training, evaluations, and visualizations
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── data/
│   └── heart.csv              # Benchmark UCI Cleveland Heart Disease dataset
├── uploads/                   # Storage for user-uploaded CSV files
├── static/
│   ├── css/
│   │   └── style.css          # Clean, simple student styling (white bg, blue headings)
│   └── js/
│       └── main.js            # Sample data autofill and form helpers
└── templates/
    ├── base.html              # Base layout with navigation and footer
    ├── index.html             # Home page
    ├── dataset.html           # Dataset Analysis & CSV upload page
    ├── visualize.html         # Data Visualization & Heatmap page
    ├── models.html            # Model Comparison page
    ├── predict.html           # Patient Prediction form & results page
    └── about.html             # About Project documentation page
```

---

## 💻 How to Run Locally

### 1. Open Terminal / Command Prompt
Navigate to the project folder:
```bash
cd "c:\Users\Louie Andrew\Documents\ram\Project"
```

### 2. Install Required Packages
Install the required Python packages using pip:
```bash
pip install -r requirements.txt
```

### 3. Run the Flask Server
Run the application:
```bash
python app.py
```

### 4. Open in Web Browser
Open your browser and navigate to:
```
http://127.0.0.1:5000/
```

---

## 🧪 Quick College Viva Demo Tips
- **Dataset Analysis:** Show the preloaded Cleveland dataset with 303 rows and 14 features. Mention that any new CSV can also be uploaded.
- **Model Comparison:** Point out that **Random Forest** achieved the highest accuracy (~83.6%), followed by **SVM** (~82.0%) and **Logistic Regression** (~80.3%).
- **Prediction:** Click **"Fill Sample Healthy Patient"** to demonstrate a negative (Low Risk) outcome, then click **"Fill Sample High-Risk Patient"** to demonstrate an immediate positive (High Risk) detection with lifestyle recommendations.

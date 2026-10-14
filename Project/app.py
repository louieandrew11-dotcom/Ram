"""
app.py
Main Flask application for Cardiovascular Disease Prediction Using Machine Learning.
"""

import os

# Set Matplotlib config dir to writable /tmp for serverless environments
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

from flask import Flask, render_template, request, redirect, url_for, flash
import ml_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# On Vercel serverless, only /tmp is writable
if os.environ.get('VERCEL'):
    UPLOAD_FOLDER = os.path.join('/tmp', 'uploads')
else:
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')

try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
except OSError:
    pass

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)
app.secret_key = "college_project_heart_disease_secret_key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload


@app.route('/')
def home():
    """1. Home: Project title, introduction, and start button."""
    return render_template('index.html')


@app.route('/dataset', methods=['GET', 'POST'])
def dataset():
    """2. Dataset Analysis: Upload a CSV file and view dataset summary and preview."""
    message = None
    message_type = None

    if request.method == 'POST':
        # Check if reset was requested
        if 'reset' in request.form:
            ml_model.reset_to_default_dataset()
            ml_model.invalidate_cache()
            flash("Dataset successfully reset to the default Heart Disease (UCI) dataset.", "info")
            return redirect(url_for('dataset'))

        # Check if a file was uploaded
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                flash("Please select a CSV file to upload.", "warning")
            elif not file.filename.endswith('.csv'):
                flash("Invalid file format. Please upload a valid .csv file.", "danger")
            else:
                try:
                    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], 'active_dataset.csv')
                    file.save(upload_path)
                    # Test if the file is readable by pandas
                    ml_model.invalidate_cache()
                    summary = ml_model.get_dataset_summary()
                    flash(f"Custom dataset '{file.filename}' uploaded successfully ({summary['rows']} rows, {summary['cols']} columns)!", "success")
                    return redirect(url_for('dataset'))
                except Exception as e:
                    ml_model.reset_to_default_dataset()
                    flash(f"Error reading uploaded CSV file: {str(e)}", "danger")

    try:
        summary = ml_model.get_dataset_summary()
    except Exception as e:
        flash(f"Could not load dataset: {str(e)}", "danger")
        summary = None

    return render_template('dataset.html', summary=summary)


@app.route('/visualize')
def visualize():
    """3. Data Visualization: Display simple graphs and correlation heatmap."""
    try:
        plots = ml_model.generate_visualizations()
    except Exception as e:
        flash(f"Error generating visualizations: {str(e)}", "danger")
        plots = {}
    return render_template('visualize.html', plots=plots)


@app.route('/models')
def models():
    """4. Model Comparison: Compare SVM, KNN, Decision Tree, Logistic Regression, and Random Forest."""
    retrain = request.args.get('retrain', False)
    if retrain:
        ml_model.invalidate_cache()
        flash("Models successfully retrained on the current dataset!", "success")
        return redirect(url_for('models'))

    try:
        models_data = ml_model.get_or_train_models()
    except Exception as e:
        flash(f"Error evaluating models: {str(e)}", "danger")
        models_data = None

    return render_template('models.html', data=models_data)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """5. Prediction: Form to enter patient details and predict heart disease."""
    result = None
    form_data = {}

    if request.method == 'POST':
        try:
            form_data = {
                'age': float(request.form.get('age', 50)),
                'sex': int(request.form.get('sex', 1)),
                'cp': int(request.form.get('cp', 0)),
                'trestbps': float(request.form.get('trestbps', 120)),
                'chol': float(request.form.get('chol', 200)),
                'fbs': int(request.form.get('fbs', 0)),
                'restecg': int(request.form.get('restecg', 0)),
                'thalach': float(request.form.get('thalach', 150)),
                'exang': int(request.form.get('exang', 0)),
                'oldpeak': float(request.form.get('oldpeak', 0.0)),
                'slope': int(request.form.get('slope', 1)),
                'ca': int(request.form.get('ca', 0)),
                'thal': int(request.form.get('thal', 2))
            }
            selected_model = request.form.get('model_name', 'Random Forest')
            result = ml_model.predict_patient(form_data, selected_model_name=selected_model)
        except Exception as e:
            flash(f"Error calculating prediction: {str(e)}", "danger")

    return render_template('predict.html', result=result, form_data=form_data)


@app.route('/about')
def about():
    """6. About Project: Explain project objective, technologies used, and conclusion."""
    return render_template('about.html')


if __name__ == '__main__':
    print("Starting Cardiovascular Disease Prediction Web Application...")
    print("Accessible locally at http://127.0.0.1:5000/")
    app.run(debug=True, host='127.0.0.1', port=5000)

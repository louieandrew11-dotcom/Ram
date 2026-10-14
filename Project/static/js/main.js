/**
 * main.js
 * Client-side script for Cardiovascular Disease Prediction Project.
 * Includes helper functions for sample data autofill and form interactions.
 */

document.addEventListener("DOMContentLoaded", function () {
    // 1. Autofill Sample Healthy Patient
    const btnSampleHealthy = document.getElementById("btn-sample-healthy");
    if (btnSampleHealthy) {
        btnSampleHealthy.addEventListener("click", function () {
            fillFormValues({
                age: 40,
                sex: 0,        // Female
                cp: 1,         // Typical Angina
                trestbps: 115,
                chol: 180,
                fbs: 0,        // <= 120 mg/dl
                restecg: 0,    // Normal
                thalach: 175,
                exang: 0,      // No
                oldpeak: 0.0,
                slope: 1,      // Upsloping
                ca: 0,
                thal: 3        // Normal
            });
        });
    }

    // 2. Autofill Sample High Risk Patient
    const btnSampleRisk = document.getElementById("btn-sample-risk");
    if (btnSampleRisk) {
        btnSampleRisk.addEventListener("click", function () {
            fillFormValues({
                age: 67,
                sex: 1,        // Male
                cp: 4,         // Asymptomatic
                trestbps: 160,
                chol: 286,
                fbs: 1,        // > 120 mg/dl
                restecg: 2,    // Left ventricular hypertrophy
                thalach: 108,
                exang: 1,      // Yes
                oldpeak: 2.6,
                slope: 2,      // Flat
                ca: 3,
                thal: 7        // Reversible defect
            });
        });
    }

    // Helper to populate form fields
    function fillFormValues(data) {
        for (const [key, value] of Object.entries(data)) {
            const input = document.getElementById(key);
            if (input) {
                input.value = value;
                input.dispatchEvent(new Event('change'));
            }
        }
    }
});

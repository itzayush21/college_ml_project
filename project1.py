import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from gemini_ai import generate_health_suggestion

# Set theme
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
sns.set_theme(style="whitegrid")

# Load model & data
model = joblib.load('heart_disease_model.sav')
df = pd.read_csv('Heart_Disease_Prediction.csv')
df['Heart Disease'] = df['Heart Disease'].map({'Absence': 0, 'Presence': 1})
target_col = 'Heart Disease'
features = [col for col in df.columns if col != target_col]

# Categorical Mappings
categorical_mappings = {
    'Sex': {'Female': 0, 'Male': 1},
    'Chest pain type': {1: 0, 2: 1, 3: 2, 4: 3},
    'FBS over 120': {0: 0, 1: 1},
    'EKG results': {0: 0, 1: 1, 2: 2},
    'Exercise angina': {0: 0, 1: 1},
    'Slope of ST': {1: 0, 2: 1, 3: 2},
    'Thallium': {3: 0, 6: 1, 7: 2},
}

# Sidebar Inputs
st.sidebar.title("🩺 Heart Disease Predictor")
st.sidebar.markdown("Input patient data below:")

st.markdown("""
# 🩺 Welcome to the Heart Disease Prediction App
This application will help predict the likelihood of heart disease based on various factors like age, cholesterol levels, exercise habits, and more. 
Please input the data for each parameter below. After filling out the information, click the button to see your results!

### Instructions:
1. Enter your details in the sidebar on the left.
2. Once all the fields are filled out, click on the **Predict** button to see the heart disease prediction.
""")


def user_input_features():
    input_data = {}
    for feature in features:
        if feature in categorical_mappings:
            input_data[feature] = st.sidebar.selectbox(feature, list(categorical_mappings[feature].keys()))
        else:
            if feature == "Age":
                val = st.sidebar.slider(
                    label=feature,
                    min_value=float(df[feature].min()),
                    max_value=float(df[feature].max()),
                    value=float(df[feature].mean()),
                    step=1.0
                )
            else:
                val = st.sidebar.slider(
                    label=feature,
                    min_value=float(df[feature].min()),
                    max_value=float(df[feature].max()),
                    value=float(df[feature].mean())
                )
            input_data[feature] = val
    for k, mapping in categorical_mappings.items():
        input_data[k] = mapping[input_data[k]]
    return pd.DataFrame([input_data]), input_data
input_df, input_data = user_input_features()
# Button to collect inputs
if st.sidebar.button("Get Prediction"):

    # Preprocessing
    numerical_features = df[features].select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    scaler.fit(df[numerical_features])
    input_scaled = scaler.transform(input_df[numerical_features])

    # Title & Input Summary
    st.title("🔬 Academic Heart Disease Prediction Dashboard")
    st.subheader("📝 Patient Input Summary")
    st.dataframe(input_df.style.highlight_max(axis=1))

    # Prediction
    st.subheader("📡 Prediction Result")
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    # Define thresholds for decision-making
    threshold = 0.5  # Default threshold for disease prediction
    high_risk_threshold = 0.75  # High risk threshold
    low_risk_threshold = 0.25  # Low risk threshold

    # Extract the probability values
    probability_disease = prediction_proba[0][1]
    probability_no_disease = prediction_proba[0][0]

    # Display based on the probability threshold
    if probability_disease > threshold:
        st.error(f"⚠️ **Heart Disease Detected** with {probability_disease*100:.2f}% confidence")
    else:
        st.success(f"✅ **No Heart Disease Detected** with {probability_no_disease*100:.2f}% confidence")

    # Display Prediction Result
    st.subheader("📡 Heart Disease Prediction Result")

    # Display diagnostic result based on probability
    if probability_disease > high_risk_threshold:
        st.error(f"⚠️ **High Risk of Heart Disease Detected** with {probability_disease*100:.2f}% confidence.")
        st.warning("It is recommended to consult with a healthcare provider for further diagnosis.")
    elif probability_disease > low_risk_threshold:
        st.warning(f"🟠 **Moderate Risk of Heart Disease** with {probability_disease*100:.2f}% confidence.")
        st.info("Consider lifestyle changes and monitor symptoms. Consult a doctor for further evaluation.")
    else:
        st.success(f"✅ **Low Risk of Heart Disease Detected** with {probability_no_disease*100:.2f}% confidence.")
        st.info("Keep maintaining a healthy lifestyle. Regular check-ups are advised for precaution.")

    # Additional insights based on specific features:
    # Example: If the cholesterol level is high, highlight it in the results.
    if input_df['Cholesterol'].values[0] > 240:  # Assuming a high cholesterol level
        st.warning("⚠️ **High Cholesterol** detected. This is a key risk factor for heart disease.")
    elif input_df['Cholesterol'].values[0] < 120:  # Assuming very low cholesterol
        st.info("🟢 **Cholesterol levels are on the lower side**, which is generally good for heart health.")

    # Example: If the Blood Pressure (BP) level is high or low, highlight it in the results.
    if input_df['BP'].values[0] > 140:  # High BP (Hypertension)
        st.warning("⚠️ **High Blood Pressure (BP)** detected. High BP is a significant risk factor for heart disease.")

    # Display metrics with probability values
    st.metric(label="💚 Probability (No Disease)", value=f"{probability_no_disease*100:.2f}%")
    st.metric(label="🩺 Probability (Disease)", value=f"{probability_disease*100:.2f}%")

    # Additional message for moderate confidence predictions
    if probability_disease > 0.5 and probability_disease < 0.75:
        st.info("ℹ️ **Moderate Confidence**: The model has some uncertainty in this prediction, and a consultation with a doctor is advised.")

    # Suggest next steps or further investigation
    st.markdown("---")
    st.subheader("🩺 Suggested Actions:")
    response = generate_health_suggestion(input_data)
    st.write(response)

    # ------------------ Insights & Visualizations ------------------
    st.markdown("---")
    st.header("📊 Medical Data Visual Insights")

    def dual_plot(fig_func1, fig_func2, title1, title2):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(title1)
            fig_func1()
        with col2:
            st.subheader(title2)
            fig_func2()

    # ============ Insight Plots ============
    def plot_chest_pain_distribution():
        fig, ax = plt.subplots()
        sns.countplot(x='Chest pain type', hue='Heart Disease', data=df, palette='Set2', ax=ax)
        ax.set_title("Chest Pain Type vs Heart Disease")
        st.pyplot(fig)

    def plot_sex_distribution():
        fig, ax = plt.subplots()
        sns.countplot(x='Sex', hue='Heart Disease', data=df, palette='pastel', ax=ax)
        ax.set_xticklabels(['Female', 'Male'])
        ax.set_title("Gender Distribution vs Heart Disease")
        st.pyplot(fig)

    def plot_thalach_comparison():
        fig, ax = plt.subplots()
        sns.boxplot(x='Heart Disease', y='Max HR', data=df, palette='coolwarm', ax=ax)
        ax.axhline(input_df['Max HR'][0], color='green', linestyle='--', label="Your Value")
        ax.set_xticklabels(['No Disease', 'Disease'])
        ax.set_title("Max Heart Rate vs Heart Disease")
        ax.legend()
        st.pyplot(fig)

    def plot_age_distribution():
        fig, ax = plt.subplots()
        sns.histplot(data=df, x='Age', hue='Heart Disease', multiple='stack', bins=15, ax=ax)
        ax.axvline(input_df['Age'][0], color='purple', linestyle='--', label="Your Age")
        ax.set_title("Age Distribution by Heart Disease")
        ax.legend()
        st.pyplot(fig)

    def plot_cholesterol_comparison():
        fig, ax = plt.subplots()
        sns.boxplot(x='Heart Disease', y='Cholesterol', data=df, ax=ax, palette="autumn")
        ax.axhline(input_df['Cholesterol'][0], color='blue', linestyle='--', label="Your Cholesterol")
        ax.set_xticklabels(['No Disease', 'Disease'])
        ax.set_title("Cholesterol Levels by Heart Disease")
        ax.legend()
        st.pyplot(fig)

    def plot_exercise_angina():
        fig, ax = plt.subplots()
        sns.countplot(x='Exercise angina', hue='Heart Disease', data=df, ax=ax, palette="Set1")
        ax.set_xticklabels(['No', 'Yes'])
        ax.set_title("Exercise-Induced Angina")
        st.pyplot(fig)

    def plot_resting_bp():
        fig, ax = plt.subplots()
        sns.kdeplot(data=df, x='BP', hue='Heart Disease', fill=True, ax=ax)
        ax.axvline(input_df['BP'][0], color='red', linestyle='--', label="Your BP")
        ax.set_title("Resting BP Distribution")
        ax.legend()
        st.pyplot(fig)

    def plot_oldpeak():
        fig, ax = plt.subplots()
        sns.histplot(data=df, x='ST depression', hue='Heart Disease', bins=20, multiple="dodge", ax=ax)
        ax.axvline(input_df['ST depression'][0], color='orange', linestyle='--', label="Your Oldpeak")
        ax.set_title("ST Depression (Oldpeak) vs Heart Disease")
        ax.legend()
        st.pyplot(fig)

    def plot_slope_vs_hd():
        fig, ax = plt.subplots()
        sns.countplot(x='Slope of ST', hue='Heart Disease', data=df, ax=ax, palette="Set3")
        ax.set_title("ST Slope vs Heart Disease")
        st.pyplot(fig)

    def plot_thallium_scan():
        fig, ax = plt.subplots()
        sns.countplot(x='Thallium', hue='Heart Disease', data=df, ax=ax, palette="coolwarm")
        ax.set_title("Thallium Scan Result vs Heart Disease")
        st.pyplot(fig)

    # ============ Display in 5 Rows of 2 ============
    dual_plot(plot_chest_pain_distribution, plot_sex_distribution,
              "Chest Pain Type Distribution", "Gender and Heart Disease")

    dual_plot(plot_thalach_comparison, plot_age_distribution,
              "Max Heart Rate Comparison", "Age vs Heart Disease")

    dual_plot(plot_cholesterol_comparison, plot_exercise_angina,
              "Cholesterol Levels", "Exercise-Induced Angina")

    dual_plot(plot_resting_bp, plot_oldpeak,
              "Resting BP Distribution", "ST Depression Level (Oldpeak)")

    dual_plot(plot_slope_vs_hd, plot_thallium_scan,
              "Slope of ST vs Disease", "Thallium Scan vs Disease")

    st.markdown("---")
    st.caption("📘 Powered by predictive modeling using medical dataset. For educational purposes — consult a doctor for clinical decisions.")

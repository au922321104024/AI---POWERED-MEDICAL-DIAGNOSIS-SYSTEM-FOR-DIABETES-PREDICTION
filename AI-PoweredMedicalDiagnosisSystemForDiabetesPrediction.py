import streamlit as st   
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Function to load the dataset
def load_data(url, columns):
    return pd.read_csv(url, names=columns)


# Function to preprocess the data 
def preprocess_data(data):
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]
    return X, y


# Function to split the data into training and testing sets
def split_data(X, y):
    return train_test_split(X, y, test_size=0.48, random_state=45)


# Function to scale the features
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# Function to select and train the model
def train_selected_model(model_type, X_train_scaled, y_train):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42),
        "SVM": SVC(probability=True, random_state=42),\
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "LightGBM": LGBMClassifier(random_state=42)
    }
    model = models.get(model_type, RandomForestClassifier(random_state=42))
    model.fit(X_train_scaled, y_train)
    return model


# Modified function to evaluate the model with all metrics
def evaluate_model(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    return accuracy, precision, recall, f1, roc_auc, fpr, tpr


# Function to plot ROC curve
def plot_roc_curve(fpr, tpr, roc_auc, model_type):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_type} ROC Curve")
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    st.pyplot(plt)


# Function to make predictions on new data
def make_prediction(model, scaler, input_data):
    input_data_scaled = scaler.transform(input_data)
    return model.predict(input_data_scaled)


# Main function
def main():
    # Custom CSS styling with !important to override Streamlit defaults
    st.markdown("""
        <style>
            body, .stApp {
                background-color: #e6f2ff !important;
                color: #000000;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            section[data-testid="stSidebar"] {
                background-color: #2C3E50;
                color: white;
            }
            section[data-testid="stSidebar"] * {
                font-size: 26px !important;
                color: white !important;
            }
            section[data-testid="stSidebar"] select,
            section[data-testid="stSidebar"] select option {
                font-size: 16px !important;
                background-color: white;
                color: black;
                padding: 6px;
                border: 1px solid #ccc;
                border-radius: 4px;
                appearance: none;
            }
            div[data-baseweb="select"] div[role="button"] {
                color: black !important;
            }
            div[data-baseweb="select"] div[role="listbox"] span {
                font-size:15px;
            }
            button[data-baseweb="tab"] > div {
                font-size: 25px !important;
                font-weight: bold !important;
            }
            table {
                border-collapse: collapse;
                width: 80%;
                margin-left: auto;
                margin-right: auto;
            }
            th, td {
                font-size: 20px !important;
                padding: 10px !important;
                text-align: center !important;
            }
            tr:nth-child(even) {
                background-color: #d0e7ff;
            }
            tr:hover {
                background-color: #a8cfff;
            }
            thead {
                background-color: #6495ED;
                color: white;
            }
            .input-label {
                font-size: 25px !important;
                font-weight: bold;
                margin-top: 10px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a Section", ["Home", "Model Training"])

    st.markdown("""
    <div style="text-align: center;">
        <h1 style="color: #FF0000;font-size: 50px;">AI-Powered Medical Diagnosis System</h1>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    div[data-baseweb="select"] div[role="button"] {
        color: black !important;
    }
    div[data-baseweb="select"] div[role="listbox"] span {
        font-size:15px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.title("Choose Model")
    model_type = st.sidebar.selectbox(
        "Select the machine learning model",
        ("Random Forest", "Logistic Regression", "SVM","XGBoost", "LightGBM")
    )
    st.sidebar.write(f"You selected: {model_type}")

    # Load and preprocess
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    ]
    data = load_data(url, columns)
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    if page == "Home":
        st.markdown("""
        <div style="background-color: #808080; color: white; padding: 50px 20px; text-align: center;">
            <h2 style="font-size: 45px;">Revolutionizing Healthcare with AI-Driven Diagnoses</h2>
            <p style="font-size: 40px;">Empowering healthcare professionals to make faster and more accurate diagnoses.</p>
        </div>
        """, unsafe_allow_html=True)

    elif page == "Model Training":
        st.write("### Model Training and Prediction")

        st.markdown("""
        <style>
        button[data-baseweb="tab"] > div {
            font-size: 25px !important;
            font-weight: bold !important;
        }
        </style>
        """, unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["Data Set Preview", "Enter Patient Data", f"{model_type} ROC Curve"])

        with tab1:
            st.header("Data Set Preview")
            st.markdown("""
            <style>
            .center-table-wrapper {
                display: flex;
                justify-content: center;
            }
            table {
                border-collapse: collapse;
                width: 80%;
            }
            th, td {
                font-size: 20px !important;
                padding: 10px !important;
                text-align: center !important;
            }
            thead {
                background-color: #333;
                color: white;
            }
            </style>
            """, unsafe_allow_html=True)
            st.markdown(
                f'<div class="center-table-wrapper">{data.head(25).to_html(index=False)}</div>',
                unsafe_allow_html=True
            )

        with tab2:
            st.header("Enter Patient Data")
            st.markdown("""
            <style>
            .input-label {
                font-size: 25px !important;
                font-weight: bold;
                margin-top: 10px;
            }
            </style>
            """, unsafe_allow_html=True)

            st.markdown('<div class="input-label">Pregnancies</div>', unsafe_allow_html=True)
            pregnancies = st.number_input("", min_value=0, max_value=20, value=1)

            st.markdown('<div class="input-label">Glucose</div>', unsafe_allow_html=True)
            glucose = st.number_input("", min_value=0, max_value=300, value=120)

            st.markdown('<div class="input-label">Blood Pressure</div>', unsafe_allow_html=True)
            bp = st.number_input("", min_value=0, max_value=200, value=70)

            st.markdown('<div class="input-label">Skin Thickness</div>', unsafe_allow_html=True)
            skin = st.number_input("", min_value=0, max_value=100, value=20)

            st.markdown('<div class="input-label">Insulin</div>', unsafe_allow_html=True)
            insulin = st.number_input("", min_value=0, max_value=900, value=80)

            st.markdown('<div class="input-label">BMI</div>', unsafe_allow_html=True)
            bmi = st.number_input("", min_value=0.0, max_value=70.0, value=30.0)

            st.markdown('<div class="input-label">Diabetes Pedigree Function</div>', unsafe_allow_html=True)
            dpf = st.number_input("", min_value=0.0, max_value=2.5, value=0.5)

            st.markdown('<div class="input-label">Age</div>', unsafe_allow_html=True)
            age = st.number_input("", min_value=0, max_value=120, value=30)

            input_data_array = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
            model = train_selected_model(model_type, X_train_scaled, y_train)
            prediction = make_prediction(model, scaler, input_data_array)
            accuracy, precision, recall, f1, roc_auc, fpr, tpr = evaluate_model(model, X_test_scaled, y_test)

            st.write(f"### {model_type} Accuracy: {accuracy * 100:.2f}%")
            if prediction[0] == 1:
                st.write("### The patient is likely to have diabetes.")
            else:
                st.write("### The patient is not likely to have diabetes.")

        with tab3:
            st.markdown(f'<h1 style="font-size:40px;">{model_type} ROC Curve</h1>', unsafe_allow_html=True)
            st.markdown(f'<p style="font-size:28px;">ROC AUC Score: {roc_auc:.4f}</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="font-size:28px;">Accuracy: {accuracy * 100:.2f}%</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="font-size:24px;">Precision: {precision:.4f}</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="font-size:24px;">Recall: {recall:.4f}</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="font-size:24px;">F1-Score: {f1:.4f}</p>', unsafe_allow_html=True)

            plot_roc_curve(fpr, tpr, roc_auc, model_type)


if __name__ == "__main__":
    main()

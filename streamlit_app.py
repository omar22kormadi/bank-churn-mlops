import os
import glob
import tempfile
import joblib
import pandas as pd
import streamlit as st

FEATURE_NAMES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Geography_Germany",
    "Geography_Spain",
]


def try_load_model():
    # Try common paths for a serialized model
    candidates = [
        "model/model.pkl",
        "model/model.joblib",
        "app/model.pkl",
    ]

    for p in candidates:
        if os.path.exists(p):
            try:
                return joblib.load(p), p
            except Exception:
                pass

    # Try to find MLflow model artifacts in mlruns
    for path in glob.glob("mlruns/**/artifacts/model", recursive=True):
        try:
            import mlflow.pyfunc

            m = mlflow.pyfunc.load_model(path)
            return m, path
        except Exception:
            continue

    return None, None


def load_uploaded_model(uploaded_file):
    if uploaded_file is None:
        return None
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp.flush()
        try:
            return joblib.load(tmp.name)
        except Exception:
            try:
                import mlflow.pyfunc

                return mlflow.pyfunc.load_model(tmp.name)
            except Exception:
                return None


def build_input_dataframe(values: dict) -> pd.DataFrame:
    # Ensure column order
    df = pd.DataFrame([values], columns=FEATURE_NAMES)
    return df


def main():
    st.title("Bank Customer Churn — Demo")

    st.sidebar.header("Model")
    uploaded = st.sidebar.file_uploader("Upload pickled model (.pkl, .joblib)", type=["pkl", "joblib"]) 
    model = None
    model_path = None

    if uploaded is not None:
        model = load_uploaded_model(uploaded)
        model_path = f"uploaded: {uploaded.name}"

    if model is None:
        model, model_path = try_load_model()

    st.sidebar.markdown(f"**Model:** {model_path or 'Not found — upload a model.'}")

    st.header("Customer features")

    # Inputs
    cs = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    tenure = st.slider("Tenure (years)", 0, 10, 5)
    balance = st.number_input("Balance", min_value=0.0, value=50000.0)
    nprod = st.selectbox("Number of Products", [1, 2, 3, 4], index=1)
    has_card = st.radio("Has Credit Card", [0, 1], index=1)
    active = st.radio("Is Active Member", [0, 1], index=1)
    salary = st.number_input("Estimated Salary", min_value=0.0, value=75000.0)
    geo = st.selectbox("Geography", ["France", "Germany", "Spain"])  # France is baseline

    geo_germany = 1 if geo == "Germany" else 0
    geo_spain = 1 if geo == "Spain" else 0

    values = {
        "CreditScore": int(cs),
        "Age": int(age),
        "Tenure": int(tenure),
        "Balance": float(balance),
        "NumOfProducts": int(nprod),
        "HasCrCard": int(has_card),
        "IsActiveMember": int(active),
        "EstimatedSalary": float(salary),
        "Geography_Germany": int(geo_germany),
        "Geography_Spain": int(geo_spain),
    }

    X = build_input_dataframe(values)

    if st.button("Predict"):
        if model is None:
            st.error("No model available. Upload a pickled model or place one in `model/` or mlruns artifacts.")
        else:
            try:
                if hasattr(model, "predict_proba"):
                    prob = float(model.predict_proba(X)[:, 1][0])
                else:
                    # fallback: predict returns 0/1
                    pred = int(model.predict(X)[0])
                    prob = 1.0 if pred == 1 else 0.0

                pred_label = 1 if prob >= 0.5 else 0
                if prob < 0.33:
                    risk = "Low"
                elif prob < 0.66:
                    risk = "Medium"
                else:
                    risk = "High"

                st.subheader("Prediction result")
                st.write({"churn_probability": round(prob, 4), "prediction": pred_label, "risk_level": risk})

                # Feature importances if available
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                    fi = pd.Series(importances, index=FEATURE_NAMES).sort_values(ascending=False)
                    st.subheader("Feature importances")
                    st.bar_chart(fi)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    st.markdown("---")
    st.write("Tip: If you don't have a serialized model, train and save one or upload a `.pkl` file.")


if __name__ == "__main__":
    main()

"""app.py - Streamlit prototype: Student Performance Prediction System."""
import io
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load("student_model.pkl")


try:
    bundle = load_bundle()
except FileNotFoundError:
    st.error("student_model.pkl not found. Run `python train_model.py` first.")
    st.stop()

model, features, schema, metrics = bundle["model"], bundle["features"], bundle["schema"], bundle["metrics"]

st.title("🎓 Student Performance Prediction System")
st.caption("Classroom demonstration using public, historical data (UCI Student Performance - Mathematics). "
           "Predictions are NOT diagnoses and must not be the sole basis for real student decisions.")

with st.sidebar:
    st.header("Model summary")
    st.metric("Accuracy (test set)", f"{metrics['accuracy']:.1%}")
    cm = metrics["confusion_matrix"]
    st.write(f"Records used: **{metrics['rows']}**")
    st.write(f"At-risk share: **{metrics['at_risk_share']:.1%}**")
    st.write("Rule: **G3 < 10 = AT RISK**; G3 >= 10 = NOT AT RISK")
    st.write(f"Confusion matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")
    st.warning("Key limitation: trained on one Portuguese secondary-school dataset; may not generalize "
               "to another institution or population.")


def label(p):
    return "AT RISK" if p == 1 else "NOT AT RISK"


tab1, tab2 = st.tabs(["📁 Upload CSV (batch)", "🧍 Single student"])

# ---------------- Batch upload ----------------
with tab1:
    st.subheader("Upload a CSV file")
    st.write("The file must contain these columns (G3 is not needed and is ignored if present):")
    st.code(", ".join(features))
    up = st.file_uploader("Choose a CSV file", type=["csv"])
    if up is not None:
        try:
            data = pd.read_csv(up, sep=None, engine="python")   # auto-detects , or ;
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()
        missing = [c for c in features if c not in data.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            if "G3" in data.columns:
                st.info("G3 column found - it is ignored (G3 must never be an input).")
            X = data[features]
            proba = model.predict_proba(X)[:, 1]
            pred = (proba >= 0.5).astype(int)
            out = data.copy()
            out["risk_probability"] = proba.round(3)
            out["prediction"] = [label(p) for p in pred]
            c1, c2, c3 = st.columns(3)
            c1.metric("Students", len(out))
            c2.metric("AT RISK", int(pred.sum()))
            c3.metric("NOT AT RISK", int(len(pred) - pred.sum()))
            st.dataframe(out, use_container_width=True)
            st.download_button("⬇️ Export results (CSV)", out.to_csv(index=False).encode("utf-8"),
                               "prediction_results.csv", "text/csv")

# ---------------- Single student ----------------
with tab2:
    st.subheader("Enter student information")
    values = {}
    cols = st.columns(3)
    i = 0
    for name in features:
        with cols[i % 3]:
            if name in schema["numeric"]:
                s = schema["numeric"][name]
                values[name] = st.number_input(name, min_value=int(s["min"]), max_value=int(s["max"]),
                                               value=int(round(s["median"])), step=1)
            else:
                opts = schema["categorical"][name]
                values[name] = st.selectbox(name, opts)
        i += 1
    if st.button("Predict risk", type="primary"):
        row = pd.DataFrame([values])[features]
        p = float(model.predict_proba(row)[0, 1])
        if p >= 0.5:
            st.error(f"Prediction: AT RISK (probability {p:.1%})")
        else:
            st.success(f"Prediction: NOT AT RISK (risk probability {p:.1%})")

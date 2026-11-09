"""
train_model.py
Student Performance Prediction System - Data Mining Laboratory
Steps: (A) retrieve + inspect data, (B) preprocess, train, evaluate, save model.
Target rule: G3 < 10 -> AT RISK (1); G3 >= 10 -> NOT AT RISK (0). G3 is NEVER an input feature.
"""
import io
import json
import os
import urllib.request
import zipfile
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Always work inside the folder that contains this script (fixes "file not found" when
# the terminal is opened in a different folder).
os.chdir(Path(__file__).resolve().parent)

ZIP_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
CSV_PATH = "student_data.csv"
MODEL_PATH = "student_model.pkl"


# ------------------------------------------------------------------
# ACTIVITY A - Retrieve and inspect the dataset
# ------------------------------------------------------------------
def fetch_math_dataset() -> pd.DataFrame:
    """Return the MATHEMATICS file (student-mat.csv, 395 rows): local copy first, else download."""
    if os.path.exists("student-mat.csv"):
        print("Using local student-mat.csv")
        return pd.read_csv("student-mat.csv", sep=None, engine="python")
    try:
        print("Downloading UCI Student Performance dataset (ID 320)...")
        raw = urllib.request.urlopen(ZIP_URL, timeout=60).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as outer:
            # The archive may contain student.zip (nested) or the csv files directly.
            names = outer.namelist()
            if "student-mat.csv" in names:
                return pd.read_csv(outer.open("student-mat.csv"), sep=";")
            for n in names:
                if n.endswith(".zip"):
                    with zipfile.ZipFile(io.BytesIO(outer.read(n))) as inner:
                        if "student-mat.csv" in inner.namelist():
                            return pd.read_csv(inner.open("student-mat.csv"), sep=";")
        raise FileNotFoundError("student-mat.csv not found inside the downloaded zip")
    except Exception as err:  # fall back to ucimlrepo (as in the lab handout)
        print(f"Direct download failed ({err}). Falling back to ucimlrepo...")
        try:
            from ucimlrepo import fetch_ucirepo
            student = fetch_ucirepo(id=320)
        except Exception as err2:
            raise SystemExit(
                "\nCould not get the dataset (" + str(err2) + ").\n"
                "Manual fix: download https://archive.ics.uci.edu/dataset/320/student+performance ,\n"
                "extract student-mat.csv, put it in this folder (" + os.getcwd() + ") and run again."
            )
        df = pd.concat([student.data.features, student.data.targets], axis=1)
        print("WARNING: ucimlrepo may return the Portuguese records (649 rows), "
              "not Mathematics (395 rows). Check the shape printed below.")
        return df


if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH, sep=None, engine="python")   # handles , or ; separators
    print(f"Loaded existing {CSV_PATH}")
else:
    df = fetch_math_dataset()
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved {CSV_PATH}")

print("\nRows and columns:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nData types:\n", df.dtypes)
print("\nMissing values per column:\n", df.isnull().sum())

# ------------------------------------------------------------------
# ACTIVITY B - Preprocess, train and evaluate
# ------------------------------------------------------------------
df = df.drop_duplicates()
y = (df["G3"] < 10).astype(int)          # 1 = AT RISK, 0 = NOT AT RISK
X = df.drop(columns=["G3"])               # remove target -> prevents target leakage

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

preprocess = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), numeric_cols),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
])

model = Pipeline([
    ("preprocess", preprocess),
    ("clf", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)
report_txt = classification_report(y_test, pred, target_names=["NOT AT RISK", "AT RISK"])
cm = confusion_matrix(y_test, pred)

print("\nClass balance (1 = at risk):\n", y.value_counts().to_string())
print(f"\nAccuracy: {acc:.3f}")
print("\nClassification report:\n", report_txt)
print("Confusion matrix [[TN FP],[FN TP]]:\n", cm)

# Top features (optional analysis)
names = model.named_steps["preprocess"].get_feature_names_out()
imp = pd.Series(model.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)
print("\nTop 10 feature importances:\n", imp.head(10).to_string())

# ------------------------------------------------------------------
# Save model + schema (used by app.py to build the form and validate uploads)
# ------------------------------------------------------------------
schema = {
    "numeric": {c: {"min": float(X[c].min()), "max": float(X[c].max()),
                    "median": float(X[c].median())} for c in numeric_cols},
    "categorical": {c: sorted(X[c].dropna().astype(str).unique().tolist()) for c in categorical_cols},
}
metrics = {
    "accuracy": float(acc),
    "rows": int(df.shape[0]),
    "at_risk_share": float(y.mean()),
    "confusion_matrix": cm.tolist(),
    "report": classification_report(y_test, pred, target_names=["NOT AT RISK", "AT RISK"], output_dict=True),
}
joblib.dump({"model": model, "features": X.columns.tolist(), "schema": schema, "metrics": metrics}, MODEL_PATH)

with open("classification_report.txt", "w") as f:
    f.write(f"Accuracy: {acc:.3f}\n\n{report_txt}\nConfusion matrix [[TN FP],[FN TP]]:\n{cm}\n")
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nSaved {MODEL_PATH}, classification_report.txt, metrics.json")

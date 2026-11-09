
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("STARTING MODEL TRAINING...")

df = pd.read_csv("student_data.csv")
df = df.drop_duplicates().copy()

y = (df["G3"] < 10).astype(int)
X = df.drop(columns=["G3"])

numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MODEL ACCURACY:", accuracy_score(y_test, y_pred) * 100, "%")
print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))
print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, zero_division=0))

artifact = {
    "model": model,
    "feature_columns": X.columns.tolist(),
    "numeric_features": numeric_features,
    "categorical_features": categorical_features
}

joblib.dump(artifact, "student_model.pkl")

print("\nMODEL SAVED SUCCESSFULLY!")
print("TRAINING COMPLETED!")

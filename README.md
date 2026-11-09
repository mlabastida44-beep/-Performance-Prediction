# Student Performance Prediction System

Data Mining Laboratory prototype (Python · GitHub · Streamlit · Predictive Modeling · SAD).

## 1. Problem
A school wants to explore whether historical student records can help flag learners who may need additional
academic support. The system predicts whether a student is **AT RISK of failing final Mathematics**.

**Prediction rule:** `G3 < 10` = AT RISK (1); `G3 >= 10` = NOT AT RISK (0).
`G3` is the target only and is **never** used as an input feature.

> **Responsible use:** classroom demonstration with public, historical data. Predictions are not diagnoses
> and must not be the sole basis for real student decisions.

## 2. Dataset
UCI Machine Learning Repository - Student Performance (ID 320), **Mathematics** file (`student-mat.csv`, 395 rows).
https://archive.ics.uci.edu/dataset/320/student+performance

## 3. Project structure
```
student-performance-prediction/
├── train_model.py     # download, preprocess, train Random Forest, evaluate, save model
├── app.py             # Streamlit app (CSV upload + single-student form)
├── requirements.txt
└── README.md
```

## 4. How to run
```bash
pip install -r requirements.txt
python train_model.py        # creates student_data.csv and student_model.pkl
streamlit run app.py         # opens http://localhost:8501
```

## 5. Method
Remove duplicates → define target (G3 < 10) → drop G3 from features → impute (median / most frequent) →
scale numeric → one-hot encode categorical → Random Forest (200 trees, balanced class weights) →
80/20 stratified split → accuracy + classification report.

## 6. Results
See `classification_report.txt` after running `train_model.py`. (Paste your accuracy / recall for AT RISK here.)

## 7. SAD mapping
| Stage | In this project |
|---|---|
| Problem | Late detection of academic risk |
| Requirements | Upload CSV, validate columns, predict, display, export |
| Design | Pipeline: validate → preprocess → model → label; Streamlit UI |
| Implementation | `train_model.py`, `app.py`, GitHub repo |
| Testing | Valid CSV, missing-column CSV, single-student form, metrics review |

## 8. Limitations
- Trained on one Portuguese secondary-school dataset; may not generalize to other institutions or populations.
- G1 and G2 (earlier grades) are strong predictors, so the system works best *after* the first two grading periods.
- Dataset is small (395 rows); results vary with the split.
- Contains sensitive attributes (sex, family background); risk of bias, so use for support, never for punishment.

## 9. Author
Name(s): ________  Section: ________  Date: November 9, 2026

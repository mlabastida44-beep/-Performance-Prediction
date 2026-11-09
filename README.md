
# Student Performance Prediction System

## Project Description

This project predicts whether students may be
academically at risk using machine learning.

## Dataset

UCI Student Performance Dataset
Mathematics Records.

## Algorithm

Random Forest Classifier.

## Prediction Categories

- AT RISK: Final grade below 10
- NOT AT RISK: Final grade 10 or above

## Technologies Used

- Python
- Google Colab
- Pandas
- Scikit-learn
- Streamlit
- GitHub

## Project Files

- train_model.py
- app.py
- requirements.txt
- README.md
- student_data.csv
- student_model.pkl

## How to Run

Install dependencies:

pip install -r requirements.txt

Train the model:

python train_model.py

Run the application:

streamlit run app.py

## Limitation

The dataset contains historical information
from Portuguese secondary schools.

The model may not generalize to other
institutions or populations.

Predictions are for educational purposes only.

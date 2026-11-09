"""Creates test_valid.csv (20 real rows, no G3) and test_missing_cols.csv (to trigger the validation error)."""
import pandas as pd
df = pd.read_csv("student_data.csv")
df.sample(20, random_state=1).drop(columns=["G3"]).to_csv("test_valid.csv", index=False)
df.sample(20, random_state=1).drop(columns=["G3", "G1", "G2"]).to_csv("test_missing_cols.csv", index=False)
print("Created test_valid.csv and test_missing_cols.csv")

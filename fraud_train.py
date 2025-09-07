# fraud_train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Try to load Kaggle dataset, else create fake dataset
try:
    df = pd.read_csv("creditcard.csv")   # Kaggle dataset
    df = df.rename(columns={"Class": "isFraud"})  # make column consistent
except:
    print("Dataset not found, using a small fake dataset...")
    np.random.seed(42)
    df = pd.DataFrame({
        "amount": np.random.uniform(10, 5000, 1000),
        "hour": np.random.randint(0, 24, 1000),
        "card_present": np.random.randint(0, 2, 1000),
    })
    df["isFraud"] = ((df["amount"] > 3000) & (df["card_present"] == 0)).astype(int)

# Split
X = df.drop("isFraud", axis=1)
y = df["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save
joblib.dump(model, "fraud_model.joblib")
joblib.dump(list(X.columns), "features.pkl")
print("✅ Model trained and saved as fraud_model.joblib + features.pkl")

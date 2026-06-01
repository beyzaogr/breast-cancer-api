import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib


df = pd.read_csv("cancer_risk.csv")


df = df[df["Cancer_Type"] == "Breast"]


df["target"] = (df["Risk_Level"] == "High").astype(int)


X = df.drop(columns=[
    "Patient_ID",
    "Cancer_Type",
    "Risk_Level",
    "target",
    "Overall_Risk_Score"   
])

y = df["target"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


joblib.dump(model, "breast_cancer_model.pkl")

print("✔ Yeni model kaydedildi")
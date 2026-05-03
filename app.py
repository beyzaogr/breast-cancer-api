from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import sessionmaker, declarative_base

# ==========================
# 1) FLASK UYGULAMASI
# ==========================
app = Flask(__name__)
CORS(app)

# ==========================
# 2) DATABASE BAĞLANTI
# ==========================
DATABASE_URL = "mysql+pymysql://root:KYKapevzfPrTzVMwDpReQwwThExlyvrV@switchback.proxy.rlwy.net:52815/railway"

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
Base = declarative_base()

# ==========================
# 3) MODEL YÜKLE
# ==========================
model = joblib.load("breast_cancer_model.pkl")

# ==========================
# 4) TABLO MODELİ
# ==========================
class Result(Base):
    __tablename__ = "results"

    id = Column(Integer, primary_key=True)
    risk = Column(Integer)
    date = Column(String(50))
    answers = Column(Text)

# 🔥 TABLOYU OLUŞTUR
Base.metadata.create_all(engine)

# ==========================
# 5) FEATURE LIST
# ==========================
FEATURE_NAMES = [
    "Age", "Gender", "Smoking", "Alcohol_Use", "Obesity",
    "Family_History", "Diet_Red_Meat", "Diet_Salted_Processed",
    "Fruit_Veg_Intake", "Physical_Activity", "Air_Pollution",
    "Occupational_Hazards", "BRCA_Mutation", "H_Pylori_Infection",
    "Calcium_Intake", "BMI", "Physical_Activity_Level"
]

# Feature importance
print("\n📊 FEATURE IMPORTANCE:\n")
for name, importance in zip(FEATURE_NAMES, model.feature_importances_):
    print(f"{name}: {importance:.4f}")

RISK_LABELS = {
    0: "Düşük",
    1: "Orta",
    2: "Yüksek"
}

# ==========================
# 6) TEST ENDPOINT
# ==========================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Breast cancer risk API çalışıyor 🚀"}), 200

# ==========================
# 7) PREDICT
# ==========================
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    missing = [f for f in FEATURE_NAMES if f not in data]
    if missing:
        return jsonify({
            "error": "Eksik alanlar var",
            "missing_fields": missing
        }), 400

    df_input = pd.DataFrame([[data[f] for f in FEATURE_NAMES]],
                            columns=FEATURE_NAMES)

    pred_class = int(model.predict(df_input)[0])
    probas = model.predict_proba(df_input)[0]

    # ==========================
    # MODEL SONUÇ
    # ==========================
    if len(probas) == 2:
        low = float(probas[0])
        high = float(probas[1])
        medium = 0.0

        predicted_label = "Yüksek" if high >= 0.5 else "Düşük"

        result = {
            "predicted_class": pred_class,
            "predicted_label": predicted_label,
            "probabilities": {
                "low": low,
                "medium": medium,
                "high": high
            }
        }

    else:
        result = {
            "predicted_class": pred_class,
            "predicted_label": RISK_LABELS.get(pred_class, "Bilinmiyor"),
            "probabilities": {
                "low": float(probas[0]),
                "medium": float(probas[1]),
                "high": float(probas[2])
            }
        }

    # ==========================
    # 🔥 DATABASE KAYIT
    # ==========================
    session = Session()

    new_result = Result(
        risk=int(result["probabilities"]["high"] * 100),
        date="bugün",
        answers=str(data)
    )

    session.add(new_result)
    session.commit()
    session.close()

    return jsonify(result), 200


# ==========================
# 8) RUN
# ==========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

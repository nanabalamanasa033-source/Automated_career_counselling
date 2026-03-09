from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import os

# Initialize Flask app
app = Flask(__name__)

# Get current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------
# Load Machine Learning Model
# -------------------------
try:
    model_path = os.path.join(BASE_DIR, "model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print("Error loading model:", e)

# -------------------------
# Load Label Encoders
# -------------------------
try:
    encoder_path = os.path.join(BASE_DIR, "label_encoders.pkl")
    with open(encoder_path, "rb") as f:
        label_encoders = pickle.load(f)

    le = label_encoders["Career"]
    career_classes = dict(zip(le.transform(le.classes_), le.classes_))
except Exception as e:
    print("Error loading encoders:", e)

# -------------------------
# Home Route
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")

# -------------------------
# Prediction Route
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form

        encoded_data = {
            "GPA": float(data.get("GPA")),
            "Internships": int(data.get("Internships")),
            "Projects": int(data.get("Projects")),
            "Coding_Skills": int(data.get("Coding_Skills")),
            "Communication_Skills": int(data.get("Communication_Skills")),
            "Leadership_Experience": int(data.get("Leadership_Experience")),
            "Extracurricular_Activities": int(data.get("Extracurricular_Activities")),
            "Preferred_Work_Environment": int(data.get("Preferred_Work_Environment"))
        }

        df = pd.DataFrame([encoded_data])

        prediction = model.predict(df)[0]

        career = career_classes.get(prediction, "Unknown Career")

        return render_template("predict.html", career=career)

    except Exception as e:
        return jsonify({"error": str(e)})

# -------------------------
# Run App (Render Compatible)
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

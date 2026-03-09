from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import os

# Initialize Flask app
app = Flask(__name__)

# Get current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model
model_path = os.path.join(BASE_DIR, "model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load label encoders
encoder_path = os.path.join(BASE_DIR, "label_encoders.pkl")
with open(encoder_path, "rb") as f:
    label_encoders = pickle.load(f)

le = label_encoders["Career"]
career_classes = dict(zip(le.transform(le.classes_), le.classes_))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form

        encoded_data = {
            "GPA": float(data["GPA"]),
            "Internships": int(data["Internships"]),
            "Projects": int(data["Projects"]),
            "Coding_Skills": int(data["Coding_Skills"]),
            "Communication_Skills": int(data["Communication_Skills"]),
            "Leadership_Experience": int(data["Leadership_Experience"]),
            "Extracurricular_Activities": int(data["Extracurricular_Activities"]),
            "Preferred_Work_Environment": int(data["Preferred_Work_Environment"])
        }

        df = pd.DataFrame([encoded_data])

        prediction = model.predict(df)[0]

        career = ""
        if prediction in career_classes:
            career = career_classes[prediction]

        return render_template("predict.html", career=career)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)

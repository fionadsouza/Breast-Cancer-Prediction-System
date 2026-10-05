from flask import Flask, render_template, request
import joblib

app = Flask(__name__, template_folder="webpages")

model = joblib.load("breast_cancer_model_10f.pkl")
scaler = joblib.load("scaler_10f.pkl")

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    confidence = None
    form_data = {}

    if request.method == "POST":

        form_data = request.form

        values = [
            float(request.form["worst_texture"]),
            float(request.form["radius_error"]),
            float(request.form["worst_symmetry"]),
            float(request.form["mean_concave_points"]),
            float(request.form["worst_concavity"]),
            float(request.form["area_error"]),
            float(request.form["worst_radius"]),
            float(request.form["worst_area"]),
            float(request.form["mean_concavity"]),
            float(request.form["worst_concave_points"])
        ]

        values_scaled = scaler.transform([values])

        prediction = model.predict(values_scaled)

        probability = model.predict_proba(values_scaled)
        confidence = round(max(probability[0]) * 100, 2)

        if prediction[0] == 1:
            result = "🟢 Benign"
        else:
            result = "🔴 Malignant"

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        form_data=form_data
    )

if __name__ == "__main__":
    app.run(debug=True)
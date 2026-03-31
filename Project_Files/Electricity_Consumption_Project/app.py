from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("electricity_model.pkl")
state_encoder = joblib.load("state_encoder.pkl")
region_encoder = joblib.load("region_encoder.pkl")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/story")
def story():
    return render_template("story.html")


@app.route("/visualizations")
def visualizations():
    return render_template("visualizations.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


# Forecast page
@app.route("/prediction")
def prediction():
    return render_template("prediction.html")


@app.route("/predict_page")
def predict_page():
    return render_template("predict.html")

@app.route("/predict", methods=["POST"])
def predict():

    year = int(request.form["year"])
    month = int(request.form["month"])
    latitude = request.form.get("latitude")
    longitude = request.form.get("longitude")

# make optional
    latitude = float(latitude) if latitude else 0.0
    longitude = float(longitude) if longitude else 0.0

    state = request.form["state"]
    region = request.form["region"]

    state_encoded = state_encoder.transform([state])[0]
    region_encoded = region_encoder.transform([region])[0]

    year_index = year - 2019

    data = [[
        year,
        month,
        latitude,
        longitude,
        state_encoded,
        region_encoded,
        year_index
    ]]

    prediction = model.predict(data)[0]

    # Generate predictions for all months in the selected year
    months = []
    values = []

    for m in range(1,13):

        temp = [[
            year,
            m,
            latitude,
            longitude,
            state_encoded,
            region_encoded,
            year_index
        ]]

        pred = model.predict(temp)[0]

        months.append(m)
        values.append(round(pred,2))

    return render_template(
        "result.html",
        prediction=round(prediction,2),
        months=months,
        values=values,
        selected_year=year
    )

if __name__ == "__main__":
    app.run(debug=True)
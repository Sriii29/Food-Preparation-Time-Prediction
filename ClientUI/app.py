from flask import Flask, render_template, request
import requests

app = Flask(__name__)

FASTAPI_URL = "http://127.0.0.1:8000/predict"

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        data = {
            "items": request.form["items"],
            "hour": int(request.form["hour"]),
            "rain": int(request.form["rain"]),
            "event": int(request.form["event"]),
            "kitchen_load": int(request.form["kitchen_load"])
        }

        try:
            response = requests.post(FASTAPI_URL, json=data)
            prediction = response.json()
        except Exception as e:
            prediction = {"error": str(e)}

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
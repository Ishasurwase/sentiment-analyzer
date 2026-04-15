from flask import Flask, render_template, request
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)

# Load sentiment model
model = pipeline("sentiment-analysis")

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Analyze route
@app.route("/analyze", methods=["POST"])
def analyze():
    text = request.form["text"].strip()

    # If empty input
    if text == "":
        return render_template("index.html", text=text, label="NEUTRAL", score=0)

    result = model(text)[0]

    label = result["label"]
    score = result["score"]

    # Convert to neutral using smarter logic
    neutral_words = ["okay", "average", "fine", "normal", "not bad", "not good"]

    if any(word in text.lower() for word in neutral_words):
        label = "NEUTRAL"
    elif score < 0.7:
        label = "NEUTRAL"

    score = round(score, 2)

    return render_template("index.html", text=text, label=label, score=score)

# Run app
if __name__ == "__main__":
    app.run(debug=True)
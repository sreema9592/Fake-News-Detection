from flask import Flask, request, render_template_string
import pickle

# Load saved model and vectorizer
with open("fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Fake News Detector</title>
</head>
<body style="font-family:Arial; padding:40px">
    <h2>📰 Fake News Detection</h2>
    <form method="post">
        <textarea name="news" rows="8" cols="80" placeholder="Paste news text here"></textarea><br><br>
        <button type="submit">Check</button>
    </form>
    {% if result %}
        <h3>{{ result }}</h3>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        news = request.form["news"]
        data = tfidf.transform([news])
        prediction = model.predict(data)[0]
        result = "🚨 FAKE NEWS" if prediction == 1 else "✔ REAL NEWS"
    return render_template_string(HTML, result=result)

if __name__ == "__main__":
    app.run(debug=True)

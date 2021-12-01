import joblib
from flask import Flask
from flask import render_template, url_for, redirect, request, jsonify
import json
import tensorflow_hub as hub
import pandas as pd

from src.process import remove_low_tfidf
from src.get_data import load_wiki
app = Flask(__name__, template_folder="app/templates",
            static_folder="app/static")


@app.route("/", methods=['GET', 'POST'])
def index():
    with open('artifacts/results.json') as f:
        results = json.load(f)
    return render_template('index.html', results=results)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global google
    if request.method == 'POST':
        news = request.form['text']
        vectorizer = joblib.load('artifacts/TfidfVectorizer,vectorizer.pkl')
        scaler = joblib.load('artifacts/StandardScaler,google_scaler.pkl')
        model = joblib.load('artifacts/SVC,google_svc.pkl')

        cleaned_news = remove_low_tfidf(vectorizer, news)
        wiki_df = load_wiki([cleaned_news], n_results=1)
        if len(wiki_df) > 0:
            X_test = google([cleaned_news]) - google(wiki_df['wiki'])
            X_test_scaled = scaler.transform(X_test)

            wiki_df['predict'] = model.predict(X_test_scaled)

            relevant = wiki_df.loc[wiki_df["predict"] == 1, "title"]
            irrelevant = wiki_df.loc[wiki_df["predict"] == 0, "title"]

            results = [{"news": news,
                        "relevant": list(relevant),
                        "irrelevant": list(irrelevant)}]
            return render_template('index.html', results=results)
        else:
            return render_template('index.html', message='no Wiki results found for: ' + news)


@app.route("/data")
def data():
    with open('artifacts/results.json') as f:
        results = json.load(f)
    return jsonify(results)


if __name__ == '__main__':
    google = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    app.run(debug=True)

from flask import Flask
from flask import render_template, jsonify
import json
import tensorflow_hub as hub

app = Flask(__name__)


if 'google' not in globals():
    google = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


@app.route("/")
def index():
    with open('../artifacts/results.json') as f:
        results = json.load(f)
    return render_template('index.html', results=results)

@app.route("/data")
def data():
    with open('results.json') as f:
        results = json.load(f)
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)

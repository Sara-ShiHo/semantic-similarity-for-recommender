from flask import Flask
from flask import render_template, jsonify
import json

app = Flask(__name__)


@app.route("/")
def index():
    with open('results.json') as f:
        results = json.load(f)
    return render_template('index.html', results=results)

@app.route("/data")
def data():
    with open('results.json') as f:
        results = json.load(f)
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)

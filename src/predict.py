import logging
import pickle
import json

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.utils import resample

from process import clean_text, remove_low_tfidf
from embeddings import (
    get_vectors_glove,
    counter_embedding,
    get_vectors_from_count,
    get_cosine_sim,
    google,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

RANDOM_STATE = 414


def upsample(train_df):
    positive = train_df.loc[train_df["label"] == 1]
    negative = train_df.loc[train_df["label"] == 0]

    positive_upsampled = resample(
        positive, random_state=RANDOM_STATE, n_samples=len(negative) - len(positive)
    )

    return pd.concat([positive, positive_upsampled, negative])


def confusion_mat(actual, predicted):
    cm1 = confusion_matrix(actual, predicted)
    total1 = sum(sum(cm1))

    accuracy = (cm1[0, 0] + cm1[1, 1]) / total1
    sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    f1 = f1_score(actual, predicted)

    return {
        "accuracy": round(accuracy, 3),
        "sensitivity": round(sensitivity, 3),
        "specificity": round(specificity, 3),
        "f1": round(f1, 3),
    }


def exp_similarity_cutoff(labeled):
    sim = []
    for i, row in labeled[["news", "wiki"]].iterrows():
        clean_news = clean_text(row["news"])
        clean_wiki = clean_text(row["wiki"])

        news_count = counter_embedding(clean_news)
        wiki_count = counter_embedding(clean_wiki)

        news_vec, wiki_vec = get_vectors_from_count(news_count, wiki_count)
        count_similarities = get_cosine_sim(news_vec, wiki_vec)
        sim.append(count_similarities)

    labeled["sim"] = sim
    return labeled


def exp_ml_prediction(X_train, X_test, y_train, y_test):

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info("Logistic Regression")
    lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000).fit(
        X_train_scaled, y_train
    )
    predicted = lr.predict(X_train_scaled)
    logger.info("train predictions:")
    annotate(y_train, predicted)

    predicted = lr.predict(X_test_scaled)
    logger.info("test predictions:")
    annotate(y_test, predicted)

    logger.info("Support Vector Classifier")
    svc = SVC(random_state=RANDOM_STATE).fit(X_train_scaled, y_train)
    predicted = svc.predict(X_train_scaled)
    logger.info("train predictions:")
    annotate(y_train, predicted)

    predicted = svc.predict(X_test_scaled)
    logger.info("test predictions:")
    annotate(y_test, predicted)

    return scaler, lr, svc


def annotate(actual, predicted):
    print(confusion_mat(actual, predicted))

    print("there are %i total matches", len(actual))
    print("there are %i actual matches", actual.sum())
    print("predicted %i relevant matches", predicted.sum())


def run(train, test, tdidf=False):

    y_train = train["label"].copy()
    y_test = test["label"].copy()

    results = {}
    if tdidf:
        corpus = list(train["news"].unique())
        vectorizer = TfidfVectorizer()
        print("fitting tfidf")
        vectorizer.fit(corpus)
        results["vectorizer"] = vectorizer

        train["news"] = train["news"].apply(lambda x: remove_low_tfidf(vectorizer, x))
        test["news"] = test["news"].apply(lambda x: remove_low_tfidf(vectorizer, x))

    # Experiment 1: similarity cutoff
    logger.info("Experiment 1")
    labeled_similarity = exp_similarity_cutoff(train)
    predicted = (labeled_similarity["sim"] > 0.3).astype(int)
    annotate(y_train, predicted)

    # Experiment 2: Glove Vectors
    logger.info("Experiment 2")

    X_train = train["news"].apply(get_vectors_glove) - train["wiki"].apply(
        get_vectors_glove
    )
    X_train = pd.DataFrame(X_train.to_list())

    X_test = test["news"].apply(get_vectors_glove) - test["wiki"].apply(
        get_vectors_glove
    )
    X_test = pd.DataFrame(X_test.to_list())

    scaler_glove, lr_glove, svc_glove = exp_ml_prediction(
        X_train, X_test, y_train, y_test
    )

    # Experiment 3: Google Vectors
    logger.info("Experiment 3")
    X_train = google(train["news"]) - google(train["wiki"])
    X_test = google(test["news"]) - google(test["wiki"])

    scaler_google, lr_google, svc_google = exp_ml_prediction(
        X_train, X_test, y_train, y_test
    )

    results["glove_scaler"] = scaler_glove
    results["glove_lr"] = lr_glove
    results["glove_svc"] = svc_glove
    results["google_scaler"] = scaler_google
    results["google_lr"] = lr_google
    results["google_svc"] = svc_google
    return results


def save_model(model, addition=""):

    filepath = gen_filepath(model, addition=addition)
    with open(f"../artifacts/{filepath}.pkl", "wb") as f:
        pickle.dump(model, f)
    logger.info("model saved under filename %s", filepath)


def gen_filepath(model, addition=""):
    try:
        return f"{model.model_name_},{addition}"
    except AttributeError:
        return f"{type(model).__name__},{addition}"


def save_to_json(df):
    results = []
    for nid in df["news"].unique():
        sub_df = df.loc[df["news"] == nid]
        relevant = sub_df.loc[sub_df["predict"] == 1, ["title", "wiki_url"]].to_dict(
            "records"
        )
        irrelevant = sub_df.loc[sub_df["predict"] == 0, ["title", "wiki_url"]].to_dict(
            "records"
        )
        news = sub_df["news"].unique()[0]

        results.append({"news": news, "relevant": relevant, "irrelevant": irrelevant})

    with open("../artifacts/results.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":

    train = pd.read_csv("../data/labeled.csv")

    test1 = pd.read_csv("../data/wikinews_11-24-2021.csv")
    test2 = pd.read_csv("../data/wikinews_11-29-2021.csv")
    test = test1.append(test2)

    train = train.fillna("")
    test = test.fillna("")
    train = upsample(train)

    # models = run(train, test)
    models = run(train, test, tdidf=True)

    # pickle files
    for key, value in models.items():
        save_model(value, key)

    # save some test data to display as a sample in the flask app
    test3 = pd.read_csv("../data/wikinews_12-01-2021.csv")
    test3 = test3.fillna("")
    test3["cleaned_news"] = test3["news"].apply(
        lambda x: remove_low_tfidf(models["vectorizer"], x)
    )
    X_test = google(test3["cleaned_news"]) - google(test3["wiki"])
    X_test_scaled = models["google_scaler"].transform(X_test)

    test3["predict"] = models["google_svc"].predict(X_test_scaled)
    print(test3["predict"].value_counts())
    save_to_json(test3)

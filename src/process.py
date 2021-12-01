import string

from unidecode import unidecode
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

from nltk.corpus import stopwords

stop_words = stopwords.words("english")


def clean_text(text):
    text = unidecode(text)
    text = text.replace("-", " ")
    text = text.replace("'s", "")
    lower_text = text.translate(str.maketrans("", "", string.punctuation)).lower()
    cleaned_text = " ".join([w for w in lower_text.split(" ") if w not in stop_words])
    return cleaned_text


def plot_sim(vecs, fileinfo="plot"):
    """vecs is a list of paired vectors
    [
        [vec1, vec2],
        [vec3, vec4],
        ...
    ]
    """

    plt.figure(figsize=(15, 5))

    n = len(vecs)
    i = 1
    for pair in vecs:
        plt.subplot(1, n, i)
        plt.scatter(pair[0], pair[1])
        i += 1

        plt.xlabel("base")
        plt.ylabel("compare")

    plt.savefig(f"../images/{fileinfo}.png")


def tfidf_transform(vectorizer, text):
    vectors = vectorizer.transform(text)
    feature_names = vectorizer.get_feature_names_out()
    return pd.DataFrame(vectors.todense().tolist(), columns=feature_names)


def remove_low_tfidf(vectorizer, text):
    df = tfidf_transform(vectorizer, [text])
    ref = dict(df.iloc[0])

    scores = []
    words = np.unique(text.split(" "))
    for word in words:
        try:
            scores.append(ref[word])
        except KeyError:
            scores.append(0)
    least = (
        pd.DataFrame({"words": words, "scores": scores})
        .sort_values("scores", ascending=False)
        .tail()["words"]
        .values
    )
    return " ".join([word for word in text.split(" ") if word not in least])


if __name__ == "__main__":

    labeled = pd.read_csv("../data/labeled.csv")
    labeled = labeled.fillna("")

    corpus = list(labeled["news"].unique())
    vectorizer = TfidfVectorizer()
    print("fitting tfidf")
    vectorizer.fit(corpus)

    labeled["news"] = labeled["news"].apply(clean_text)
    print("applying tfidf")
    cleaned_news = labeled["news"].apply(lambda x: remove_low_tfidf(vectorizer, x))

    text = "Biden warns against Omicron panic, pledges no new lockdowns -  President Joe Biden urged Americans on Monday not to panic about the new COVID-19 Omicron variant and said the United States was making contingency plans with pharmaceutical companies if new vaccines are needed."
    print(text)
    print(remove_low_tfidf(vectorizer, text))

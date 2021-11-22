from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

from predict import confusion_mat


def tfidf_transform(vectorizer, text):
    vectors = vectorizer.transform(text)
    feature_names = vectorizer.get_feature_names_out()
    return pd.DataFrame(vectors.todense().tolist(), columns=feature_names)


def remove_low_tfidf(text):
    df = tfidf_transform(vectorizer, [text])
    ref = dict(df.iloc[0])

    scores = []
    words = np.unique(text.split(' '))
    for word in words:
        try:
            scores.append(ref[word])
        except KeyError:
            scores.append(0)
    least = pd.DataFrame({'words': words,
                          'scores': scores}).sort_values('scores', ascending=False).tail()['words'].values
    return ' '.join([word for word in text.split(' ') if word not in least])


def dfidf_run():

    news = list(labeled['news'].unique())
    clean_news = [clean_text(n) for n in news]

    print(confusion_mat(labeled))

    vectorizer = TfidfVectorizer()
    vectorizer.fit(news)

    print(clean_news[0])
    remove_low_tfidf(clean_news[0])

    labeled = pd.read_csv('../labeled.csv')
    labeled = labeled.fillna('')

    labeled['news'] = labeled['news'].apply(clean_text).apply(remove_low_tfidf)
    labeled['wiki'] = labeled['wiki'].apply(clean_text).apply(remove_low_tfidf)


if __name__ == '__main__':

    labeled = pd.read_csv('../labeled.csv')
    labeled = labeled.fillna('')

    # Define X and y
    X = labeled['news'].apply(get_vector) - labeled['wiki'].apply(get_vector)
    X = pd.DataFrame(X.to_list())

    y = labeled['label']

    # Experiment 1: Logistic Regression
    lr = LogisticRegression(random_state=0).fit(X, y)
    print(lr.score(X, y))

    labeled['predict'] = lr.predict(X)
    print(confusion_mat(labeled))

    # Experiment 2: SVC
    lr = SVC(random_state=0).fit(X, y)
    print(lr.score(X, y))

    labeled['predict'] = lr.predict(X)
    print(confusion_mat(labeled))

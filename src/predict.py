import logging

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

from process import clean_text
from embeddings import get_vectors_glove, counter_embedding, get_vectors_from_count, get_cosine_sim

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def confusion_mat(labeled):
    cm1 = confusion_matrix(labeled['label'], labeled['predict'])
    total1 = sum(sum(cm1))

    accuracy = (cm1[0, 0]+cm1[1, 1])/total1
    sensitivity = cm1[0, 0]/(cm1[0, 0] + cm1[0, 1])
    specificity = cm1[1, 1]/(cm1[1, 0] + cm1[1, 1])
    f1 = f1_score(labeled['label'], labeled['predict'])

    return {'accuracy': round(accuracy, 3),
            'sensitivity': round(sensitivity, 3),
            'specificity': round(specificity, 3),
            'f1': round(f1, 3)}


def cutoff_predict(data, cutoff):

    data['predict'] = data['sim'] > cutoff
    print("there are %i total matches", len(data))
    print("there are %i actual matches", data['label'].sum())
    print("predicted %i relevant matches", data['predict'].sum())

    return data


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


def tfidf_run():

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

    labeled = pd.read_csv('../data/labeled.csv')
    labeled = labeled.fillna('')

    # Experiment 1: similarity cutoff
    sim = []
    for i, row in labeled[['news', 'wiki']].iterrows():
        clean_news = clean_text(row['news'])
        clean_wiki = clean_text(row['wiki'])

        news_count = counter_embedding(clean_news)
        wiki_count = counter_embedding(clean_wiki)

        news_vec, wiki_vec = get_vectors_from_count(news_count, wiki_count)
        count_similarities = get_cosine_sim(news_vec, wiki_vec)
        sim.append(count_similarities)

    labeled['sim'] = sim

    predicted = cutoff_predict(labeled, 0.75)
    print(confusion_mat(predicted))

    # Define X and y
    # X is the distance between news and wiki text as vectors
    X = labeled['news'].apply(get_vectors_glove) - labeled['wiki'].apply(get_vectors_glove)
    X = pd.DataFrame(X.to_list())

    y = labeled['label']

    # Experiment 2: Logistic Regression
    lr = LogisticRegression(random_state=0).fit(X, y)

    labeled['predict'] = lr.predict(X)
    print(confusion_mat(labeled))

    # Experiment 3: SVC
    lr = SVC(random_state=0).fit(X, y)

    labeled['predict'] = lr.predict(X)
    print(confusion_mat(labeled))

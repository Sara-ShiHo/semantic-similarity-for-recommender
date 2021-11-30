import logging

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.utils import resample

from process import clean_text
from embeddings import get_vectors_glove, counter_embedding, get_vectors_from_count, get_cosine_sim, google

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

RANDOM_STATE = 414


def upsample(train_df):
    positive = train_df.loc[train_df['label'] == 1]
    negative = train_df.loc[train_df['label'] == 0]

    positive_upsampled = resample(positive, random_state=RANDOM_STATE,
                                  n_samples=len(negative) - len(positive))

    return pd.concat([positive, positive_upsampled, negative])


def confusion_mat(actual, predicted):
    cm1 = confusion_matrix(actual, predicted)
    total1 = sum(sum(cm1))

    accuracy = (cm1[0, 0]+cm1[1, 1])/total1
    sensitivity = cm1[0, 0]/(cm1[0, 0] + cm1[0, 1])
    specificity = cm1[1, 1]/(cm1[1, 0] + cm1[1, 1])
    f1 = f1_score(actual, predicted)

    return {'accuracy': round(accuracy, 3),
            'sensitivity': round(sensitivity, 3),
            'specificity': round(specificity, 3),
            'f1': round(f1, 3)}


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


def exp_similarity_cutoff(labeled):
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
    return labeled


def exp_ml_prediction(X_train, X_test, y_train, y_test):

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info('Logistic Regression')
    lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000).fit(X_train_scaled, y_train)
    predicted = lr.predict(X_train_scaled)
    logger.info('train predictions:')
    annotate(y_train, predicted)

    predicted = lr.predict(X_test_scaled)
    logger.info('test predictions:')
    annotate(y_test, predicted)

    logger.info('Support Vector Classifier')
    svc = SVC(random_state=RANDOM_STATE).fit(X_train_scaled, y_train)
    predicted = svc.predict(X_train_scaled)
    logger.info('train predictions:')
    annotate(y_train, predicted)

    predicted = svc.predict(X_test_scaled)
    logger.info('test predictions:')
    annotate(y_test, predicted)

    return lr, svc


def annotate(actual, predicted):
    print(confusion_mat(actual, predicted))

    print("there are %i total matches", len(actual))
    print("there are %i actual matches", actual.sum())
    print("predicted %i relevant matches", predicted.sum())


def run(train, test):
    train = train.fillna('')
    test = test.fillna('')
    train = upsample(train)

    y_train = train['label'].copy()
    y_test = test['label'].copy()

    # Experiment 1: similarity cutoff
    logger.info('Experiment 1')
    labeled_similarity = exp_similarity_cutoff(train)
    predicted = (labeled_similarity['sim'] > 0.3).astype(int)
    annotate(y_train, predicted)

    # Experiment 2: Glove Vectors
    logger.info('Experiment 2')

    X_train = train['news'].apply(get_vectors_glove) - train['wiki'].apply(get_vectors_glove)
    X_train = pd.DataFrame(X_train.to_list())

    X_test = test['news'].apply(get_vectors_glove) - test['wiki'].apply(get_vectors_glove)
    X_test = pd.DataFrame(X_test.to_list())

    lr, svc = exp_ml_prediction(X_train, X_test, y_train, y_test)

    # Experiment 3: Google Vectors
    logger.info('Experiment 3')
    X_train = google(train['news']) - google(train['wiki'])
    X_test = google(test['news']) - google(test['wiki'])

    lr, svc = exp_ml_prediction(X_train, X_test, y_train, y_test)



if __name__ == '__main__':

    labeled = pd.read_csv('../data/labeled.csv')

    test1 = pd.read_csv('../data/wikinews_11-24-2021.csv')
    test2 = pd.read_csv('../data/wikinews_11-29-2021.csv')
    test = test1.append(test2)

    run(labeled, test)

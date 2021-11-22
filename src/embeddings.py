import gensim.downloader as api
from scipy import spatial
import tensorflow_hub as hub
import re
import math
from collections import Counter
import pandas as pd
import numpy as np
import logging

from helpers import clean_text, plot_sim

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

logging.basicConfig(level=logging.INFO)


if 'glove' not in globals():
    logging.info('loading model')
    glove = api.load("glove-wiki-gigaword-50")

if 'google' not in globals():
    google = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Experiment 1: Frequency Counter


def counter_embedding(text):
    word = re.compile(r"\w+")
    words = word.findall(text)
    return Counter(words)


def get_cosine(x):

    vec1 = counter_embedding(x[0])
    vec2 = counter_embedding(x[1])

    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    return float(numerator) / denominator


def get_vectors_from_dict(x):
    vec1 = counter_embedding(x[0])
    vec2 = counter_embedding(x[1])

    df1 = pd.DataFrame.from_dict(vec1, orient='index')
    df2 = pd.DataFrame.from_dict(vec2, orient='index')

    joined = pd.merge(df1, df2, left_index=True,
                      right_index=True, how='outer').fillna(0)

    return joined['0_x'].values, joined['0_y'].values

# Experiment 2: Glove Embeddings

def glove_embedding(word):
    """choose from multiple models
    https://github.com/RaRe-Technologies/gensim-data
    """
    try:
        return glove[word]
    except KeyError:
        return np.zeros(50)


def get_vectors_glove(s):
    return np.sum(np.array([glove_embedding(i)
                            for i in s.split(' ')]), axis=0)


def get_cosine_sim(x):
    return 1 - spatial.distance.cosine(x[0], x[1])


if __name__ == '__main__':

    # Example of how code works:

    text1 = """Ohio State vs. Michigan State score, takeaways: Buckeyes obliterate Spartans, make strong playoff statement - Ohio State tore through Michigan State's defense on its way to an impactful win"""
    text2 = """The Ohio State Buckeyes football team competes as part of the NCAA Division I Football Bowl Subdivision, representing The Ohio State University in the East Division of the Big Ten Conference. Ohio State has played their home games at Ohio Stadium in Columbus, Ohio since 1922. The Buckeyes are recognized by the university and NCAA as having won eight national championships (including six wire-service: AP or Coaches) along with 41 conference championships (including 39 Big"""
    text3 = """Ohio ( (listen)) is a state in the Midwestern region of the United States. Of the fifty states, it is the 34th-largest by area, and with a population of nearly 11.8 million, is the seventh-most populous and tenth-most densely populated. The state's capital and largest city is Columbus, with the Columbus metro area, Greater Cincinnati, and Greater Cleveland being the largest metropolitan areas. Ohio is bordered by Lake Erie to the north, Pennsylvania to the"""
    text4 = """The Planetary Society is an American internationally-active non-governmental nonprofit organization. It is involved in research, public outreach, and political space advocacy for engineering projects related to astronomy, planetary science, and space exploration.  It was founded in 1980 by Carl Sagan, Bruce Murray, and Louis Friedman, and has about 60,000 members from more than 100 countries around the world.The Society is dedicated to the exploration of the Solar System, the search for near-Earth objects, and"""

    clean_texts = [clean_text(s) for s in [text1, text2, text3, text4]]

    # Experiment 2: Glove Embedding
    logging.info('Experiment 1')
    glove_vectors = [get_vectors_glove(text) for text in clean_texts]
    glove_similarities = [get_cosine_sim([glove_vectors[0], vec])
                          for vec in glove_vectors[1:]]
    print(glove_similarities)
    plot_sim(glove_vectors[0], glove_vectors[1:])

    # Experiment 3: Google Embedding
    logging.info('Experiment 2')
    google_vectors = google(clean_texts)

    # https://towardsdatascience.com/use-cases-of-googles-universal-sentence-encoder-in-production-dd5aaab4fc15#:%7E:text=The%20Universal%20Sentence%20Encoder%20encodes,and%20other%20natural%20language%20tasks.&text=It%20comes%20with%20two%20variations,Deep%20Averaging%20Network%20(DAN).
    google_similarities = np.inner(google_vectors[0], google_vectors[1:])
    print(google_similarities)
    plot_sim(google_vectors[0], google_vectors[1:])

import re
from collections import Counter
import logging

from scipy import spatial
import pandas as pd
import numpy as np
import tensorflow_hub as hub
import gensim.downloader as api

from process import clean_text, plot_sim

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if 'glove' not in globals():
    logging.info('loading model')
    glove = api.load("glove-wiki-gigaword-50")

if 'google' not in globals():
    google = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Experiment 1: Frequency Counter


def counter_embedding(text):
    """ return Counter object with frequency of words"""
    words = re.compile(r"\w+").findall(text)
    return Counter(words)


def get_vectors_from_count(vec1, vec2):
    """ align Counter objects by converting them to series, then merging """
    df1 = pd.DataFrame.from_dict(vec1, orient='index', columns=['vec1'])
    df2 = pd.DataFrame.from_dict(vec2, orient='index', columns=['vec2'])

    joined = pd.merge(df1, df2,
                      left_index=True, right_index=True,
                      how='outer').fillna(0)

    return joined['vec1'].values, joined['vec2'].values

# Experiment 2: Glove Embeddings


def glove_embedding(word):
    """choose from https://github.com/RaRe-Technologies/gensim-data """
    try:
        return glove[word]
    except KeyError:
        return np.zeros(50)


def get_vectors_glove(s):
    return np.sum(np.array([glove_embedding(i)
                            for i in s.split(' ')]), axis=0)


def get_cosine_sim(vec1, vec2):
    return 1 - spatial.distance.cosine(vec1, vec2)


if __name__ == '__main__':

    # Example of how code works

    # set up example data
    BASE = """Ohio State vs. Michigan State score, takeaways: Buckeyes obliterate Spartans, make strong playoff statement - Ohio State tore through Michigan State's defense on its way to an impactful win"""
    TEXT1 = """The Ohio State Buckeyes football team competes as part of the NCAA Division I Football Bowl Subdivision, representing The Ohio State University in the East Division of the Big Ten Conference. Ohio State has played their home games at Ohio Stadium in Columbus, Ohio since 1922. The Buckeyes are recognized by the university and NCAA as having won eight national championships (including six wire-service: AP or Coaches) along with 41 conference championships (including 39 Big"""
    TEXT2 = """Ohio ( (listen)) is a state in the Midwestern region of the United States. Of the fifty states, it is the 34th-largest by area, and with a population of nearly 11.8 million, is the seventh-most populous and tenth-most densely populated. The state's capital and largest city is Columbus, with the Columbus metro area, Greater Cincinnati, and Greater Cleveland being the largest metropolitan areas. Ohio is bordered by Lake Erie to the north, Pennsylvania to the"""
    TEXT3 = """The Planetary Society is an American internationally-active non-governmental nonprofit organization. It is involved in research, public outreach, and political space advocacy for engineering projects related to astronomy, planetary science, and space exploration.  It was founded in 1980 by Carl Sagan, Bruce Murray, and Louis Friedman, and has about 60,000 members from more than 100 countries around the world.The Society is dedicated to the exploration of the Solar System, the search for near-Earth objects, and"""

    clean_texts = [clean_text(s) for s in [BASE, TEXT1, TEXT2, TEXT3]]

    # Experiment 1: Obtain vectors based on word count
    logger.info('Experiment 1: Count frequency vectors')

    counts = [counter_embedding(text) for text in clean_texts]
    count_vector_pairs = [get_vectors_from_count(counts[0], count_dict)
                          for count_dict in counts[1:]]
    count_similarities = [get_cosine_sim(vectors[0], vectors[1])
                          for vectors in count_vector_pairs]
    print(count_similarities)
    print(np.shape(count_vector_pairs))
    plot_sim(count_vector_pairs, fileinfo='count_vectorizer_sim')

    # Experiment 2: Glove Embedding
    logger.info('Experiment 2: Glove embedding')
    glove_vectors = [get_vectors_glove(text) for text in clean_texts]
    glove_similarities = [get_cosine_sim(glove_vectors[0], vec)
                          for vec in glove_vectors[1:]]
    print(glove_similarities)
    plot_sim([[glove_vectors[0], vec] for vec in glove_vectors[1:]],
             fileinfo='glove_embedding_sim')

    # Experiment 3: Google Embedding
    logger.info('Experiment 3: Google Universal Sentence Encoder')
    google_vectors = google(clean_texts)

    google_similarities = np.inner(google_vectors[0], google_vectors[1:])
    print(google_similarities)
    plot_sim([[google_vectors[0], vec] for vec in google_vectors[1:]],
             fileinfo='google_encoding_sim')

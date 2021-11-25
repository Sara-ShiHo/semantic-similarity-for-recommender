from unidecode import unidecode
import string
from matplotlib import pyplot as plt
import numpy as np

from nltk.corpus import stopwords
stop_words = stopwords.words('english')


def clean_text(text):
    text = unidecode(text)
    text = text.replace('-', ' ')
    text = text.replace("'s", '')
    lower_text = text.translate(str.maketrans(
        '', '', string.punctuation)).lower()
    clean_text = ' '.join(
        [w for w in lower_text.split(' ') if w not in stop_words])
    return clean_text


def plot_sim(vecs, fileinfo='plot'):
    """ vecs is a list of paired vectors
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

        # format plot
        plt.xlabel('base')
        plt.ylabel('compare')

    plt.savefig(f'../images/{fileinfo}.png')

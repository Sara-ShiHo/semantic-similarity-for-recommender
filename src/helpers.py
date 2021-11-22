from unidecode import unidecode
import string
from matplotlib import pyplot as plt

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


def plot_sim(base_vec, compare_vecs):
    plt.figure(figsize=(15, 5))

    max_ax = max([max(v) for v in compare_vecs] + [max(base_vec)])
    min_ax = min([min(v) for v in compare_vecs] + [min(base_vec)])

    n = len(compare_vecs)
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.scatter(base_vec, compare_vecs[i])

        # format plot
        plt.xlabel('base')
        plt.ylabel('compare')
        plt.xlim(min_ax - abs(min_ax) * 0.1, max_ax + max_ax * 0.1)
        plt.ylim(min_ax - abs(min_ax) * 0.1, max_ax + max_ax * 0.1)
    plt.show()

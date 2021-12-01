import pandas as pd
from statistics import median
from process import clean_text

RANDOM_STATE = 414


def describe_text_list(text_list):
    print(
        f"size: {len(text_list)}",
        f"median: {median(text_list)}",
        f"minimum: {min(text_list)}",
        f"maximum: {max(text_list)}",
    )


def summarize(df):
    n_relevant = df["label"].sum()
    print(f"Number of total observations: {len(df)}")
    print(f"Number of positives (relevant): {n_relevant}")
    print(f"Number of negatives (irrelevant): {len(df) - n_relevant}")

    news = df["news"].unique()
    wiki = df["wiki"].unique()
    print(f"number of news: {len(news)}")
    print(f"number of wikis: {len(wiki)}")

    wikis_per_news = df.groupby("news")["wiki"].count()
    print("Wiki suggestions per news")
    describe_text_list(wikis_per_news)

    n_words_clean = [len(clean_text(text).split(" ")) for text in news]
    print("Number of words per news")
    describe_text_list(n_words_clean)

    n_words_clean = [len(clean_text(text).split(" ")) for text in wiki]
    print("Number of words per wiki")
    describe_text_list(n_words_clean)


if __name__ == "__main__":

    TEXT = """New Swedish PM resigns on first day in job, hopes for swift return -  Sweden's first female prime minister, Social Democrat Magdalena Andersson, resigned on Wednesday after less than 12 hours in the top job after the Green Party quit their two-party coalition, stoking political uncertainty."""
    cleaned_text = clean_text(TEXT)
    print(cleaned_text)

    labeled = pd.read_csv("../data/labeled.csv")
    labeled = labeled.fillna("")

    summarize(labeled)

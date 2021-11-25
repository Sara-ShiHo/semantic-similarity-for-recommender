import pandas as pd
from process import clean_text


def describe_text_list(text_list):
    print(sum(text_list)/len(text_list),
          min(text_list),
          max(text_list))

if __name__ == '__main__':
    TEXT = """New Swedish PM resigns on first day in job, hopes for swift return -  Sweden's first female prime minister, Social Democrat Magdalena Andersson, resigned on Wednesday after less than 12 hours in the top job after the Green Party quit their two-party coalition, stoking political uncertainty."""
    cleaned_text = clean_text(TEXT)
    print(cleaned_text)

    labeled = pd.read_csv('../data/labeled.csv')
    labeled = labeled.fillna('')

    news = labeled['news'].unique()
    wiki = labeled['wiki'].unique()

    wikis_per_news = labeled.groupby('news_id')['wiki'].count()
    print(wikis_per_news.mean(),
          wikis_per_news.min(),
          wikis_per_news.max())

    n_words = [len(text.split(' ')) for text in news]
    n_words_clean = [len(clean_text(text).split(' ')) for text in news]
    describe_text_list(n_words)

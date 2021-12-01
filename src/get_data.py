from datetime import date

import requests
import spacy
import pandas as pd

from config import NEWS_API_KEY


def load_wiki(
    news_table,
    stop_spacy=["PERSON", "FAC", "ORG", "NORP", "PRODUCT"],
    spacy_model="en_core_web_sm",
    n_results=3,
):

    results = []
    for _, row in news_table[["news_id", "news"]].iterrows():
        news_id, news = row

        print("----Processing ", news[0:25])

        entities = news2entities(news, stop_spacy, spacy_model)
        wiki_obs = entities2wiki(entities, n_results)
        wiki_obs["news_id"] = news_id
        results.append(wiki_obs)

    return pd.concat(results)


def load_news(source_words=[]):

    data = news_top()

    all_headlines = []
    all_news = []
    all_imgs = []
    all_urls = []

    for article in data["articles"]:
        all_headlines.append(article["title"])
        all_imgs.append(article["urlToImage"])
        all_urls.append(article["url"])

        # parse news headline and description
        news = article["title"]
        if article["description"] is not None:
            news += " " + article["description"]

        # remove words pertaining to publications
        source_words.append(article["source"]["name"])
        news = remove_stopwords(news, source_words)
        all_news.append(news)

    news_table = pd.DataFrame(
        {
            "news_id": [*range(len(all_news))],
            "headline": all_headlines,
            "news": all_news,
            "news_image": all_imgs,
            "news_url": all_urls,
        }
    )
    return news_table


def remove_stopwords(text, stopwords):
    for word in stopwords:
        text = text.replace(word, "")
    return text


def news_top():
    """query top news; return JSON data"""

    session = requests.Session()
    url = "https://newsapi.org/v2/top-headlines?"
    params = {"country": "US", "apiKey": NEWS_API_KEY}

    resp = session.get(url=url, params=params)
    session.close()

    if resp.json()["status"] == "error":
        print("API error: %s", resp.json()["message"])
        raise Exception("API error")
    else:
        return resp.json()


def news2entities(news, stop_spacy, spacy_model):
    nlp = spacy.load(spacy_model)
    doc = nlp(news)

    entities = []
    for ent in doc.ents:
        if (ent.label_ in stop_spacy) and (ent.text not in entities):
            text = ent.text
            if ent.label_ == "ORG":
                text += " (organization)"
            entities.append(text)
    return entities


def entities2wiki(entities, n_results=1):
    all_data = []
    all_titles = []
    for ent in entities:
        articledata = wiki_query(ent)
        search_results = articledata["query"]["search"]

        # n_results is how many search results to consider matching
        for result in search_results[0:n_results]:

            title = result["title"]
            if title in all_titles:
                continue  # bypass the rest of the for loop

            info = wiki_content(title)

            categories = info["categories"]
            categories = [kv["title"].lower() for kv in categories]

            wiki = info["extract"]
            wiki = wiki_special_truncate(wiki).split("\n")[0]
            wiki = " ".join(wiki.split(" ")[0:75])
            image = wiki_image(info)

            all_titles.append(title)
            all_data.append(
                {
                    "entity": ent,
                    "title": info["title"],
                    "wiki": wiki,
                    "wiki_image": image,
                }
            )

    return pd.DataFrame(all_data)


def wiki_special_truncate(text):
    if "==" in text:
        end = text.find("==")
        return text[0:end]
    return text


def wiki_image(data):
    try:
        return data["thumbnail"]["source"]
    except KeyError:
        return ""


def wiki_query(query):

    session = requests.Session()
    url = "https://en.wikipedia.org/w/api.php"
    params = {"action": "query", "format": "json", "list": "search"}

    params["srsearch"] = query

    return session.get(url=url, params=params).json()


def wiki_content(title):

    session = requests.Session()
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts|pageimages|info|categories",
        "inprop": url,
        "exsentences": 10,
        "explaintext": 1,
        "pithumbsize": 100,
    }
    params["titles"] = title  # add title to the parameters, required by API

    try:
        resp = session.get(url=url, params=params)
        return list(resp.json()["query"]["pages"].values())[0]

    except requests.RequestException as exc:
        print("General Error: %s", exc)
        return None


if __name__ == "__main__":
    news_data = load_news()
    wiki_data = load_wiki(news_data)

    joined = pd.merge(news_data, wiki_data)
    joined = joined.loc[~joined["title"].str.startswith("List of")]

    today_date = date.today().strftime("%m-%d-%Y")

    joined.to_csv(f"../data/wikinews_{today_date}.csv", index=False)

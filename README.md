# Semantic Similarity Methods for Content Recommendation


## Directory structure 

To load new data, put constant variable `NEWS_API_KEY` into `src/config.py`. 
No API key is needed to run experiments or the app. 

```
├── README.md             <- You are here
├── api.py                <- Flask app which exposes best model
├── src
│   ├── embeddings.py     <- Run embedding models
│   ├── get_data.py       <- Run part I of pipeline, loading and matching new data
│   ├── predict.py        <- Run experiments for paper results
│   ├── process.py        <- Helper functions to process data
│   ├── summary.py        <- Run data summarization for informational purposes
├── app                   <- static and templates for web app
├── artifacts             <- placeholder folder for experiment pickled output
├── images                <- placeholder folder for experiment output
├── data                  <- placeholder folder for experiment input and output
```
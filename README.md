# MINDALAE

NLP-based tourism recommendation system. Uses semantic embeddings (SBERT) to surface places matching natural language queries, drawn from real-world reviews in the Yelp Open Dataset.

## Project status

Under development. Academic project — first semester, AI Engineering, Universidad de Caldas.

## Architecture

The system is composed of four modules:

1. **`preparar_datos.py`** — Offline setup. Transforms raw Yelp data into vectorized artifacts ready for runtime consumption.
2. **`buscar.py`** — Recommender core. Receives a user query, vectorizes it, and returns the most relevant reviews via dot product similarity.
3. **`narrar.py`** — Contextual narrative generation over the recommendations using the OpenAI API.
4. **`app.py`** — Streamlit user interface.

## Stack

- Python 3.13
- pandas, numpy
- sentence-transformers (`all-MiniLM-L6-v2` model)
- Streamlit (pending)
- OpenAI API (pending)

## Dataset

Yelp Open Dataset — `business.json` and `review.json`. Raw files are not versioned (see `.gitignore`); they must be downloaded manually from [yelp.com/dataset](https://www.yelp.com/dataset) and placed under `data/raw/`.

## Structure
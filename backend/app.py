import logging
import re
import sqlite3
import time
from collections import Counter
from contextlib import closing
from typing import List, Tuple

import nltk
import pandas as pd
from fastapi import FastAPI, HTTPException
from nltk.corpus import stopwords
from nltk.util import ngrams
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Setup for nltk
nltk.download("stopwords", quiet=True)
stop_words = stopwords.words("english")

# Logging configuration
logging.basicConfig(level=logging.DEBUG)

# Database Path: Wikibooks (https://dumps.wikimedia.org/enwikibooks/latest/)
DB_PATH = "data/wikibooks.sqlite"

# CSV File Path: British Airways Reviews
CSV_PATH = "data/reviews.csv"


class PredictionRequest(BaseModel):
    word: str
    data_source: str  # "csv" or "db"


class PredictionResponse(BaseModel):
    prediction: str
    top_n_predictions: List[Tuple[str, float]]


def timer(func):
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        logging.info(
            f"{func.__name__} -> Time taken: {time.time() - start:.2f} seconds"
        )
        return result

    return wrapper


async def read_db():
    logging.info("Reading database")
    with closing(sqlite3.connect(DB_PATH)) as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT body_text FROM en")
            rows = [row[0] for row in cursor.fetchall()]
    logging.info(f"Finished reading database, total rows: {len(rows)} rows")
    return rows


async def read_csv():
    logging.info("Reading csv file")
    df = pd.read_csv(CSV_PATH)
    rows = df["ReviewBody"].tolist()
    logging.info(f"Finished reading csv file, total rows: {len(rows)} rows")
    return rows


async def preprocess(rows: List[str]):
    logging.info("Preprocessing data")
    return [
        " ".join(
            [
                word
                for word in re.sub(r"[^a-zA-Z0-9\s]", "", row).lower().split()
                if word not in stop_words
            ]
        )
        for row in rows
        if isinstance(row, str)
    ]


async def tokenize(rows: List[str]):
    logging.info("Tokenizing data")
    return [row.split() for row in rows]


async def get_ngrams(tokens: List[List[str]], n: int):
    logging.info("Creating n-grams")
    ngram_counts = Counter()
    for token_list in tokens:
        ngram_counts.update(ngrams(token_list, n, pad_right=True, pad_left=True))
    logging.info(f"Total n-grams: {len(ngram_counts)}")
    return ngram_counts


async def predict(
    ngram_counts: Counter, word: str, n: int = 5
) -> Tuple[str, List[Tuple[str, float]]]:
    logging.info("Predicting next word")
    predictions = {
        ng[1]: count
        for ng, count in ngram_counts.items()
        if ng[0] == word and ng[1] != None
    }
    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[
        :n
    ]
    total = sum(predictions.values())
    return (
        sorted_predictions[0][0] if sorted_predictions else "",
        [(word, count / total) for word, count in sorted_predictions],
    )


@app.post("/predict", response_model=PredictionResponse)
async def make_prediction(request: PredictionRequest):
    try:
        rows = await read_csv() if request.data_source == "csv" else await read_db()
        processed_rows = await preprocess(rows)
        tokens = await tokenize(processed_rows)
        ngram_counts = await get_ngrams(tokens, 2)

        prediction, top_n = await predict(ngram_counts, request.word, n=5)

        return PredictionResponse(prediction=prediction, top_n_predictions=top_n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)

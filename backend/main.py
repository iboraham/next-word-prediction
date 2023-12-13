import logging
import re
import sqlite3
import time
from collections import Counter

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.util import ngrams

# Database and nltk setup
conn = sqlite3.connect("data/wikibooks.sqlite")
nltk.download("stopwords", quiet=True)
stop_words = stopwords.words("english")

logging.basicConfig(level=logging.DEBUG)


# Decorator for timing functions
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logging.info(
            f"{func.__name__} -> Time taken: {time.time() - start:.2f} seconds"
        )
        return result

    return wrapper


@timer
def read_db(conn: sqlite3.connect):
    logging.info("Reading database")
    c = conn.cursor()
    c.execute("SELECT body_text FROM en")
    rows = [row[0] for row in c.fetchall()]
    logging.info(f"Finished reading database, total rows: {len(rows)} rows")
    return rows


@timer
def read_csv(fp):
    logging.info("Reading csv file")
    df = pd.read_csv(fp)
    rows = df["ReviewBody"].tolist()
    logging.info(f"Finished reading csv file, total rows: {len(rows)} rows")
    return rows


@timer
def preprocess(rows):
    logging.info("Preprocessing data")
    processed_rows = []
    for row in rows:
        if isinstance(row, str):
            row = re.sub(r"[^a-zA-Z0-9\s]", "", row).lower()
            row = " ".join([word for word in row.split() if word not in stop_words])
            processed_rows.append(row)
    return processed_rows


@timer
def tokenize(rows):
    logging.info("Tokenizing data")
    return [row.split() for row in rows]


@timer
def get_ngrams(tokens, n):
    logging.info("Creating n-grams")
    ngram_counts = Counter()
    for token_list in tokens:
        for ng in ngrams(token_list, n, pad_right=True, pad_left=True):
            ngram_counts[ng] += 1
    logging.info(f"Total n-grams: {len(ngram_counts)}")
    return ngram_counts


@timer
def predict(ngram_counts, word):
    logging.info("Predicting next word")
    predictions = {}
    for ng in ngram_counts:
        if ng[0] == word:
            predictions[ng[1]] = ngram_counts[ng]

    return max(predictions, key=predictions.get) if predictions else "", predictions


def main():
    rows = read_csv("data/reviews.csv")  # or read_db(conn)
    processed_rows = preprocess(rows)
    tokens = tokenize(processed_rows)
    ngram_counts = get_ngrams(tokens, 2)

    prediction, all_matches = predict(ngram_counts, "english")
    # Convert counts to percentages
    total = sum(all_matches.values())
    all_matches = {k: v / total for k, v in all_matches.items()}
    all_matches = Counter(all_matches)
    logging.info(f"Top 10 matches: {all_matches.most_common(10)}")
    logging.info(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()
    conn.close()

import os

import pandas as pd
from tqdm import tqdm
from transformers import pipeline

from utils import string_to_month_year

def get_sentiments(
        date,
        tweets,
        sentiment_analyser=None,
        save_every=2000,
        percentage_per_chunk=0.05,
        chunk=0):
    # TODO Implement chunk to choose i load range to download
    # TODO Check reduction used first time and match

    if sentiment_analyser is None:
        sentiment_analyser = pipeline("sentiment-analysis")

    scaled_tweets = scale_tweet_list(percentage_per_chunk, save_every, tweets)
    results = []

    i = 0
    for tweet in tqdm(scaled_tweets, desc=date):

        result = sentiment_analyser(tweet)
        results.append(result)

        i += 1
        if i % save_every == 0:
            save_sentiments(results, date)
        if i > (length * percentage_per_chunk):
            break
    return results


def scale_tweet_list(percentage_per_chunk, save_every, tweets):
    length = len(tweets)
    tweet_cap = length / percentage_per_chunk
    last_tweet = tweet_cap - (tweet_cap % save_every)
    scaled_tweets = tweets[:last_tweet]
    return scaled_tweets


def get_paths(results_folder = None, source = None, reset=False) -> (str, str):
    if source is None:
        source = '../data/bitcoin_tweets/'
    if results_folder is None:
        results_folder = '../data/bitcoin_scores/'

    if not os.path.exists(results_folder):
            os.mkdir(results_folder)

    if reset:
    return source, results_folder

def to_dict_of_lists(LD):
    nd = {}
    for d in LD:
        for k, v in d[0].items():
            try:
                nd[k].append(v)
            except KeyError:
                nd[k] = [v]
    return nd


def get_tweets(source, date):


    month = string_to_month_year(date)
    path = source + month + "/MTurk_" + date + ".csv"

    with open(path) as f:
        df = pd.read_csv(f)

    return df["id"].values.tolist(), df["tweet"].values.tolist()


def save_sentiments(results, date):
    df = pd.DataFrame(results)

    results_path = results_folder + date + ".csv"

    df.to_csv(results_path, mode="a", header=False, index=False)
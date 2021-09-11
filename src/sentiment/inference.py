import os

import pandas as pd
from tqdm import tqdm
from transformers import pipeline

from utils import string_to_month_year

# This attempt at a class is so wrong
# class Tweet_Inference(
#     results_folder: str,
#     source: str
#     # ToDo Parmaterise for a task
#     # task: str,
#     # model: Optional = None,
#     # config: Optional[Union[str, PretrainedConfig]] = None,
#     # tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
#     # feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None,
#     # framework: Optional[str] = None,
#     # revision: Optional[str] = None,
#     # use_fast: bool = True,
#     # use_auth_token: Optional[Union[str, bool]] = None,
#     # model_kwargs: Dict[str, Any] = {},
#     # **kwargs
# )

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
    length = len(tweets)

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


def get_paths(reset=False, results_folder=None, source=None, data_folder="../data") -> (str, str):
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)

    if source is None:
        source = '../data/bitcoin_tweets'
    if results_folder is None:
        results_folder = '../data/bitcoin_scores/'

    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    # if reset:
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
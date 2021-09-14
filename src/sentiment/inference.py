import os

import pandas as pd
from tqdm import tqdm
from transformers import pipeline

from utils import string_to_month_year, to_dict_of_lists


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
        ids,
        results_folder,
        sentiment_analysis=None,
        save_every=2000,
        percentage_per_chunk=5,
        chunk=0):

    """"""
    # TODO Implement chunk to choose i load range to download
    # TODO Check reduction used first time and match

    if sentiment_analysis is None:
        sentiment_analysis = pipeline("sentiment-analysis")

    scaled_tweets, scaled_length = scale_tweet_list(percentage_per_chunk, save_every, tweets)

    i, saves = 0, 0
    results = []
    for tweet in tqdm(scaled_tweets, desc=date):

        try:
            result = sentiment_analysis(tweet)
            results.append(result)

        except ValueError as bug:
            track_bug(results_folder, date, tweet, bug)

        i += 1
        # if i % save_every == 0:
        #     saves += 1
        #     save_legnth = saves * save_every
        #     save_sentiments(ids[:save_legnth], results, results_folder, date)
        if i >= scaled_length:
            save_sentiments(ids[:scaled_length], results, results_folder, date)
            break



def save_sentiments(ids, results, results_folder, date):

    outputs = to_dict_of_lists(results)
    outputs["ids"] = ids

    df = pd.DataFrame(outputs)

    date_csv = date + ".csv"
    results_path = os.path.join(results_folder, date_csv)
    df.to_csv(results_path, mode="a", header=False, index=False)


def scale_tweet_list(percentage_per_chunk, save_every, tweets):
    # scales to minimum of the save_every time size

    length_of_tweets = len(tweets)
    percent_of_length = int(length_of_tweets * percentage_per_chunk/100)
    # percent_of_length = int(percent_of_length)
    last_tweet = percent_of_length - (percent_of_length % save_every)
    # last_tweet = int((length_of_tweets / percentage_per_chunk) - (percent_of_length % save_every))

    if last_tweet == 0:
        last_tweet = save_every

    scaled_tweets = tweets[:last_tweet]
    return scaled_tweets, last_tweet


def get_paths(reset=False,
              crypto='bitcoin',
              data_folder='../data',
              model_name=None):
    # ToDo move model name to start and make not kwarg
    # if reset:
    #  ToDo Remove old data

    if model_name:
        try:
            model_developer, model_name, = model_name.split('/', 1)
        except:
            pass

        model_folder = '-'.join(model_name.split('-')[:4])
        try:
            model_folder = f"{model_folder}_{model_developer}"
        except:
                pass
    else:
        model_folder = "distilbert-base-uncased-finetuned"


    raw_results_folder = f'{crypto}_scores/'
    raw_source_folder = f'{crypto}_tweets/'

    source_folder = os.path.join(data_folder, raw_source_folder)
    results_folder = os.path.join(data_folder, raw_results_folder, model_folder)

    # if not os.path.exists(results_folder):
    #     throw

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    return source_folder, results_folder


def get_tweets(date, source='../data/bitcoin_tweets/'):

    month = string_to_month_year(date)
    path = source + month + "/MTurk_" + date + ".csv"

    with open(path) as f:
        df = pd.read_csv(f)

    return df["id"].values.tolist(), df["tweet"].values.tolist()


def track_bug(results_folder, date, tweet, bug):
    log_folder = os.path.join(results_folder, "bug")
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)

    log_path = os.path.join(log_folder, f"{date}.log")

    with open(log_path, 'a') as f:
        f.write(f"{date}, {tweet}, {bug} \n")

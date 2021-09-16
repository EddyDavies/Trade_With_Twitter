import os

import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from torch.utils.data import Dataset, DataLoader

from utils.dates import string_to_month_year
from utils.lists_dicts import to_dict_of_lists


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
class SimpleDataset(Dataset):
    def __init__(self, texts, slice_size=None):
        if slice_size:
            self.texts = [text[:125] for text in texts]
        else:
            self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def get_sentiments(
        date,
        tweets,
        ids,
        results_folder,
        sentiment_analysis=None,
        save_every=2000,
        percentage_per_chunk=5,
        slice_size=None,
        batch_size=100,
        chunk=0):

    """"""
    # TODO Implement chunk to choose i load range to download
    # TODO Check reduction used first time and match

    if sentiment_analysis is None:
        sentiment_analysis = pipeline("sentiment-analysis")

    if percentage_per_chunk == 100:
        scaled_tweets, scaled_ids = tweets, ids
    else:
        scaled_tweets, scaled_ids = scale_tweet_list(percentage_per_chunk, save_every, tweets, ids)

    # i, saves = 0, 0
    full_results = []

    pred_dataset = SimpleDataset(scaled_tweets, slice_size)
    dataloader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    for tweets in tqdm(dataloader, desc=date):

        # try:
        results = sentiment_analysis(tweets)
        full_results += results

        # except ValueError as bug:
        #     track_bug(results_folder, date, tweet, bug)

        # i += 1
        # if i % save_every == 0:
        #     saves += 1
        #     save_legnth = saves * save_every
        #     save_sentiments(ids[:save_legnth], results, results_folder, date)

    save_sentiments(scaled_ids, full_results, results_folder, date)


def save_sentiments(ids, results, results_folder, date):

    outputs = to_dict_of_lists(results)
    outputs["ids"] = ids

    df = pd.DataFrame(outputs)

    date_csv = date + ".csv"
    results_folder = os.path.normpath(results_folder)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    results_path = os.path.join(results_folder, date_csv)
    df.to_csv(results_path, mode="a", header=False, index=False)


def scale_tweet_list(percentage_per_chunk, save_every, tweets, ids):
    # scales to minimum of the save_every time size

    length_of_tweets = len(tweets)
    percent_of_length = int(length_of_tweets * percentage_per_chunk/100)
    # percent_of_length = int(percent_of_length)
    last_tweet = percent_of_length - (percent_of_length % save_every)
    # last_tweet = int((length_of_tweets / percentage_per_chunk) - (percent_of_length % save_every))

    if last_tweet == 0:
        last_tweet = save_every

    scaled_tweets = tweets[:last_tweet]
    scaled_ids = ids[:last_tweet]
    return scaled_tweets, scaled_ids


def get_tweets(date, source):

    month = string_to_month_year(date)
    file_name = "MTurk_" + date + ".csv"
    path = os.path.join(source, month, file_name)

    # with open(path) as f:
    path = os.path.normpath(path)
    print(path)

    df = pd.read_csv(path)

    return df["id"].values.tolist(), df["tweet"].values.tolist()


def track_bug(results_folder, date, tweet, bug):
    log_folder = os.path.join(results_folder, "bug")
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)

    log_path = os.path.join(log_folder, f"{date}.log")

    with open(log_path, 'a') as f:
        f.write(f"{date}, {tweet}, {bug} \n")

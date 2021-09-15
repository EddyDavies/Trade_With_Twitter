import os

import pandas as pd

from sentiment.inference import get_paths, convert_model_name
from utils import get_date_array

def get_tweets(date, scores):

    path = os.path.join(scores, f"{date}.csv")

    columns = ['score', 'label', 'id']
    with open(path) as f:
        df = pd.read_csv(f, names=columns)

    # return df["id"].values.tolist(), df["tweet"].values.tolist()
    return df


def extract_percents(df):
    global pos_percent, neu_percent
    length = float(df.shape[0])

    pos = df.loc[(df['label'] == 'POS')]
    pos_count = float(pos.shape[0])
    pos_weight = float(pos['score'].mean())
    neu = df.loc[(df['label'] == 'NEU')]
    neu_count = float(neu.shape[0])
    neu_weight = float(neu['score'].mean())

    pos_percent = (pos_count * pos_weight / length) * 100
    neu_percent = (neu_count * neu_weight / length) * 100

    return pos_percent, neu_percent


def meta_data_weight(df, weight=None):
    if not weight:
        return df
    else:
        pass

if __name__ == '__main__':

    dates_range = ("2017-01-01", "2021-05-31")
    dates_range = ("2017-01-01", "2017-02-28")
    dates = get_date_array(dates_range)

    data_folder = '../data'
    model_name = "finiteautomata/bertweet-base-sentiment-analysis"
    _, scores_folder = get_paths(data_folder=data_folder, model_name=model_name)

    for date in dates:
        df = get_tweets(date, scores_folder)

        df = meta_data_weight(df)

        pos_percent, neu_percent = extract_percents(df)

        model_csv = f"{convert_model_name(model_name)}.csv"
        result_path = os.path.join(data_folder, model_csv)
        with open(result_path, 'a') as f:
            f.write(f"{date}, {pos_percent}, {neu_percent} \n")



import os
from datetime import datetime

import pandas as pd

from sentiment.folder import convert_model_name

def convert_to_date(date, format='%Y-%m-%d'):
    return datetime.strptime(date, format)


def load_date_index(folder, names=None):
    if names:
        df = pd.read_csv(folder, names=names)
    else:
        df = pd.read_csv(folder)
    df['date'] = df["date"].apply(convert_to_date)

    pd.to_datetime(df['date'])
    df.set_index("date", inplace=True)
    return df


def select_data_type(style, crypto, data_folder, model_names):

    model_folders = [convert_model_name(model) for model in model_names]

    prices_folder = os.path.join(data_folder, "market-data", f"{crypto}.csv")

    df = load_date_index(prices_folder)
    df_ta, df_sa = None, None

    if 'ta' in style:
        ta_folder = os.path.join(data_folder, "market-data", f"{crypto}_ta.csv")
        df_ta = load_date_index(ta_folder)

    data = []
    if 'sa' in style:
        for i, model in enumerate(model_folders):
            if str(i+1) in style:
                sa_folder = os.path.join(data_folder, f"{model_folders[i]}.csv")
                loaded = load_date_index(sa_folder)
                data.append(loaded)
        if data is not []:
            df_sa = pd.concat(data, axis=1)

    df = pd.concat([df, df_sa, df_ta], axis=1)

    begin = pd.Timestamp('2017-01-01 00:00:00')
    end = pd.Timestamp('2021-05-31 00:00:00')

    return df.truncate(before=begin, after=end)


if __name__ == '__main__':

    crypto = 'bitcoin'
    model_names = ["finiteautomata/bertweet-base-sentiment-analysis",
                  "siebert/sentiment-roberta-large-english"]

    data_folder = '../data'
    merge_folder = os.path.join(data_folder, "trade")
    if not os.path.exists(merge_folder):
        os.mkdir(merge_folder)

    styles = ['ta_sa_12', 'ta_sa_2', 'ta_sa_1', 'sa_12', 'sa_2', 'sa_1', 'ta', 'p']
    for style in styles:
        df = select_data_type(style, crypto, data_folder, model_names)
        for column in df.columns:
            df[column] = df[column] / df[column].abs().max()

        merge_path = os.path.join(merge_folder, f"{crypto}_{style}.csv")
        df.to_csv(merge_path)


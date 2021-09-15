import os
import pandas as pd

from sentiment.folder import convert_model_name


def select_data_type(style, crypto, data_folder, model_folder):

    prices_folder = os.path.join(data_folder, "market-data", f"{crypto}.csv")
    ta_folder = os.path.join(data_folder, "market-data", f"{crypto}_ta.csv")
    sa_folder = os.path.join(data_folder, f"{model_folder}.csv")
    sa_metrics_folder = os.path.join(data_folder, f"{model_folder}_metrics.csv")

    df_prices = pd.read_csv(prices_folder)

    if 'ta' in style:
        df_ta = pd.read_csv(ta_folder)
        df_ta.set_index("date", inplace=True)
        # df_ta.set_index(pd.DatetimeIndex(df_ta["date"]), inplace=True)
        df = pd.concat([df_prices, df_ta], axis=1)
    if 'sa' in style:
        if 'metrics' in style:
            df_sa = pd.read_csv(sa_metrics_folder, names=['date', 'pos', 'neu'])
        else:
            df_sa = pd.read_csv(sa_folder, names=['date', 'pos', 'neu'])
        # df_sa.set_index("date", inplace=True)
        df = pd.concat([df_prices, df_sa], axis=1)

    if 'ta' in style and 'sa' in style:
        df = pd.concat([df_prices, df_sa, df_ta], axis=1)

    if 'p' in style:
        df = df_prices

    df.set_index("date", inplace=True)

    return df


if __name__ == '__main__':

    crypto = 'bitcoin'
    model_name = "finiteautomata/bertweet-base-sentiment-analysis"
    model_folder = convert_model_name("finiteautomata/bertweet-base-sentiment-analysis")

    data_folder = '../data'
    merge_folder = os.path.join(data_folder, "trade")
    if not os.path.exists(merge_folder):
        os.mkdir(merge_folder)

    styles = ['ta_sa', 'ta', 'sa', 'p']
    for style in styles:
        df = select_data_type(style, crypto, data_folder, model_folder)
        for column in df.columns:
            df[column] = df[column] / df[column].abs().max()

        merge_path = os.path.join(merge_folder, f"{crypto}_{style}.csv")
        df.to_csv(merge_path)


from transformers import pipeline

from sentiment.inference import get_tweets, get_sentiments
from sentiment.folder import get_paths
from utils.dates import get_date_array, get_month_array

if __name__ == '__main__':

    dates_range = ("2017-01-01", "2021-05-31")
    months_range = ["Jan 17", "Jun 21"]

    months = get_month_array(months_range)
    dates = get_date_array(dates_range)

    sentiment_analysis = pipeline("sentiment-analysis")

    results_folder, source = get_paths(reset=True)

    for date in dates:
        ids, tweets = get_tweets(date)

        percentage_per_chunk = 1
        save_every = 200
        get_sentiments(date, tweets, ids, results_folder,
                         sentiment_analysis=sentiment_analysis,
                         save_every=save_every,
                         percentage_per_chunk=percentage_per_chunk,
                         slice_size=None,
                         batch_size=100)




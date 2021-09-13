from transformers import pipeline

from sentiment.inference import get_paths, get_tweets, get_sentiments, save_sentiments
from utils import get_date_array, get_month_array, string_to_month_year, last_day_in_month, check_last_day

if __name__ == '__main__':
    results_folder, source = get_paths(reset=True)

    dates_range = ("2017-01-01", "2021-06-20")
    months_range = ["Jan 17", "Jun 21"]

    months = get_month_array(months_range)
    dates = get_date_array(dates_range)

    sentiment_analysis = pipeline("sentiment-analysis")

    for date in dates:
        ids, tweets = get_tweets(date)

        percentage_per_chunk = 1
        save_every = 50
        results = get_sentiments(date, tweets, ids, results_folder,
                                 sentiment_analysis=sentiment_analysis,
                                 save_every=save_every,
                                 percentage_per_chunk=percentage_per_chunk)

        save_sentiments(ids, tweets, results_folder, date)
        check_last_day(results_folder, date)




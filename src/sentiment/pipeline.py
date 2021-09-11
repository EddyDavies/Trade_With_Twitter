from transformers import pipeline

from sentiment.inference import get_paths, get_tweets, get_sentiments, to_dict_of_lists, save_sentiments, check_last_day
from utils import get_date_array, get_month_array, string_to_month_year, last_day_in_month


if __name__ == '__main__':
    results_folder, source = get_paths(reset=True)

    dates_range = ("2017-01-01", "2021-06-20")
    months_range = ["Jan 17", "Jun 21"]

    months = get_month_array(months_range)
    dates = get_date_array(dates_range)

    sentiment_analysis = pipeline("sentiment-analysis")

    for date in dates:
        ids, tweets = get_tweets(date)

        percentage_per_chunk = 10
        save_every = 2000
        results = get_sentiments(date, tweets, sentiment_analysis,
            save_every=save_every, percentage_per_chunk=percentage_per_chunk)

        outputs = to_dict_of_lists(results)
        outputs["ids"] = ids[:100]

        save_sentiments(outputs, date)

        check_last_day(results, results_folder, date)




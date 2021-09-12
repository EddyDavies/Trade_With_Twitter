import os

from utils import last_day_in_month, string_to_month_year


def to_dict_of_lists(LD):

    nd = {}
    for d in LD:
        for k, v in d[0].items():
            try:
                nd[k].append(v)
            except KeyError:
                nd[k] = [v]
    return nd


def check_last_day(results_folder, date):

    log_path = os.path.join(results_folder, "progress.log", )

    if last_day_in_month(date):
        with open(log_path, 'a') as f:
            month = string_to_month_year(date)
            f.write("'" + month + "',")
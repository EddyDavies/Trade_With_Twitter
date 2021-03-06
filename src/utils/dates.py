import calendar
import os
from datetime import datetime, timedelta, timezone
from typing import List, Generator
from dataclasses import dataclass

from pydantic import BaseModel



@dataclass
class SimpleDate:
    year: int
    month: int
    day: int

    def to_string(self) -> str:
        return self.to_datetime().strftime("%Y-%m-%d")

    def to_datetime(self) -> datetime:
        return datetime(year=self.year, month=self.month, day=self.day)

    @staticmethod
    def from_datetime(date: datetime) -> 'SimpleDate':
        return SimpleDate(date.year, date.month, date.day)


class TimePeriod(BaseModel):
    begin_datetime: datetime
    end_datetime: datetime

    def is_date_within_time_period(self, date: datetime) -> bool:
        return self.begin_datetime <= date < self.end_datetime

    def dates_within_range(self) -> List[SimpleDate]:
        return [SimpleDate.from_datetime(date)
                for date in self._date_generator(self.begin_datetime, self.end_datetime)]

    @staticmethod
    def _date_generator(begin_date: datetime, end_date: datetime) -> Generator[datetime, None, None]:
        curr_date = begin_date
        yield curr_date
        while curr_date < end_date:
            next_date = curr_date + timedelta(days=1)
            yield next_date
            curr_date = next_date


def twitter_date_format(date: str, time_string=None, end_of_day=False):
    # convert date in '%Y-%m-%d' format into twitter date format

    date_obj = datetime.strptime(date, '%Y-%m-%d')
    if time_string is not None:
        t = datetime.strptime(time_string, '%H:%M:%S')
        date_obj = date_obj.replace(
            hour=t.hour, minute=t.minute, second=t.second)

    elif end_of_day:
        date_obj = date_obj.replace(hour=23, minute=59, second=59)

    utc_date = date_obj.replace(tzinfo=timezone.utc)
    return str(utc_date).replace(" ", "T")


def twitter_date_format_to_day(date: str):
    # Get day in '%Y-%m-%d' format from exact twitter datetime

    date_obj = datetime.strptime(date.split('.', 1)[0], '%Y-%m-%dT%H:%M:%S')
    day = datetime.strftime(date_obj, '%Y-%m-%d')

    return day


def twitter_date_format_to_time(date: str):
    # Get day in '%Y-%m-%d' format from exact twitter datetime

    date_obj = datetime.strptime(date.split('.', 1)[0], '%Y-%m-%dT%H:%M:%S')
    time_string = datetime.strftime(date_obj, "%H:%M:%S")

    return time_string


def string_to_datetime(date: str, format: str = None):
    # get datetime object from string in '%Y-%m-%d' format
    if not format:
        return datetime.strptime(date, format)
    else:
        return datetime.strptime(date, "%Y-%m-%d")


def datetime_to_string(date: str, format: str = None):
    # get datetime object from string in '%Y-%m-%d' format
    if not format:
        return datetime.strftime(date, format)
    else:
        return datetime.strftime(date, "%Y-%m-%d")

def string_to_month_year(date: str):
    return datetime.strptime(date, "%Y-%m-%d").strftime('%b %y')


# @json_print
def get_date_range(months: list):
    # returns first and last date for a specified month or range between 2 months

    first = datetime.strptime(months[0], "%b %y")

    if len(months) != 1:
        last_month = datetime.strptime(months[-1], "%b %y")
    else:
        last_month = first

    # set the last datetime to the last day in the month
    last = calendar.monthrange(int(last_month.strftime("%y")), int(last_month.strftime("%m")))[1]
    last = datetime.strptime(f"{last} {months[-1]}", "%d %b %y")

    first = datetime.strftime(first, "%Y-%m-%d")
    last = datetime.strftime(last, "%Y-%m-%d")

    return first, last


# @json_print
def get_date_array(date_range: tuple):
    # get a list of dates from year and month name in '%Y-%m-%d' format
    first, last = date_range

    current = datetime.strptime(first, "%Y-%m-%d")
    last = datetime.strptime(last, "%Y-%m-%d")

    date_array: List[str] = []
    end_of_range = True

    while end_of_range:
        current_string = datetime.strftime(current, "%Y-%m-%d")
        date_array.append(current_string)
        end_of_range = current != last
        current += timedelta(days=1)

    return date_array


def last_day_in_month(date_string):

    date = datetime.strptime(date_string, "%Y-%m-%d")
    last_day = calendar.monthrange(date.year, date.month)[1]

    if last_day == date.day:
        return True
    else:
        return False


def get_month_array(month_range):
    # returns first and last month with year for a specified range between 2 months with years

    current_month = datetime.strptime(month_range[0], "%b %y")
    last_month = datetime.strptime(month_range[-1], "%b %y")
    month_array: List[str] = []
    end_of_range = True

    while end_of_range:
        current_string = datetime.strftime(current_month, "%b %y")
        month_array.append(current_string)

        next_month = int(datetime.strftime(current_month, "%m"))
        next_year = int(datetime.strftime(current_month, "%Y"))

        if next_month == 12:
            next_month = 0
            next_year += 1
        next_month += 1

        end_of_range = current_month != last_month
        current_month = current_month.replace(month=next_month, year=next_year)

    return month_array


if __name__ == "__main__":
    print(get_date_array(get_date_range(["Jan 17"])))

    # months = get_month_array(["Jan 18", "Jun 21"])


def check_last_day(results_folder, date):

    log_path = os.path.join(results_folder, "progress.log", )

    if last_day_in_month(date):
        with open(log_path, 'a') as f:
            month = string_to_month_year(date)
            f.write("'" + month + "',")
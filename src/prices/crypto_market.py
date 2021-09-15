from dataclasses import dataclass
from itertools import tee

from pandas import DataFrame

from typing import Optional, List, Literal, Dict, Iterable, TypeVar, Tuple, cast, Generator
import os
from datetime import datetime, timedelta
from copy import deepcopy
from pycoingecko import CoinGeckoAPI

from prices.technical_indicators import generate_ta


class ApiKeyEnvVarMissing(Exception):
    pass


T = TypeVar('T')


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


@dataclass
class TimePeriod:
    begin_datetime: datetime
    end_datetime: datetime

    def is_simple_date_within_time_period(self, date_string: str) -> bool:
        parsed_date = TimePeriod._parse_date_string(date_string)
        return self.is_date_within_time_period(parsed_date)

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

    @staticmethod
    def from_simple_date_strings(begin_date_string: str, end_date_string: str) -> 'TimePeriod':
        return TimePeriod(TimePeriod._parse_date_string(begin_date_string),
                          TimePeriod._parse_date_string(end_date_string))

    @staticmethod
    def _parse_date_string(date_string: str) -> datetime:
        split_date_string = date_string.split("-")
        year = int(split_date_string[0])
        month = int(split_date_string[1])
        day = int(split_date_string[2])
        return datetime(year=year, month=month, day=day)


@dataclass
class MarketDataRecord:
    date: str
    open_price: float
    close_price: float
    percent_change_open: float
    percent_change_close: float
    volume: float
    market_cap: float


# @dataclass
# class AlphaVantageCredentials:
#     api_key: str
#
#     @staticmethod
#     def from_env_var(env_var: Optional[str] = "ALPHA_VANTAGE_KEY") -> 'AlphaVantageCredentials':
#         try:
#             key = os.environ[env_var]
#             return AlphaVantageCredentials(key)
#         except KeyError as e:
#             raise ApiKeyEnvVarMissing(f'Could not find env var {env_var} for setting up AlphaVantage credentials')
#
#
class CryptoMarketUtils:
    """
    Uses the Coingecko API under the hood
    """

    def __init__(self):
        pass

    def generate_crypto_dataset(self, cryptos: List[str],
                                save_dir: Optional[str] = "../data/market-data",
                                time_period: Optional[TimePeriod] = None) -> List[DataFrame]:

        dataframes = [DataFrame(self.market_data_records(crypto, time_period))
                      for crypto in cryptos]

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        for crypto, df in zip(cryptos, dataframes):
            df.to_csv(f'{save_dir}/{crypto}.csv', index=False)

        return dataframes

    def market_data_records(self, crypto: str, time_period: Optional[TimePeriod] = None) -> List[MarketDataRecord]:
        # we use this long time period to get extra data from Coingecko
        # the extra values ensure that we don't need to artificially engineer any percentage_diff numbers
        long_time_period = TimePeriod.from_simple_date_strings("2016-06-01", "2021-07-01")
        cg_data = CoinGeckoAPI().get_coin_market_chart_range_by_id(crypto, vs_currency="USD",
                                                                   from_timestamp=long_time_period.begin_datetime.timestamp(),
                                                                   to_timestamp=long_time_period.end_datetime.timestamp())

        crypto_data = CryptoMarketUtils.process_cg_data(cg_data)
        crypto_data_dates = [simple_date.to_string() for simple_date in long_time_period.dates_within_range()]
        crypto_data_processed = self._process_market_data(crypto_data)

        time_period_filter_fn = (
            lambda date: time_period.is_simple_date_within_time_period(date)) if time_period is not None else (
            lambda _: True)
        return [MarketDataRecord(date, data["Open"], data["Close"],
                                 data["percentage_diff_open"], data["percentage_diff_close"],
                                 data["Volume"], data["Market_Cap"])
                for date, data in zip(crypto_data_dates, crypto_data_processed)
                if time_period_filter_fn(date)]


    @staticmethod
    def process_cg_data(cg_data: Dict[str, List]) -> List[Dict]:
        """
        Just turns the Coingecko data response into a format expected by rest of the code
        :param df:
        :return: the dataframe as an appropriate dictionary
        """
        prices = cg_data["prices"]
        volumes = cg_data["total_volumes"]
        market_caps = cg_data["market_caps"]

        market_data = []
        for x, price in enumerate(prices):

            try:
                market_cap = market_caps[x][1]
            except:
                market_cap = 0
            try:
                volume = volumes[x][1]
            except:
                volume = 0

            day_data = {
                "Open": price[1],
                 "Close": price[1],
                 "Volume": volume,
                 "Market_Cap": market_cap
            }

            market_data.append(day_data)

        return market_data
        # return [{"Open": price[1], "Close": price[1]} for price in prices]


    def _process_market_data(self, market_data_dicts: List[Dict]) -> List[Dict]:
        """
        Given market data from the API, adds in the daily percentage increase/decrease
        :param market_data_dicts: dictionaries of market data (according to format supplied by the AlphaVantage API)
        :return: the market data dicts, but with some post-processing (adding percentage increase/decrease for now)
        """
        open_prices = [float(market_data_dict["Open"]) for market_data_dict in market_data_dicts]
        close_prices = [float(market_data_dict["Close"]) for market_data_dict in market_data_dicts]

        percentage_differences_open = [0] + [(tup[1] - tup[0]) * 100 / tup[0] for tup in self.pairwise(open_prices)]
        percentage_differences_close = [0] + [(tup[1] - tup[0]) * 100 / tup[0] for tup in self.pairwise(close_prices)]

        updated_dicts = cast(List[Dict], [])
        for market_data_dict, percentage_diff_open, percentage_diff_close in zip(market_data_dicts,
                                                                                 percentage_differences_open,
                                                                                 percentage_differences_close):
            updated_dict = deepcopy(market_data_dict)
            updated_dict.update({
                "percentage_diff_open": percentage_diff_open,
                "percentage_diff_close": percentage_diff_close
            })

            updated_dicts.append(updated_dict)

        return updated_dicts

    @staticmethod
    def pairwise(seq: Iterable[T]) -> Iterable[Tuple[T, T]]:
        """
        s -> (s0,s1), (s1,s2), (s2, s3), ...
        stolen from https://docs.python.org/3/library/itertools.html#recipes
        :param seq:
        :return:
        """
        a, b = tee(seq)

        next(b, None)
        return zip(a, b)


if __name__ == '__main__':
    cmu = CryptoMarketUtils()

    cryptos = ["bitcoin", "ethereum"]

    cmu.generate_crypto_dataset(cryptos)
    generate_ta(cryptos)

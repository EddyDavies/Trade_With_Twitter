import copy
from typing import Optional

import numpy as np
import pandas as pd


STOCKS = [
    # "AAPL",
    "BTC-USD",
    # "INTC",
]

PATH = "../data/test/{}.csv"
HOLD, BUY, SELL = 0, 1, 2


class Stonks:
    def __init__(
            self,
            training_dataset_filepath: str,
            currency: str,
            initial_money=1000,
            verbose=0,
            use_sentiment: Optional[bool] = True,
            testing=False,
            window_size=10,
    ):
        # The shape of the state returned at each time step and the number of actions the agent can make
        # self.observation_shape = ((window_size * 2) + 1) if use_sentiment else (window_size + 1),
        self.observation_shape = (window_size * len(list(pd.read_csv(training_dataset_filepath).columns)[1:]))+1
        self.actions = 3

        self.training_dataset_filepath = training_dataset_filepath
        self.use_sentiment = use_sentiment
        self.currency = currency

        # Total reward obtained over time
        self.total_reward = 0
        # Current step in the episode
        self._step = 0
        self.__end_step = 0

        self.window_size = window_size

        # Store the name of the currency and a dataframe for the data.
        self.currency: Optional[str] = None
        self.data: Optional[pd.DataFrame] = None

        # Store the amount of money owned and the amoaunt invested in coins.
        self._initial_money = initial_money
        self._wallet = 0
        self._portfolio = 0

        self._transaction_fee_buy = 0.01
        self._transaction_fee_sell = 0.01

        # Show print statements
        self.verbose = verbose

        self.testing = testing

    def reset(self):
        """
        Reset the environment.

        Reset the environment and load up a new stock to load then returns the current observation.

        Returns:
            Current environment observation.
        """
        self.total_reward = 0

        self._portfolio = 0
        self._wallet = self._initial_money

        # Load up a chosen currency to run
        self.currency = np.random.choice(STOCKS)
        self.data = self._load_data(self.training_dataset_filepath)
        self.observation_shape = self.data.shape[1]

        self._step = self.window_size
        self.__end_step = (len(self.data) // 4) * 3

        if self.testing:
            self._step = (len(self.data) // 4) * 3
            self.__end_step = len(self.data) - 1

        self._print(f"Loaded: {self.currency}")

        return self._get_observation()

    def step(self, action: int) -> tuple:
        """
        Interact with the environment.

        Stepping through the environment will perform the action given [Hold, Buy, Sell].

        Args:
            action. The action taken.

        Return:
            A tuple of data:
                current environment observation,
                reward of the performed action,
                is the episode over,
                current episode info.
        """
        if self.data is None:
            raise Exception("Call .reset() on environment.")

        # Perform action and return action
        self._perform_action(action)

        # Calculate initial assets
        # Reward should be percentage improvement on next day due to action taken
        reward = self._calculate_reward()
        self.total_reward += reward

        # Add info about the current environ that may be useful for plotting but not for training.
        info = {"currency": self.currency, "value": self._crypto_value(), "assets": self.total_assets()}

        # Increment the step.
        self._step += 1

        # Get new observation and check if this is the terminal step.
        new_observation = self._get_observation()

        done = self._step == self.__end_step
        # End the environment
        if done:
            self.data = None
            self.currency = None

        return new_observation, reward, done, info

    def _calculate_reward(self):
        """
        If the portfolio value increases tomorrow, then we took a good action today. This method returns the percentage
        increase/decrease of our portfolio for tomorrow, which is a suitable metric for calculating reward
        :return:
        """
        coin_value_today = self._crypto_value()
        coin_value_tomorrow = self._crypto_value_tomorrow()

        if coin_value_today < coin_value_tomorrow:
            # Value goes up so we want to own
            if self._portfolio > 0:
                return (coin_value_tomorrow - coin_value_today) / coin_value_today
            else:
                return -(coin_value_tomorrow - coin_value_today) / coin_value_today

        if coin_value_today > coin_value_tomorrow:
            # Coin value goes down so we want to sell
            if self._wallet > 0:
                return -(coin_value_tomorrow - coin_value_today) / coin_value_today
            else:
                return (coin_value_tomorrow - coin_value_today) / coin_value_today

        return 0

    def _get_observation(self):
        """
        Observe the current state of the environment.

        Returns:
            Current observation of the environment at the current time step.
        """
        assert self.data is not None



        # Add an int to describe whether we've invested
        observations = []
        names = self.data.columns
        for name in names:
            obvs = self.data[name][self._step - self.window_size:self._step].to_numpy().copy()
            # obvs = self.data[name][self._step].copy()
            observations.append(obvs)

        # col = self.data["percent_change_close"]
        # window_value = self.data["percent_change_close"][self._step - self.window_size:self._step].to_numpy().copy()
        # positive_window = self.data["Positive"][self._step - self.window_size:self._step].to_numpy().copy()
        # negative_window = self.data["Negative"][self._step - self.window_size:self._step].to_numpy().copy()
        #
        # # Add an int to describe whether we've invested
        # observations = []
        # observations.append(window_value)
        # observations.append(np.array([int(self._portfolio > 0)]))
        # # current capital, not invested
        # # current value of invested in BTC
        # if self.use_sentiment:
        #     observations.append(positive_window-negative_window)
        #

        # observations.append(int(self._portfolio > 0))
        observations.append(np.array([int(self._portfolio > 0)]))
        # current capital, not invested
        # current value of invested in BTC
        # Add LowPass instead of MMA and other technical indicators

        return np.concatenate(observations)
        # return observations

    def _perform_action(self, action: int):
        """
        Interact with the environment to purchase crypto.

        This function adds buy and sell functionality into the environment. The number of coins purchased is calculated
        by dividing the _wallet with the current _crypto_value. When selling we simply multiply the current value by
        the number of coins owned.

        TODO: Incoperate transaction fees.

        Args:
            action: The action to perform

        Return:
            Profit from performing that action.
        """
        assert 0 <= action <= 2

        if action == BUY:
            if self._wallet > 0:
                total_coins = (self._wallet * (1 - self._transaction_fee_buy)) / self._crypto_value()

                self._print(f"\tBought {total_coins} coins.")

                self._portfolio = total_coins
                self._wallet = 0

        if action == SELL:
            if self._portfolio > 0:
                port_value = self._crypto_value() * (self._portfolio * (1 - self._transaction_fee_sell))

                self._print(f"\tSold {self._portfolio} coins.")

                self._wallet = port_value
                self._portfolio = 0

    def total_assets(self):
        """
        Calculate the total weath of the actor.

        This function is used to calculate the total weather of the actor which is used to calculate the gain/loss per
        time step.

        Return:
            float of the total weath.
        """
        return self._wallet + self._portfolio * self._crypto_value()

    def _total_assets_tomorrow(self):
        """
        Similar to self._total_assets() but calculates what will be the total wealth of the actor tomorrow
        :return:
        """
        return self._wallet + self._portfolio * self._crypto_value_tomorrow()

    def _crypto_value(self):
        """
        Get the current crypto value at the current time step.

        This function will access the "Adj Close" stock price currently. This should be remoddelled to the actual data
        we train upon.

        Returns:
            Stock value at the current time step.
        """
        return self.data["close_price"][self._step]

    def _crypto_value_tomorrow(self):
        """
        Similar to self._crypto_value, but returns the crypto's value for the following day
        :return:
        """
        return self.data["close_price"][self._step + 1]

    def _print(self, message: str):
        """
        Print a message.

        This function will check if the verbose setting has been turned on. Potentially useful for debugging. If set to
        true, then this function will print out the message given.

        Args:
            message: The string to print.
        """
        if self.verbose:
            print(message)

    @staticmethod
    def _load_data(dataset_filepath: str) -> pd.DataFrame:
        return pd.read_csv(dataset_filepath, index_col='date')
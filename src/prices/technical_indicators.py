import pandas as pd
import pandas_ta as ta
import yfinance as yf
import matplotlib as plt


if __name__ == '__main__':

    df = pd.DataFrame() # Empty DataFrame

    # Load data
    df = pd.read_csv("../data/market-data/bitcoin.csv", sep=",")
    # OR if you have yfinance installed
    # df = df.ta.ticker("aapl")

    # VWAP requires the DataFrame index to be a DatetimeIndex.
    # Replace "datetime" with the appropriate column from your DataFrame
    df.set_index(pd.DatetimeIndex(df["date"]), inplace=True)
    df = df[['open_price', 'close_price', 'percent_change_open', 'percent_change_close']]


    CustomStrategy = ta.Strategy(
        name="Momo and Volatility",
        description="SMA 50,200, BBANDS, RSI, MACD and Volume SMA 20",
        ta=[
            {"kind": "sma", "length": 50},
            {"kind": "sma", "length": 200},
            {"kind": "bbands", "length": 20},
            {"kind": "rsi"},
            {"kind": "macd", "fast": 8, "slow": 21},
            {"kind": "sma", "close": "volume", "length": 20, "prefix": "VOLUME"},
        ]
    )
    # To run your "Custom Strategy"
    df.ta.strategy(CustomStrategy)

    # Apply Strategy
    # df.ta.strategy(ta.CommonStrategy)
    #
    # df.ta.log_return(cumulative=True, append=True)
    # df.ta.percent_return(cumulative=True, append=True)

    # New Columns with results
    df.columns
    df.reset_index(inplace=True)

    df.plot(x='date', y='close_price', figsize=(15, 6), linestyle='--', marker='*', markerfacecolor='r', color='y',
            markersize=10)

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

    # Take a peek
    df.tail()

# vv Continue Post Processing vv
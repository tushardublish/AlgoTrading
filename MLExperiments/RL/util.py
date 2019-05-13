"""MLT: Utility code."""

import os
import pandas as pd
import matplotlib.pyplot as plt

def get_path(symbol, base_dir=os.path.join("C:\Program Files\AmiBroker\Formulas\Data")):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def get_data(symbol):
    df = pd.DataFrame()
    df = pd.read_csv(get_path(symbol), index_col='Date/Time',
            parse_dates=True, dayfirst=True, na_values=['nan'])
    df = df.dropna()
    df = df.drop(columns=['Ticker'])
    return df

def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()

def get_date_range(dates, start_date, end_date):
    pass

import numpy as np
import pandas as pd

# Σταθερές 
SMA_10_PERIOD = 10
SMA_50_PERIOD = 50

# Η λίστα των χαρακτηριστικών
FEATURES = ['Daily_Return', 'Close_vs_SMA10', 'Close_vs_SMA50', 'SMA10_vs_SMA50']


def add_features(df):
    df = df.copy()

    # Ημερήσια ποσοστιαία απόδοση
    df['Daily_Return'] = df['Close'].pct_change()

    df['SMA_10'] = df['Close'].rolling(window=SMA_10_PERIOD).mean()
    df['SMA_50'] = df['Close'].rolling(window=SMA_50_PERIOD).mean()

    # Σχετικοί δείκτες:
    df['Close_vs_SMA10'] = df['Close'] / df['SMA_10'] - 1
    df['Close_vs_SMA50'] = df['Close'] / df['SMA_50'] - 1
    df['SMA10_vs_SMA50'] = df['SMA_10'] / df['SMA_50'] - 1

    # Προστασία από άπειρες τιμές 
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    return df

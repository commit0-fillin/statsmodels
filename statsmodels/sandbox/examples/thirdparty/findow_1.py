"""A quick look at volatility of stock returns for 2009

Just an exercise to find my way around the pandas methods.
Shows the daily rate of return, the square of it (volatility) and
a 5 day moving average of the volatility.
No guarantee for correctness.
Assumes no missing values.
colors of lines in graphs are not great

uses DataFrame and WidePanel to hold data downloaded from yahoo using matplotlib.
I have not figured out storage, so the download happens at each run
of the script.

Created on Sat Jan 30 16:30:18 2010
Author: josef-pktd
"""
import os
from statsmodels.compat.python import lzip
import numpy as np
import matplotlib.finance as fin
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
start_date = dt.datetime(2007, 1, 1)
end_date = dt.datetime(2009, 12, 31)
dj30 = ['MMM', 'AA', 'AXP', 'T', 'BAC', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DD', 'XOM', 'GE', 'HPQ', 'HD', 'INTC', 'IBM', 'JNJ', 'JPM', 'KFT', 'MCD', 'MRK', 'MSFT', 'PFE', 'PG', 'TRV', 'UTX', 'VZ', 'WMT', 'DIS']
mysym = ['msft', 'ibm', 'goog']
indexsym = ['gspc', 'dji']
dmall = {}
for sy in dj30:
    dmall[sy] = getquotes(sy, start_date, end_date)
pawp = pd.WidePanel.fromDict(dmall)
print(pawp.values.shape)
paclose = pawp.getMinorXS('close')
paclose_ratereturn = paclose.apply(np.log).diff()
if not os.path.exists('dj30rr'):
    paclose_ratereturn.save('dj30rr')
plt.figure()
paclose_ratereturn.plot()
plt.title('daily rate of return')
paclose_ratereturn_vol = paclose_ratereturn.apply(lambda x: np.power(x, 2))
plt.figure()
plt.title('volatility (with 5 day moving average')
paclose_ratereturn_vol.plot()
paclose_ratereturn_vol_mov = paclose_ratereturn_vol.apply(lambda x: np.convolve(x, np.ones(5) / 5.0, 'same'))
paclose_ratereturn_vol_mov.plot()
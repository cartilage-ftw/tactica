import pandas as pd
import os
import matplotlib.pyplot as plt

from plotting_prefs import *
#plt.rcParams['figure.dpi'] = 150

"""
load all .csv files
"""
contents = os.listdir('../data/mar16/')

file_names = []
other_names = []
for f in contents:
    if '.5kV' in f and ".csv" in f:
        file_names.append(f)
    elif ".csv" in f:
        other_names.append(f)

dataframes = []
for f in other_names:
    df = pd.read_csv('../data/mar16/' + f, skiprows=2, names=['Arrival Time', 'Signal', 'Trigger'])
    dataframes.append(df)

#print(dataframes[0].head())

"""
Plot the data
"""

fig, axes = plt.subplots(len(dataframes), 1, sharex=True, figsize=(6,8))
for ax, df, f in zip(axes, dataframes, other_names):
    ax.plot(df['Arrival Time']*1E6, df['Signal_BGS'], label=f)
    ax.legend()

plt.tight_layout()
plt.show()
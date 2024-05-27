import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib as mpl

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

set_df = df[df["set"] == 1]
plt.plot(set_df["acc_y"])

plt.plot(set_df["acc_y"].reset_index(drop=True))

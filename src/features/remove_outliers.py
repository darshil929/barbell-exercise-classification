import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

outliers_columns = list(df.columns[:6])

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100

# Plotting outliers using Box Plot method
df[["acc_y", "label"]].boxplot(by="label", figsize=(20, 10))
plt.show()

df[outliers_columns[:3] + ["label"]].boxplot(
    by="label", figsize=(20, 10), layout=(1, 3)
)
plt.show()

df[outliers_columns[3:] + ["label"]].boxplot(
    by="label", figsize=(20, 10), layout=(1, 3)
)
plt.show()


# Marking Outliers using IQR method
def mark_outliers_iqr(dataset, col):
    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset


# Plotting Outliers on the basis of binary outlier score(True / False)
def plot_binary_outliers(dataset, col, outlier_col, reset_index):
    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()

    plt.xlabel("samples")
    plt.ylabel("value")

    # Plots non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    # Plots data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )

    plt.legend(
        ["not outlier " + col, "outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()


# plotting outliers for a single column
col = "acc_x"
dataset = mark_outliers_iqr(df, col)
plot_binary_outliers(
    dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
)

for col in outliers_columns:
    dataset = mark_outliers_iqr(df, col)
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
    )

# Checking for Normal Distribution
df[outliers_columns[:3] + ["label"]].plot.hist(
    by="label", figsize=(20, 20), layout=(3, 3)
)
plt.show()

df[outliers_columns[3:] + ["label"]].plot.hist(
    by="label", figsize=(20, 20), layout=(3, 3)
)
plt.show()


# Chauvenet method for outlier detection
# Finding outliers in the specified column of datatable and adding a binary column with the same name extended with '_outlier' that expresses the result per data point
def mark_outliers_chauvenet(dataset, col, C=2):
    dataset = dataset.copy()
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    deviation = abs(dataset[col] - mean) / std

    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    for i in range(0, len(dataset.index)):
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset


for col in outliers_columns:
    dataset = mark_outliers_chauvenet(df, col)
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
    )


# Local Outlier Factor Function for Outlier Detection
def mark_outliers_lof(dataset, columns, n=20):
    dataset = dataset.copy()

    lof = LocalOutlierFactor(n_neighbors=n)
    data = dataset[columns]
    outliers = lof.fit_predict(data)
    X_scores = lof.negative_outlier_factor_

    dataset["outlier_lof"] = outliers == -1
    return dataset, outliers, X_scores


dataset, outliers, X_scores = mark_outliers_lof(df, outliers_columns)
for col in outliers_columns:
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col="outlier_lof", reset_index=True
    )

# grouping by labels and checking for outliers
label = "bench"

for col in outliers_columns:
    dataset = mark_outliers_iqr(df[df["label"] == label], col)
    plot_binary_outliers(dataset, col, col + "_outlier", reset_index=True)

for col in outliers_columns:
    dataset = mark_outliers_chauvenet(df[df["label"] == label], col)
    plot_binary_outliers(dataset, col, col + "_outlier", reset_index=True)

dataset, outliers, X_scores = mark_outliers_lof(
    df[df["label"] == label], outliers_columns
)
for col in outliers_columns:
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col="outlier_lof", reset_index=True
    )

# Testing on individual columns and replacing outlier values with NaN
col = "gyr_y"
dataset = mark_outliers_chauvenet(df, col=col)
dataset[dataset["gyr_y_outlier"]]
dataset.loc[dataset["gyr_y_outlier"], "gyr_z"] = np.nan

# looping over all the columns
outliers_removed_df = df.copy()
for col in outliers_columns:
    for label in df["label"].unique():
        dataset = mark_outliers_chauvenet(df[df["label"] == label], col)

        dataset.loc[dataset[col + "_outlier"], col] = np.nan

        # Updating the columns in the dataframe
        outliers_removed_df.loc[(outliers_removed_df["label"] == label), col] = dataset[
            col
        ]

        n_outliers = len(dataset) - len(dataset[col].dropna())
        print(f"Removed {n_outliers} from {col} for {label}")

outliers_removed_df.info()

# Exporting new dataframe
outliers_removed_df.to_pickle("../../data/interim/02_outliers_removed_chauvenet.pkl")

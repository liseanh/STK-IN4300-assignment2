import numpy as np
import pandas as pd
import sklearn.model_selection as sklms
import sklearn.preprocessing as sklpre

df = pd.read_csv("data/ozone_496obs_25vars.txt", header=0, sep=" ")

variables = df.loc[:, df.columns != "FFVC"]
outcome = df["FFVC"].values

onehot_sex = pd.get_dummies(df["SEX"]).set_axis(
    ["Male", "Female"], axis=1, inplace=False
)

variables.drop("SEX", axis=1, inplace=True)

variables = variables.join(onehot_sex)


X_train, X_test, y_train, y_test = sklms.train_test_split(
    variables.values, outcome, test_size=0.2
)

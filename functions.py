# functions.py
import numpy
import pandas as pd
import csv
import requests as rq
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


def update_sales(sales, teams):
    for team in teams:
        if team in sales:
            sales[team] += 1

def compute_vif(df, cols):
    X = df[cols].dropna()
    vif = pd.DataFrame()
    vif["feature"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif

def compute_val_trend_slope(row):
    years = np.array([2018, 2022, 2025]).reshape(-1, 1)
    vals = np.array([row['val_18'], row['val_22'], row['val_25']]).reshape(-1, 1)
    model = LinearRegression().fit(years, vals)
    return model.coef_[0][0]
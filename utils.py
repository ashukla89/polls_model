import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
import math
import random

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# function to help deal with categorical v. numerical inputs
def split_data(df,X_vars_cat,X_vars_num,y_vars):
    X_num = df[X_vars_num]
    if len(X_vars_cat) > 0:
        X_cat = pd.concat([pd.get_dummies(df[X_vars_cat],drop_first=True,dtype=float)],axis=1)
        X = pd.concat([X_cat,X_num],axis=1)
    else:
        X = X_num
    if len(y_vars) > 0:
        y = df[y_vars]
    else:
        y = None
    
    return X, y

# find the most recent quarterly economic research date
def quarterly_date(dt: datetime):
    if dt.month < 4:
        new_date = f"{dt.year-1}-12-31"
    elif dt.month < 7:
        new_date = f"{dt.year}-03-31"
    elif dt.month < 10:
        new_date = f"{dt.year}-06-30"
    else:
        new_date = f"{dt.year}-09-30"
    return datetime.strptime(new_date,"%Y-%m-%d")

def calc_variance(n,p):
    var = n * (p * (1-p))
    return var
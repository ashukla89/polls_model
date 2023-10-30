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

from utils import split_data, quarterly_date, calc_variance
from import_and_process import import_and_process
from fundamentals import generate_fundamentals

# disable warnings-- we are getting a lot

import warnings
warnings.filterwarnings('ignore')

### IMPORT ALL DATA ###

demo, map_dict, demo_r, cal, histe, histe_q1, res, res_ee, pollse, pshares, ppshares, ppshares_sc, scenarios = import_and_process()

### SET ASIDE SOME YEARS FOR VALIDATION ###

setaside = random.choices(res_ee.year.unique(),k=8)

### BRING IN PREDICTIONS AND ERRORS FROM FUNDAMENTALS MODEL ###

fund_prov_mse, fund_prov_mae, fund_comparison_df = generate_fundamentals(res_ee,scenarios,pshares)

### BRING IN PREDICTIONS AND ERRORS FROM PROVINCE-SPECIFIC RANDOM FOREST MODELS, FOR EACH SCENARIO ###

model_outputs = []

for target_geo in res_ee.province.unique():
    row = {}
    row['target_geo'] = target_geo
    daily_tonly_mse, daily_tonly_mae, daily_tonly_avg = target_only_model(target_geo,pollse,scenarios,pshares,ppshares,pshares_sc)
    # interpolate missing values in error datasets for the two RF models
    daily_tonly_mse_interp = days_df.\
        merge(daily_tonly_mse.sort_values(by='day'),on='day',how='left').interpolate(method='linear').bfill()
    daily_tonly_mae_interp = days_df.\
        merge(daily_tonly_mae.sort_values(by='day'),on='day',how='left').interpolate(method='linear').bfill()
    row['daily_tonly_mse'] = daily_tonly_mse_interp
    row['daily_tonly_mae'] = daily_tonly_mae_interp
    row['daily_tonly_avg'] = daily_tonly_avg
    model_outputs.append(row)

### BRING IN PREDICTIONS AND ERRORS FROM ALL-BUT-PROVINCE RANDOM FOREST MODELS, FOR EACH SCENARIO ###

for row in model_outputs:
    target_geo = row['target_geo']
    daily_allbut_mse, daily_allbut_mae, daily_allbut_avg = target_only_model(target_geo,pollse,scenarios,pshares,ppshares,pshares_sc)
    daily_allbut_mse_interp = days_df.\
        merge(daily_allbut_mse.sort_values(by='day'),on='day',how='left').interpolate(method='linear').bfill()
    daily_allbut_mae_interp = days_df.\
        merge(daily_allbut_mae.sort_values(by='day'),on='day',how='left').interpolate(method='linear').bfill()
    row['daily_tonly_mse'] = daily_allbut_mse_interp
    row['daily_tonly_mae'] = daily_allbut_mae_interp
    row['daily_tonly_avg'] = daily_allbut_avg

### FOR EACH PROVINCE, GENERATE WEIGHTED DAILY POLLING ESTIMATES - NOT REALLY FINISHED ###
    
# turn the fundamentals models into a 75-day set for each province too

days_df = pd.DataFrame(pollse.days_to_election.unique(),columns=['day'])
length = 74  # Desired length

for target_geo in res_ee.province.unique():

    daily_fund_mse = pd.DataFrame(np.tile(fund_prov_mae[fund_prov_mse['province']==target_geo][pshares].values, (length, 1)), columns=pshares)
    daily_fund_mae = pd.DataFrame(np.tile(fund_prov_mae[fund_prov_mae['province']==target_geo][pshares].values, (length, 1)), columns=pshares)
    daily_fund_avg = pd.DataFrame(np.tile(fund_comparison_df[fund_comparison_df['province']==target_geo][[col + "_pred" for col in pshares]].mean().values, (length, 1)), columns=pshares)

    daily_fund_mse = days_df.merge(daily_fund_mse,left_index=True,right_index=True,how='left')
    daily_fund_mae = days_df.merge(daily_fund_mae,left_index=True,right_index=True,how='left')
    daily_fund_avg = days_df.merge(daily_fund_avg,left_index=True,right_index=True,how='left')

    daily_fund_mse.columns = daily_tonly_mse.columns
    daily_fund_mae.columns = daily_tonly_mae.columns
    daily_fund_avg.columns = daily_tonly_avg.columns
    
# actually run an average


### HERE, WE WOULD HAVE RUN A MONTE CARLO SIMULATION. SADLY WE DIDN'T GET THAT FAR ###



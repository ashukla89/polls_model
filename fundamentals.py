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

from utils import split_data, calc_variance

def generate_fundametals(res_ee,scenarios,pshares):

    X_vars_cat = ['province','region','party_in_power']
    X_vars_num = ['python_pop_share', 'cobolite_pop_share','javarian_pop_share',\
                      'year_on_year_gdp_pct_change', 'unemployment_rate',\
                       'year_on_year_inflation', 'year_on_year_stock_mkt_pct_change'
                 ]
    y_vars = ['cc_share', 'dgm_share', 'pdal_share', 'ssp_share']

    # subset for target-only
    sub = res_ee.copy()

    X, y = split_data(sub,X_vars_cat,X_vars_num,y_vars)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    tmodel = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    tmodel.fit(X_train, y_train)

    y_pred = tmodel.predict(X_test)

    # y_pred range gut check
    y_pred = pd.DataFrame(y_pred)
    y_pred.columns = [col + "_pred" for col in y_test.columns]
    y_pred.index = y_test.index
    print(y_pred.sum(axis=1).describe())

    # Scale as specified
    y_pred = y_pred.divide(y_pred.sum(axis=1),axis=0)

    # Evaluation
    mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    print("Mean squared errors for y_test:",mse)
    print("Mean absolute errors for y_test:",mae)

    # 10-fold cross-validation
    scores = cross_val_score(tmodel, X_train, y_train, cv=10)

    print(f"Cross-validated scores: {scores}")
    print(f"Mean CV Score: {scores.mean()}")
    print(f"Standard Deviation of CV Scores: {scores.std()}")

    # Retrieve feature importances for each target
    fis = []
    for i, target in enumerate(y.columns):
        importances = tmodel.estimators_[i].feature_importances_
    #     print(f"Feature importances for target {target}:")
        fi = pd.Series(importances,name=target,index=X_train.columns)
        fis.append(fi)
    #     for feat, imp in zip(X.columns, importances):
    #         print(f"{feat}: {imp:.4f}")
    #     print("-" * 30)

    imp_df = pd.concat(fis,axis=1)
    print(imp_df)

    # Concatenate the test and pred dataframes side-by-side
    fund_comparison_df = pd.concat([y_test, y_pred], axis=1)
    fund_comparison_df = \
        fund_comparison_df.merge(res_ee[['year','province']],left_index=True,right_index=True,how='left')

    # Calculate errors for each set of test-pred pairs
    fund_mse_results = []
    fund_mae_results = []
    for name, group in fund_comparison_df.groupby('province'):
        true_vals = group[pshares]
        preds = group[[col + "_pred" for col in pshares]]
        mse = mean_squared_error(true_vals, preds, multioutput='raw_values')
        mae = mean_absolute_error(true_vals, preds, multioutput='raw_values')
        mse_result = {}
        mae_result = {}
        mse_result['province'] = name
        mae_result['province'] = name
        mse_res_dict = dict(zip(pshares,mse))
        mae_res_dict = dict(zip(pshares,mae))
        mse_result.update(mse_res_dict)
        mae_result.update(mae_res_dict)
        fund_mse_results.append(mse_result)
        fund_mae_results.append(mae_result)

    fund_prov_mse = pd.DataFrame(fund_mse_results)
    fund_prov_mae = pd.DataFrame(fund_mae_results)

    return fund_prov_mse, fund_prov_mae, fund_comparison_df
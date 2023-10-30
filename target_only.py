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

from utils import split_data, quarterly_date

def target_only_model(target_geo,pollse,pshares,ppshares,pshares_sc):

    ### TARGET ONLY ###

    X_vars_cat = ['pollster','mode','population_surveyed','sponsor']
    X_vars_num = ['dgm_poll_scaled','pdal_poll_scaled','cc_poll_scaled','ssp_poll_scaled','sample_size',
                  'undecided_poll_share','days_to_election'
                 ]
    y_vars = ['dgm_share','pdal_share','cc_share','ssp_share']

    # subset for target-only
    sub = pollse[pollse['geography']==target_geo]

    X, y = split_data(sub[(~sub['year'].isin(setaside))],X_vars_cat,X_vars_num,y_vars)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # these are the best params, apparently, thanks to GridSearch.
    best_params = {'estimator__bootstrap': True,
     'estimator__max_depth': None,
     'estimator__max_features': 'sqrt',
     'estimator__min_samples_leaf': 2,
     'estimator__min_samples_split': 5,
     'estimator__n_estimators': 25}

    # unpack
    best_rf_params = {key.replace('estimator__', ''): value for key, value in best_params.items()}


    tonly_model = MultiOutputRegressor(RandomForestRegressor(**best_rf_params, random_state=42))
    tonly_model.fit(X_train, y_train)

    y_pred = tonly_model.predict(X_test)

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
    scores = cross_val_score(tonly_model, X_train, y_train, cv=10)

    print(f"Cross-validated scores: {scores}")
    print(f"Mean CV Score: {scores.mean()}")
    print(f"Standard Deviation of CV Scores: {scores.std()}")

    # Retrieve feature importances for each target
    fis = []
    for i, target in enumerate(y.columns):
        importances = tonly_model.estimators_[i].feature_importances_
    #     print(f"Feature importances for target {target}:")
        fi = pd.Series(importances,name=target,index=X_train.columns)
        fis.append(fi)
    #     for feat, imp in zip(X.columns, importances):
    #         print(f"{feat}: {imp:.4f}")
    #     print("-" * 30)

    imp_df = pd.concat(fis,axis=1)
    print(imp_df)


    ### PREDICT ON UNSEEN DATA FOR FUN! ###
    sa, sb = split_data(sub[sub['year']==setaside[0]],X_vars_cat,X_vars_num,y_vars)

    predictions_s = tonly_model.predict(sa)

    y_pred_s = pd.DataFrame(predictions_s)
    y_pred_s.columns = [col + "_pred" for col in sb.columns]
    y_pred_s.index = sb.index

    # y_pred range gut check
    print(pd.DataFrame(y_pred_s).sum(axis=1).describe())

    # And again, scale as necessary
    y_pred_s = y_pred_s.divide(y_pred_s.sum(axis=1),axis=0)

    # Evaluation
    mse = mean_squared_error(sb, y_pred_s, multioutput='raw_values')
    mae = mean_absolute_error(sb, y_pred_s, multioutput='raw_values')
    print("Mean squared errors for unseen data:",mse)
    print("Mean absolute errors for unseen data:",mae)

    # prepare data for later averaging and weighting
    tonly_preds_weights = y_pred_s.copy()
    tonly_preds_weights = tonly_preds_weights.merge(sub[sub['year']==setaside[0]]\
                                            [['sample_size','days_to_election']],\
                                           left_index=True,right_index=True,how='left')
    tonly_preds_weights['total_variance'] = tonly_preds_weights\
        .apply(lambda row: sum(calc_variance(row['sample_size'], row[col]) for col in \
                               [col+"_pred" for col in pshares]), axis=1)

    ### BACK TO SEEN DATA ###

    # Concatenate the two dataframes side-by-side, and add days_to_election
    comparison_df = pd.concat([y_test, y_pred], axis=1)
    print(comparison_df.mean())
    comparison_df = comparison_df.merge(pollse[pollse.index.isin(X_test.index)]['days_to_election'],\
                                           left_index=True,right_index=True,how='left')

    daily_tonly_mses = []
    daily_tonly_maes = []
    daily_tonly_avgs = []

    # create the daily errors and averages
    for day in range(comparison_df.days_to_election.min(),-1):
        row_mse = {}
        row_mae = {}
        row_avg = {}

        row_mse['day'] = day
        row_mae['day'] = day
        row_avg['day'] = day

        mse = mean_squared_error(comparison_df[comparison_df['days_to_election']<=day][pshares],\
                        comparison_df[comparison_df['days_to_election']<=day][[col+"_pred" for col in pshares]],\
                        multioutput='raw_values')
        mae = mean_absolute_error(comparison_df[comparison_df['days_to_election']<=day][pshares],\
                        comparison_df[comparison_df['days_to_election']<=day][[col+"_pred" for col in pshares]],\
                        multioutput='raw_values')

        # Calculate the inverse of the variance for weights
        weights = 1 / tonly_preds_weights[tonly_preds_weights['days_to_election']<=day]['total_variance']
        # Calculate the weighted average for each vote share column
        weighted_averages = {}
        for column in [col+"_pred" for col in pshares]:
            weighted_averages[column] = (tonly_preds_weights[tonly_preds_weights['days_to_election']<=day]\
                                         [column] * weights).sum() / weights.sum()

        mse_dict = dict(zip([col + "_mse" for col in pshares],mse))
        mae_dict = dict(zip([col + "_mae" for col in pshares],mae))

        row_mse.update(mse_dict)
        row_mae.update(mae_dict)
        row_avg.update(weighted_averages)

        daily_tonly_mses.append(row_mse)
        daily_tonly_maes.append(row_mae)
        daily_tonly_avgs.append(row_avg)

    daily_tonly_mse = pd.DataFrame(daily_tonly_mses)
    daily_tonly_mae = pd.DataFrame(daily_tonly_maes)
    daily_tonly_avg = pd.DataFrame(daily_tonly_avgs)
    
    return daily_tonly_mse, daily_tonly_mae, daily_tonly_avg
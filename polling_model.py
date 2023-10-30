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
from import_and_process import import_and_process
from fundamentals import generate_fundamentals

# disable warnings-- we are getting a lot

import warnings
warnings.filterwarnings('ignore')

### IMPORT ALL DATA ###

demo, map_dict, demo_r, cal, histe, histe_q1, res, res_ee, pollse, pshares, ppshares, ppshares_sc = import_and_process()
                    
### SET ASIDE SOME YEARS FOR VALIDATION ###

setaside = random.choices(res_ee.year.unique(),k=8)

### BRING IN PREDICTIONS AND ERRORS FROM FUNDAMENTALS MODEL ###

fund_prov_mse, fund_prov_mae, fund_comparison_df = generate_fundamentals(res_ee,pshares)

### BRING IN PREDICTIONS AND ERRORS FROM PROVINCE-SPECIFIC RANDOM FOREST MODELS ###



### BRING IN PREDICTIONS AND ERRORS FROM ALL-BUT-PROVINCE RANDOM FOREST MODELS ###

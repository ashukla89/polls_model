import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
import math
import random

from utils import split_data, quarterly_date

def import_and_process():

    ### IMPORT DEMOGRAPHICS ###
    
    demo = pd.read_csv('dataland_demographics.csv')

    # get a province-region mapping for whenever we need to add it
    mapping = demo[['region','province']].copy()
    mapping['r-p'] = mapping['region'] + "_" + mapping['province']
    map_dict = dict(zip(mapping['province'].tolist(),mapping['r-p'].tolist()))
    
    ### AUGMENT DEMOGRAPHICS WITH REGIONAL TOTALS ###

    demo_e = demo.copy()
    demo_e[['python_pop','cobolite_pop','javarian_pop']] = \
        demo[['python_pop_share','cobolite_pop_share','javarian_pop_share']].multiply(demo['population'],axis=0)

    demo_r = demo_e.groupby('region')[['python_pop','cobolite_pop','javarian_pop','electoral_college_votes']].sum()
    demo_r['population'] = demo_r[['python_pop','cobolite_pop','javarian_pop']].sum(axis=1)
    demo_r[['python_pop_share','cobolite_pop_share','javarian_pop_share']] = \
        demo_r[['python_pop','cobolite_pop','javarian_pop']].divide(demo_r['population'],axis=0)

    demo_r.loc['National'] = demo_r.sum(axis=0)
    demo_r.loc['National',['python_pop_share','cobolite_pop_share','javarian_pop_share']] = \
        demo_r.loc['National',['python_pop','cobolite_pop','javarian_pop']]\
            .divide(demo_r.loc['National','population']).tolist()
    demo_r = demo_r.reset_index().rename(columns={'region':'geography'})[['geography','population',\
                    'python_pop_share','cobolite_pop_share','javarian_pop_share','electoral_college_votes']]

    # add back the province data
    demo_r = pd.concat([demo_r,demo.rename(columns={'province':'geography'})[['geography','population',\
                    'python_pop_share','cobolite_pop_share','javarian_pop_share','electoral_college_votes']]],\
                       axis=0,ignore_index=True)

    ### IMPORT ELECTORAL CALENDAR ###
    
    # in practice, it seems that because the electoral calendar is so fixed, a good model
    # should learn the historical effects of conventions, etc.
    # unless I've missed something! Which is quite likely!
    cal = pd.read_csv('dataland_electoral_calendar.csv')
    cal.iloc[:,1:] = cal.iloc[:,1:].apply(pd.to_datetime)

    ### IMPORT HISTORICAL ECONOMIC DATA ###

    histe = pd.read_csv('dataland_economic_data_1984_2023.csv')

    # since we know elections always take place before 6/30, we can assume that only data as of 3/31 is public
    # let's filter to a dataset consisting only of those stats, and join to our provincial 
    # training vector on a "year" column that just corresponds to the year a 3/31 report was available
    histe_q1 = histe[histe['date'].str.contains('03-31')]
    histe_q1['year'] = histe['date'].str[:4].astype(int)
    histe_q1 = histe_q1[histe_q1['year']>=1984].drop('date',axis=1) # filter to 1984 and later too, for this purpose
    histe_q1

    ### IMPORT HISTORICAL ELECTION RESULTS ###
    
    res = pd.read_csv('dataland_election_results_1984_2023.csv')

    # define the names of the columns we'll call upon again and again
    pshares = ['pdal_share','ssp_share','dgm_share','cc_share']
    
    # expand with demographics
    res_ee = res.merge(demo,on=['province','region'],how='left')
    # and augment again with economic data
    res_ee = res_ee.merge(histe_q1,on='year',how='left')
    
    ### CALCULATE NATIONAL + REGIONAL ELECTION RESULTS, ASSUMING UNIFORM TURNOUT ##

    e = []

    res_me = res_ee[['year','province']+pshares].melt(id_vars=['year','province'],value_vars=pshares)
    res_me.columns = ['year','province','party','share']

    for party in pshares:
        for province in res_me.province.unique():
            sub = res_me[(res_me['party']==party)&(res_me['province']==province)]
            sub['pdiff'] = sub['share'].diff().abs()
            e.append(sub)

    diffs = pd.concat(e,axis=0)

    # add population
    diffs = diffs.merge(demo[['province','region','population']],on='province',how='left')
    # compute population-weighted vote (for purposes of calculating national swing)
    diffs['pvotes'] = diffs['share'] * diffs['population']

    # national results as per above
    natshares = diffs.groupby(['year','party']).pvotes.sum() / demo.population.sum()
    natshares = natshares.reset_index()
    natshares['geography'] = 'National'
    natshares.columns = ['year','party','share','geography']

    # regional results as per above

    regpops = demo.groupby('region')['population'].sum().reset_index()

    regshares = diffs.groupby(['year','region','party']).pvotes.sum().reset_index()
    regshares = regshares.merge(regpops[['region','population']],how='left')
    regshares['share'] = regshares['pvotes'] / regshares['population']
    regshares = regshares[['year','region','party','share']].rename(columns={'region':'geography'})

    # aggregate and widen
    aggshares = pd.concat([natshares,regshares],ignore_index=True)
    aggshares = aggshares.pivot(index=['year','geography'],columns='party',values='share').reset_index()

    # and narrow down provincial shares too

    provshares = res_e[['year','province','cc_share','dgm_share','pdal_share','ssp_share']]\
        .rename(columns={'province':'geography'})

    # now put them together

    aggshares = pd.concat([aggshares,provshares],ignore_index=True)
    
    ### IMPORT POLLS AND ADD A BUNCH OF VARIABLES TO THE VECTOR ###
    
    ppshares = ['pdal_poll_share','ssp_poll_share','dgm_poll_share','cc_poll_share']

    polls = pd.read_csv('updated_dataland_polls_1984_2023.csv')
    polls = polls.merge(aggshares,on=['year','geography'],how='left')
    
    # add ranked margin differences
    pollse = polls.copy()

    # add columns adjusting polls to remove undecideds
    scaled_polls = pollse[ppshares].multiply(pollse[ppshares].sum(axis=1)**-1,axis=0)
    ppshares_sc = ['pdal_poll_scaled','ssp_poll_scaled','dgm_poll_scaled','cc_poll_scaled']
    scaled_polls.columns = ppshares_sc
    pollse = pd.concat([pollse,scaled_polls],axis=1)

    # add party-wise share differences
    pollse['pdal_diff'] = pollse['pdal_poll_scaled'] - pollse['pdal_share']
    pollse['ssp_diff'] = pollse['ssp_poll_scaled'] - pollse['ssp_share']
    pollse['dgm_diff'] = pollse['dgm_poll_scaled'] - pollse['dgm_share']
    pollse['cc_diff'] = pollse['cc_poll_scaled'] - pollse['cc_share']
    pdiffs = ['pdal_diff','ssp_diff','dgm_diff','cc_diff']

    # add calendar
    pollse = pollse.merge(cal,left_on='year',right_on='election_cycle',how='left')

    # transform date columns to datetime
    pollse[[col for col in pollse.columns if 'date' in col]+['election_day']] = \
        pollse[[col for col in pollse.columns if 'date' in col]+['election_day']].apply(pd.to_datetime)

    # calculate days to election
    pollse['days_to_election'] = (pollse['date_conducted'] - pollse['election_day']).dt.days

    # add demographic data
    pollse = pollse.merge(demo_r,on='geography',how='left')

    # add economic data for the latest available quarter
    pollse = pollse.merge(histe,left_on=pollse['date_conducted'].apply(quarterly_date),\
                          right_on=pd.to_datetime(histe['date']),how='left').drop('key_0',axis=1)

    # add incumbency
    pollse = pollse.merge(res[['year','party_in_power']].drop_duplicates(),on='year',how='left')
    
    ### IMPORT POLLING AND ECONOMIC SCENARIOS ###
    
    scenarios = pd.read_csv('dataland_polls_2024_scenarios.csv')
    e_scenarios = pd.read_csv('dataland_economic_data_2024_scenarios.csv')

    return demo, map_dict, demo_r, cal, histe, histe_q1, res, res_ee, pollse, pshares, ppshares, ppshares_sc, scenarios, e_scenarios
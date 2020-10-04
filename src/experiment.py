import copy
import os
import pandas as pd
import numpy as np
import networkx as nx
import sklearn.linear_model as skl_lm
from scipy.stats import pearsonr
from scipy.stats import skew
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVR
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from itertools import chain, combinations
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
home = str(Path.home())
import sys
sys.path.insert(0,"%s/rankability_toolbox_dev"%home)
import pyrankability
from pyrankability.construct import *
sys.path.insert(0,"%s/sensitivity_study/src"%home)
from sensitivity_tests import *
from utilities import *
from base import *

# Somehow we need to figure out how to checkpoint intermediate results

standard_cols = ["team1_name", "team2_name", "team1_score", "team2_score", "date"]

# Function to read raw pairwise data into dataframe with standardized col names
def read_raw_pairwise(filepath, col_mapping):
    # filepath: where to find csv file
    # col_mapping: dictionary that maps from standard col name to csv's col name
    #  example: {"team1_name": "home_team_name"}
    # also, csv should be ordered by "date" column if exists and drop date
    # returns sorted dataframe of pairwise comparisons
    
    df = pd.read_csv(filepath)
    
    # Rename columns provided
    for standard_col, custom_col in col_mapping:
        if standard_col != custom_col:
            df[standard_col] = df[custom_col]
            df.drop(custom_col, axis=1, inplace=True)
    
    # Sort by date and drop date
    df = df.sort_values(by='date').drop('date',axis=1)
    
    # Drop extra columns
    for col in df.columns:
        if col not in col_mapping:
            df.drop(col, axis=1, inplace=True)
    
    return df

def construct_support_matrix(pairwise_df,
                             fraction,
                             direct_thres = 2,
                             spread_thres = 2,
                             weight_indirect = 0.5):
    # pairwise_df: pairwise dataframe with rows in relevant order
    # must include certain columns, see usage below
    # fraction: percent of data to use from beginning
    # direct_thres: the minimum difference in scores for direct to be counted
    # spread thres: the minimum difference in score differences for indirect to be counted
    # weight_indirect: the weighting of indirect evidence relative to direct
    # returns support matrix for the relevant teams
    #    (full matrix cut down to madness_teams for example)
    
    madness_teams = np.unique(list(pairwise_df.team1_name.loc[pairwise_df.team1_madness == 1])
                              + list(pairwise_df.team2_name.loc[pairwise_df.team2_madness == 1]))
    game_list = list(pairwise_df.index)
    
    upper = int(len(pairwise_df)*fraction)
    game_df_sample = pairwise_df.iloc[:upper,:]

    map_func = lambda linked:
        support_map_vectorized_direct_indirect_weighted(
                                                        linked,
                                                        direct_thres=direct_thres,
                                                        spread_thres=spread_thres,
                                                        weight_indirect=weight_indirect)
    return V_count_vectorized(game_df_sample,map_func).loc[madness_teams,madness_teams]


feature_creation_list = [
    'Year',
    '# X* frac',
    'k',
    '# X* frac top 40',
    'kendall_w',
    'p_lowerbound',
    'max_L2_dist',
    'mean_L2_dist',
    'min_tau',
    'mean_tau',
    'max_eigenval',
    'min_eigenval',
    'max_eigenval_xstar',
    'min_eigenval_xstar',
    'Pair'
]


def get_features_from_support(support):
    # get all of the features from the support (including solving LOP for details first)
    # returns a pd.Series of all features for *this* single support matrix
    # the support matrix for a pair for a given year

    # eigens of the support matrix
    vals, vecs = np.linalg.eig(support.fillna(0.0).values)
    determinant = np.prod(vals)
    trace = np.sum(vals)
    max_eigenval = np.real(np.max(vals))
    min_eigenval = np.real(np.min(vals))
    dsGraph = nx.from_numpy_matrix(support.fillna(0.0).values)
    
    rresults = rankability_results.iloc[c,:]
    k = rresults['k']
    details = df_details[c]
    x = pd.DataFrame(details['x'],index=support.index,columns=support.columns)
    r = x.sum(axis=0)
    order = np.argsort(r)
    xstar = x.iloc[order,:].iloc[:,order]
    xstar.loc[:,:] = pyrankability.common.threshold_x(xstar.values)
    
    vals, vecs = np.linalg.eig(xstar.to_numpy())
    det_xstar = np.real(np.prod(vals))
    max_eigenval_xstar = np.real(np.max(vals))
    min_eigenval_xstar = np.real(np.min(vals))
    
    inxs = np.triu_indices(len(xstar),k=1)
    xstar_upper = xstar.values[inxs[0],inxs[1]]
    nfrac_upper = sum((xstar_upper > 0) & (xstar_upper < 1))
    flat_frac = ((xstar > 0) & (xstar < 1)).sum(axis=0)
    nfrac_top_40 = flat_frac.iloc[:40].sum()
    entry_data = [
        year,
        nfrac_upper*2,
        k,
        nfrac_top_40,
        rresults["kendall_w"],
        rresults["p_lowerbound"],
        rresults["max_L2_dist"],
        rresults["mean_L2_dist"],
        rresults["min_tau"],
        rresults["mean_tau"],
        max_eigenval, 
        min_eigenval,
        max_eigenval_xstar,
        min_eigenval_xstar,
        pair
    ]
    entry = pd.Series(entry_data,feature_creation_list)
    return entry


def get_target_stability(support1, support1, rankingMethod, corrMethod):
    # Measure the correlation between rankings of support1 and support2
    # Maybe at this point consider checkpointing the rankings as well
    # return the correlation (single float)
    rankingMethod = rankingMethod()
    ranking1 = rankingMethod.rank(support1.fillna(0).values)
    ranking2 = rankingMethod.rank(support2.fillna(0).values)
    # rankings[year].append((ranking1,ranking2))
    # ranking1, ranking2 = rankings[year][i]
    corr = corrMethod(ranking1,ranking2)
    return corr


# iterate over this and call eval_models
model_list = [{"model":DummyRegressor(), "param_grid": {}}]


def eval_models(features, targets):
    # Train and evaluate different models on this regression task
    # Return a list of best performances per model from model_list
    # [{"modelname": "DummyRegressor", "Performance":PerformanceObject}]
    exhaustive_feat_select = list(chain.from_iterable(combinations(list(range(len(features.columns))), r) for r in range(len(features.columns))))
    # only 10 feature subsets (out of 2^n) for debug purposes
    best_score = np.Inf
    best_features = None
    for ps in tqdm(exhaustive_feat_select, ascii=True):
        features = features.iloc[:, list(ps)]
        grid = GridSearchCV(model,param_grid,refit=True,verbose=0, cv=3, iid=True, n_jobs=-1)
        exhaustive[ps] = np.mean(np.abs(cross_val_score(grid, features, y, scoring="neg_mean_absolute_error", cv=3, n_jobs=1)))
        if exhaustive[ps] < best_score:
            best_score = exhaustive[ps]
            best_features = ps
    
    return ({"MAE": best_score, "best_feature_subset": [features.columns[f] for f in best_features]}, exhaustive)
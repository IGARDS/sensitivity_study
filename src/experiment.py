import copy
import os
import pandas as pd
import numpy as np
import networkx as nx
import sklearn.linear_model as skl_lm
from sklearn.model_selection import cross_val_score, LeaveOneOut, GridSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from itertools import chain, combinations
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
home = str(Path.home())
import sys
rankability_path = "%s/rankability_toolbox_dev"%home
if rankability_path not in sys.path:
    sys.path.insert(0,rankability_path)
import pyrankability
from pyrankability.construct import *
sensitivity_study_path = "%s/sensitivity_study/src"%home
if sensitivity_study_path not in sys.path:
    sys.path.insert(0,sensitivity_study_path)
from sensitivity_tests import *
import utilities


# Somehow we need to figure out how to checkpoint intermediate results
standard_columns = ["team1_name",
                    "team2_name",
                    "team1_score",
                    "team2_score",
                    "date",
                    "team1_select",
                    "team2_select"]

# Function to read raw pairwise data into dataframe with standardized col names
def read_raw_pairwise(filepath, col_mapping):
    # filepath: where to find csv file
    # col_mapping: dictionary that maps from standard col name to csv's col name
    #  example: {"team1_name": "home_team_name"}
    # also, csv should be ordered by "date" column if exists and drop date
    # returns sorted dataframe of pairwise comparisons
    
    df = pd.read_csv(filepath)
    
    # Rename columns provided
    for standard_col in col_mapping.keys():
        custom_col = col_mapping[standard_col]
        if standard_col != custom_col:
            df[standard_col] = df[custom_col]
            df.drop(custom_col, axis=1, inplace=True)
    
    # Sort by date and drop date
    df = df.sort_values(by='date').drop('date',axis=1)
    
    # Drop extra columns
    for col in df.columns:
        if col not in standard_columns:
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
    
    selected_teams = np.unique(list(pairwise_df.team1_name.loc[pairwise_df.team1_select == 1])
                              + list(pairwise_df.team2_name.loc[pairwise_df.team2_select == 1]))
    game_list = list(pairwise_df.index)
    
    upper = int(len(pairwise_df)*fraction)
    game_df_sample = pairwise_df.iloc[:upper,:].drop(["team1_select", "team2_select"], axis=1)
    # print(game_df_sample)
    game_df_sample.reset_index(drop=True, inplace=True)

    def map_func(linked):
        return support_map_vectorized_direct_indirect_weighted(linked,
                                                               direct_thres=direct_thres,
                                                               spread_thres=spread_thres,
                                                               weight_indirect=weight_indirect)
    v = V_count_vectorized(game_df_sample,map_func)
    return v.reindex(columns=selected_teams).reindex(index=selected_teams).fillna(0)


def get_features_from_support(support, n_restarts):
    # get all of the features from the support (including solving LOP for details first)
    # returns a pd.Series of all features for *this* single support matrix
    # the support matrix for a pair for a given year
    features = {}
    support = support.fillna(0.0)
    
    # eigens of the support matrix
    eig_vals, _ = np.linalg.eig(support.values)
    features['max_eigenval_support'] = np.real(np.max(eig_vals))
    features['min_eigenval_support'] = np.real(np.min(eig_vals))
    
    features["k"], details = pyrankability.rank.solve(support,
                                                      method='lop',
                                                      num_random_restarts=n_restarts,
                                                      lazy=False,
                                                      cont=True)
    
    for key,val in get_P_stats(details["P"]).items():
        if key in features:
            raise ValueError("Feature Column collision! Check feature names!")
        else:
            features[key] = val
    
    x = pd.DataFrame(details['x'],index=support.index,columns=support.columns)
    r = x.sum(axis=0)
    order = np.argsort(r)
    xstar = x.iloc[order,:].iloc[:,order]
    xstar.loc[:,:] = pyrankability.common.threshold_x(xstar.values)
    
    # eigens of the X* matrix
    eig_vals, _ = np.linalg.eig(xstar.values)
    features['max_eigenval_xstar'] = np.real(np.max(eig_vals))
    features['min_eigenval_xstar'] = np.real(np.min(eig_vals))

    flat_frac = ((xstar > 0) & (xstar < 1)).sum(axis=0)
    features['# X* frac'] = flat_frac.sum()
    features['# X* frac top 40'] = flat_frac.iloc[:40].sum()

    return pd.Series(features)


def get_target_stability(support1, support2, rankingMethod, corrMethod):
    # Measure the correlation between rankings of support1 and support2
    # Maybe at this point consider checkpointing the rankings as well
    # return the correlation (single float)
    ranking1 = rankingMethod.rank(support1.fillna(0).values)
    ranking2 = rankingMethod.rank(support2.fillna(0).values)
    # rankings[year].append((ranking1,ranking2))
    # ranking1, ranking2 = rankings[year][i]
    corr = corrMethod(ranking1,ranking2)
    return corr


def eval_models(features, targets, model_list):
    # Train and evaluate different models on this regression task
    # Return a list of best performances per model from model_list
    # {"DummyRegressor": Performance, "LinearRegressor":Performance}
    reaults_by_model_type = {}
    exhaustive_feat_select = list(chain.from_iterable(combinations(list(range(len(features.columns))), r) for r in range(1, len(features.columns))))
    
    # For each model type, produce a dict with the best mae, best feature subset, and list of exhaustive results
    for model_dict in model_list:
        model = model_dict["model"]
        param_grid = model_dict["param_grid"]
        best_score = np.Inf
        best_features = None
        exhaustive = {}
        for ps in tqdm(exhaustive_feat_select):
            feature_subset = features.iloc[:, list(ps)]
            grid = GridSearchCV(model,
                                param_grid,
                                refit=True,
                                verbose=0,
                                cv=3,
                                iid=True,
                                n_jobs=-1)
            exhaustive[ps] = np.mean(np.abs(cross_val_score(grid,
                                                            feature_subset,
                                                            targets,
                                                            scoring="neg_mean_absolute_error",
                                                            cv=3,
                                                            n_jobs=1)))
            if exhaustive[ps] < best_score:
                best_score = exhaustive[ps]
                best_features = ps

        reaults_by_model_type[model.__class__.__name__] = {
            "MAE": best_score,
            "best_feature_subset": [features.columns[f] for f in best_features],
            "all_results": exhaustive
        }
    return reaults_by_model_type


def main():
    years = ["2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009",
             "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018"]
    
    config = {
        "col_mapping": {
            "team1_name":"team1_name",
            "team1_score":"points1",
            "team2_name":"team2_name",
            "team2_score":"points2",
            "team1_select": "team1_madness",
            "team2_select": "team2_madness",
            "date":"date"
        },
        "rankingMethod": MasseyRankingAlgorithm(),
        "correlationMethod":utilities.kendall_tau,
        "pair": (.75, 1.0),
        "raw_filepaths": ["{}/sensitivity_study/data/MarchMadnessDataFrames/march_madness_{}.csv".format(home,yr) for yr in years],
        "model_list": [{"model":DummyRegressor(), "param_grid": {}},
                       {"model":LinearRegression(), "param_grid": {'fit_intercept': [True, False]}}]
    }
    
    games = {fp: read_raw_pairwise(fp, config["col_mapping"]) for fp in config["raw_filepaths"]}
    target_list = []
    support_matricies = {}
    feature_df_list = []
    # For each raw file (equivalent to a season / tournament / single scenario)
    # get feature vector and target scalar
    for fp in tqdm(games.keys()):
        support_matricies[fp] = {}
        # For both fractions in pair, construct the support matrices
        for frac in config["pair"]:
            support_matricies[fp][frac] = construct_support_matrix(games[fp],
                                                                   frac,
                                                                   direct_thres=2,
                                                                   spread_thres=2,
                                                                   weight_indirect=0.5)
        # Now from the less-informed support, get features
        feature_df_list.append(get_features_from_support(support_matricies[fp][config["pair"][0]], config["n_restarts"]))
        feature_df_list[-1].name = fp
        
        # From both less-informed and more informed supports, get target (rank correlations)
        target_list.append(get_target_stability(support_matricies[fp][config["pair"][0]],
                                                support_matricies[fp][config["pair"][1]],
                                                config["rankingMethod"],
                                                config["correlationMethod"]))
    # good spot for a checkpoint: support_matricies
    features = pd.DataFrame(feature_df_list)
    targets = pd.Series(target_list,index=features.index)
    # good spot for a checkpoint: features, targets
    results_dict = eval_models(features, targets, config["model_list"])


if __name__ == "__main__":
    main()

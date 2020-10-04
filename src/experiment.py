import copy
import os
import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import pearsonr
from scipy.stats import skew
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
home = str(Path.home())
import sys
sys.path.insert(0,"%s/rankability_toolbox_dev"%home)
import pyrankability
sys.path.insert(0,"%s/sensitivity_study/src"%home)
from sensitivity_tests import *
from utilities import *
from base import *


# Somehow we need to figure out how to checkpoint intermediate results

# Function to read raw pairwise data into dataframe with standardized col names
def read_raw_pairwise(filename, col_mapping):
    # filename: where to find csv file
    # col_mapping: dictionary that maps from standard col name to csv's col name
    #  example: {"team1_name": "home_team_name"}
    # also, csv should be ordered by "date" column if exists and drop date
    # returns sorted dataframe of pairwise comparisons
    pass

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
    
    madness_teams = np.unique(list(pairwise_df.team1_name.loc[pairwise_df.team1_madness == 1]) + list(pairwise_df.team2_name.loc[pairwise_df.team2_madness == 1]))
    game_list = list(pairwise_df.index)
    
    upper = int(len(pairwise_df)*fraction)
    game_df_sample = pairwise_df.iloc[:upper,:]

    map_func = lambda linked: pyrankability.construct.support_map_vectorized_direct_indirect_weighted(linked,direct_thres=direct_thres,spread_thres=spread_thres,weight_indirect=weight_indirect)
    return pyrankability.construct.V_count_vectorized(game_df_sample,map_func).loc[madness_teams,madness_teams]


feature_list = []

def get_features_from_support(support):
    # get all of the features from the support (including solving LOP for details first)
    # returns a pd.Series of all features for *this* single support matrix
    pass

def get_target_stability(support1, support2, rankingMethod, corrMethod):
    # Measure the correlation between rankings of support1 and support2
    # Maybe at this point consider checkpointing the rankings as well
    # return the correlation (single float)
    pass

model_list = [{"model":DummyRegressor(), "param_grid": {}}]

def evaluate_models(features, targets):
    # Train and evaluate different models on this regression task
    # Return a list of best performances per model from model_list
    # [{"modelname": "DummyRegressor", "Performance":PerformanceObject}]
    pass
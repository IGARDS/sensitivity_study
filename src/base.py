import pandas as pd
import numpy as np
from datetime import timedelta

selectionSundayList = ['03/10/2002','03/16/2003','03/14/2004','03/13/2005','03/12/2006','03/11/2007','03/16/2008',
                       '03/15/2009','03/14/2010','03/13/2011','03/11/2012',
                       '03/17/2013','03/16/2014','03/15/2015','03/13/2016','03/12/2017','03/11/2018', '3/17/2019']

days_to_subtract=7
d = timedelta(days=days_to_subtract)

# Just a consistent way of processing files. Ignore the fact that the local variables say 2014
def read_data(teams_file,games_file,madness_teams_file):
    teams_2014 = pd.read_csv(teams_file,header=None)
    teams_2014.columns=["number","name"]
    games_2014 = pd.read_csv(games_file,header=None)
    games_2014.columns = ["notsure1","date","team1","H_A_N1","points1","team2","H_A_N2","points2"]
    team1_names = teams_2014.copy()
    team1_names.columns = ["team1","team1_name"]
    team1_names.set_index('team1',inplace=True)
    games_2014 = games_2014.set_index("team1").join(team1_names,how='inner').reset_index()
    team2_names = teams_2014.copy()
    team2_names.columns = ["team2","team2_name"]
    team2_names.set_index('team2',inplace=True)
    games_2014 = games_2014.set_index("team2").join(team2_names,how='inner').reset_index()
    games_2014["date"] = pd.to_datetime(games_2014["date"],format="%Y%m%d")
    games_2014["team1_name"] = games_2014["team1_name"].str.replace(" ","")
    games_2014["team2_name"] = games_2014["team2_name"].str.replace(" ","")
    prev_len = len(games_2014)
    madness_teams = pd.read_csv(madness_teams_file,header=None)
    madness_teams.columns=["name"]
    games_2014["team1_madness"] = 0
    games_2014["team2_madness"] = 0
    mask = games_2014.team1_name.isin(list(madness_teams["name"]))
    games_2014.loc[mask,"team1_madness"] = 1
    mask = games_2014.team2_name.isin(list(madness_teams["name"]))
    games_2014.loc[mask,"team2_madness"] = 1
    games_2014.reset_index()
    for selection_sunday in selectionSundayList:
        games = games_2014.loc[games_2014["date"] <= pd.to_datetime(selection_sunday,format="%m/%d/%Y")-d]
        if len(games) > 0:
            break
    return games

def support_map_vectorized_direct_indirect_weighted(linked,direct_thres=1,spread_thres=0,weight_indirect=0.5,verbose=False):
    # columns
    # 'team_j', 'team_i_name', 'team_i_score', 'team_i_H_A_N',
    # 'team_j_i_score', 'team_j_i_H_A_N', 'game_i_j', 'team_k_name',
    # 'team_k_score', 'team_k_H_A_N', 'team_j_k_score', 'team_j_k_H_A_N',
    # 'game_k_j'
    linked["direct"] = linked["team_i_name"] == linked["team_k_name"]
    # | (linked["team_i_name"] == linked["team_j_k_name"]) | (linked["team_k_name"] == linked["team_j_k_name"])
    for_index1 = linked[["team_i_name","team_k_name"]].copy()
    for_index1.loc[linked["direct"]] = linked.loc[linked["direct"],["team_i_name","team_j_name"]]
    for_index1.columns = ["team1","team2"]
    for_index2 = linked[["team_k_name","team_i_name"]].copy()
    for_index2.loc[linked["direct"]] = linked.loc[linked["direct"],["team_j_name","team_i_name"]]
    for_index2.columns = ["team1","team2"]
    index_ik = pd.MultiIndex.from_frame(for_index1,sortorder=0)
    index_ki = pd.MultiIndex.from_frame(for_index2,sortorder=0)
    
    #######################################
    # part to modify
    # direct
    d_ik = linked['team_i_score'] - linked['team_j_i_score']
    support_ik = (linked["direct"] & (d_ik > direct_thres)).astype(int)
    support_ki = (linked["direct"] & (d_ik < -direct_thres)).astype(int)

    # indirect
    d_ij = linked["team_i_score"] - linked["team_j_i_score"]
    d_kj = linked["team_k_score"] - linked["team_j_k_score"]
    
    # always a positive and it captures that if i beat j by 5 points and k beat j by 2 points then this spread is 3
    spread = np.abs(d_ij - d_kj) 
    
    support_ik += weight_indirect*((~linked["direct"]) & (d_ij > 0) & (d_kj < 0) & (spread > spread_thres)).astype(int)
    
    support_ki += weight_indirect*((~linked["direct"]) & (d_kj > 0) & (d_ij < 0) & (spread > spread_thres)).astype(int)
    
    # end part to modify
    #######################################    
    linked["support_ik"]=support_ik
    linked["index_ik"]=index_ik
    linked["support_ki"]=support_ki
    linked["index_ki"]=index_ki
        
    if verbose:
        print('Direct')
        print("Total:",sum(linked["direct"] & (linked["support_ik"]>0)) + sum(linked["direct"] & (linked["support_ki"]>0)),
              "ik:",sum(linked["direct"] & (linked["support_ik"]>0)), 
              "ki:",sum(linked["direct"] & (linked["support_ki"]>0)))
        print('Indirect')
        print("Total:",sum((~linked["direct"]) & (linked["support_ik"]>0)) + sum(~linked["direct"] & (linked["support_ki"]>0)),
              "ik:",sum((~linked["direct"]) & (linked["support_ik"]>0)), 
              "ki:",sum(~linked["direct"] & (linked["support_ki"]>0)))
    
    #indices_ik = linked.index[(linked["support_ik"] > linked["support_ki"]) & ~linked['direct']]    
    #indices_ki = linked.index[(linked["support_ik"] > linked["support_ki"]) & ~linked['direct']] 
    #linked.loc[~linked['direct']] = 0
    #linked.loc[indices_ik] = 0.5*(linked.loc[indices_ik]["support_ik"] - linked.loc[indices_ik]["support_ki"])
    #linked.loc[indices_ik] = 0.5*(linked.loc[indices_ik]["support_ik"] - linked.loc[indices_ik]["support_ki"])
    
    ret1 = linked.set_index(index_ik)["support_ik"]
    ret2 = linked.set_index(index_ki)["support_ki"]
    ret = ret1.append(ret2)
    ret = ret.groupby(level=[0,1]).sum()
    return ret


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

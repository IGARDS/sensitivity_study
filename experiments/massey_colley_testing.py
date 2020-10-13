year='2002'
madness_teams = np.unique(list(games[year].team1_name.loc[games[year].team1_madness == 1]) + list(games[year].team2_name.loc[games[year].team2_madness == 1]))
game_list = list(games[year].index)

game_df = pd.DataFrame({"team1_name":games[year]['team1_name'],
                        "team1_score":games[year]['points1'],
                        "team1_H_A_N": games[year]['H_A_N1'],
                        "team2_name":games[year]['team2_name'],
                        "team2_score":games[year]['points2'],
                        "team2_H_A_N": games[year]['H_A_N1'],
                        "date": games[year]['date']
                       }).sort_values(by='date').drop('date',axis=1)
upper = int(len(game_df)*frac)
game_df_sample = game_df.iloc[:upper,:]
map_func = lambda linked: pyrankability.construct.colley_matrices(linked,direct_thres=0,spread_thres=0,weight_indirect=0)
colleyMatrix_test,b_test = pyrankability.construct.map_vectorized(game_df_sample,map_func)

map_func = lambda linked: pyrankability.construct.massey_matrices(linked,direct_thres=0,spread_thres=0,weight_indirect=0)
masseyMatrix_test,massey_b_test = pyrankability.construct.map_vectorized(game_df_sample,map_func)

## massey
import numpy as np
from math import ceil 

teams = pd.Series(list(game_df_sample.team1_name)+list(game_df_sample.team2_name)).unique()

masseyMatrix = pd.DataFrame(np.zeros((len(teams),len(teams))),columns=teams,index=teams)

b = pd.Series(np.zeros((len(teams),)),index=teams)

for i in game_df_sample.index:
    team1ID = game_df_sample.loc[i,"team1_name"]
    team1Score = game_df_sample.loc[i,"team1_score"]

    team2ID = game_df_sample.loc[i,"team2_name"]
    team2Score = game_df_sample.loc[i,"team2_score"]
    
    # Update the Colley matrix and RHS
    if team1Score == team2Score:
        gameWeight = 0 # if you don'include ties and there is a tie, you exclude the game
    else: 
        gameWeight = 1
    
    masseyMatrix.loc[team1ID, team2ID] -= gameWeight
    masseyMatrix.loc[team2ID, team1ID] -= gameWeight

    masseyMatrix.loc[team1ID, team1ID] += gameWeight
    masseyMatrix.loc[team2ID, team2ID] += gameWeight
    
    pointDifferential = gameWeight*abs(team1Score - team2Score)

    if team1Score > team2Score:
        b.loc[team1ID] += pointDifferential
        b.loc[team2ID] -= pointDifferential
    elif team1Score < team2Score:
        b.loc[team1ID] -= pointDifferential
        b.loc[team2ID] += pointDifferential
        
# replace last row with ones and 0 on RHS
masseyMatrix.values[-1,:] = np.ones((1,len(teams)))
b[-1] = 0

## colley
import numpy as np
from math import ceil 

teams = pd.Series(list(game_df_sample.team1_name)+list(game_df_sample.team2_name)).unique()

colleyMatrix = pd.DataFrame(np.zeros((len(teams),len(teams))),columns=teams,index=teams)

b = pd.Series(np.zeros((len(teams),)),index=teams)

for i in game_df_sample.index:
    team1ID = game_df_sample.loc[i,"team1_name"]
    team1Score = game_df_sample.loc[i,"team1_score"]

    team2ID = game_df_sample.loc[i,"team2_name"]
    team2Score = game_df_sample.loc[i,"team2_score"]
    
    # Update the Colley matrix and RHS
    if team1Score == team2Score:
        gameWeight = 0 # if you don'include ties and there is a tie, you exclude the game
    else: 
        gameWeight = 1
    
    colleyMatrix.loc[team1ID, team2ID] -= gameWeight
    colleyMatrix.loc[team2ID, team1ID] -= gameWeight

    colleyMatrix.loc[team1ID, team1ID] += gameWeight
    colleyMatrix.loc[team2ID, team2ID] += gameWeight

    if team1Score > team2Score:
        b.loc[team1ID] += 1/2
        b.loc[team2ID] -= 1/2
    elif team1Score < team2Score:
        b.loc[team1ID] -= 1/2
        b.loc[team2ID] += 1/2
    else:  # whether including ties or not the RHS is not affected
        b.loc[team1ID] += 0; 
        b.loc[team2ID] += 0; 
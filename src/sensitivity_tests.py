import numpy as np
import sys
import random
sys.path.insert(0, "/disk/home/ubuntu/rankability_toolbox/")
from pyrankability import common as prcommon, lop, bilp, pruning_paper

class ProblemInstance:
    
    def __init__(self, dataSource, noiseGenerator):
        self.dataSource = dataSource
        self.noiseGenerator = noiseGenerator
    
    def init_D(self):
        # TODO: get data from datasource
        pass
    
    def get_ranking(self):
        # returns ranking and tau
        pass
    
    def get_sensitivity(self, rankingAlg):
        # TODO @Ethan: get perfect ranking
        perfectRanking = rankingAlg.rank(self.dataSource)
        # TODO: for N
            # TODO @Marisa: generate and apply noise
            # TODO @Jackson: run hillside count to get k,p
            #actualRanking = rankingAlg.rank(Marisa's Matrix)
            # TODO @Ethan: get ranking of distorted D
        # TODO: get perfect ranking
        # TODO: for N
            # TODO: generate and apply noise
            # TODO: get ranking of distorted D
            # TODO: calculate tau
        # TODO: produce summary of tau
        pass

class NoiseGenerator:
    def apply_noise(self, D):
        # Child classes should return numpy array of D_tilde
        raise NotImplemented("Don't use the generic NoiseGenerator class")

class PercentageFlipNoise:
    
    def __init__(self, noisePercentage):
        self.noisePercentage = noisePercentage
    
    def apply_noise(self, D):
        n = len(D)
        num_flips = (np.square(n) - n) * self.noisePercentage
        unique_elems = set()
        for flip in range(int(num_flips)):
            i, j = random.sample(range(n), 2)
            while ((i, j) in unique_elems): i, j = random.sample(range(n), 2)
            unique_elems.add((i, j))
            D[i][j] = 1 - D[i][j]

class DataSource:
    def init_D(self):
        # Child classes should return numpy array
        raise NotImplemented("Don't use the generic DataSource class")
        
class RankingAlgorithm:
        
    def rank(D):
        # Child classes should return numpy array of ranking vector
        raise NotImplemented("Don't use the generic RankingAlgorithm class")
        
class LOPRankingAlgorithm(RankingAlgorithm):
    def rank(self, D):
        # Child classes should return numpy array of ranking vector
        #lop.bilp(D) returns an odd dominance graph. Question for Paul I suppose.
        #Also, does pyrankability even have an LOP ranking alg? Question two.
        #print(D.shape)
        return np.arange(1,D.shape[1]) #lop.bilp(D)
    
class ColleyRankingAlgorithm(RankingAlgorithm):
    def rank(self, D):
        # Child classes should return numpy array of ranking vector
        #need to convert dominance graph to colley format
        wins = [sum(D[i]) for i in range(0,D.shape[0])]
        losses = [sum(np.transpose(D)[i]) for i in range(0,D.shape[0])]
        totalevents = [wins[i] + losses[i] for i in range(0,D.shape[0])]
        b = [1 + (wins[i] - losses[i])/2 for i in range(0,D.shape[0])]
        C = np.zeros(D.shape)
        for i in range(0,D.shape[0]):
            C[i][i] = 2 + totalevents[i]
        for x in range(D.shape[0]):
            for y in range(D.shape[1]):
                if x != y:
                    C[x][y] = (D[x][y] + D[y][x]) * -1
        r = np.linalg.solve(C, b)
        r = sorted([(r[i - 1], i) for i in range(1, D.shape[0] + 1)])
        retvec = [r[i][1] for i in range(len(r))]
        return retvec
    
class MasseyRankingAlgorithm(RankingAlgorithm):
    def rank(self, D):
        # Child classes should return numpy array of ranking vector
        #need to convert dominance graph to colley format
        wins = [sum(D[i]) for i in range(0,D.shape[0])]
        losses = [sum(np.transpose(D)[i]) for i in range(0,D.shape[0])]
        totalevents = [wins[i] + losses[i] for i in range(0,D.shape[0])]
        b = [1 + (wins[i] - losses[i])/2 for i in range(0,D.shape[0])]
        C = np.zeros(D.shape)
        for i in range(0,D.shape[0]):
            C[i][i] = totalevents[i]
        for x in range(D.shape[0]):
            for y in range(D.shape[1]):
                if x != y:
                    C[x][y] = (D[x][y] + D[y][x]) * -1
        C[D.shape[0] - 1] = np.ones(D.shape[0])
        b[D.shape[0] - 1] = 0
        r = np.linalg.solve(C, b)
        r = sorted([(r[i - 1], i) for i in range(1, D.shape[0] + 1)])
        retvec = [r[i][1] for i in range(len(r))]
        return retvec
    
class MarkovChainRankingAlgorithm
    
def main():
    cra = ColleyRankingAlgorithm()
    mra = MasseyRankingAlgorithm()
    fivefive = np.zeros((5,5))
    est = np.array([[0,1,0,0,0],
                    [0,0,1,0,0],
                    [0,0,0,1,0],
                    [0,0,0,0,1],
                    [0,0,0,0,0]])
    fivegood = np.array([[0,1,0,0,0],
                        [0,0,1,0,0],
                        [0,0,0,1,0],
                        [0,0,0,0,1],
                        [1,1,1,1,0]])
    completedominance = np.array([[0,1,1,1,1],
                                [0,0,1,1,1],
                                [0,0,0,1,1],
                                [0,0,0,0,1],
                                [0,0,0,0,0]])
    completedominanceweight = np.array([[0,5,5,5,5],
                                        [0,0,400,100,100],
                                        [0,0,0,1,1],
                                        [0,0,0,0,1],
                                        [0,0,0,0,0]])
    worstcase = np.zeros((5,5))
    print("Colley test:" + str(cra.rank(est)))
    print("Colley test fivegood:" + str(cra.rank(fivegood)))
    print("Colley test perfect season:" + str(cra.rank(completedominance)))
    print("Colley test worst case:" + str(cra.rank(worstcase)))
    print("Massey test:" + str(mra.rank(est)))
    print("Massey test perfect season weighted:" + str(mra.rank(completedominanceweight)))
    print("The Massey method does not work on the worst case")
    
#if __name__ == "__main__":
#   main()

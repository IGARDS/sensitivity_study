import numpy as np
import math
import sys
import random
sys.path.append("/home/jwaschur/rankability_toolbox")
import pyrankability
from scipy import stats
from tqdm import tqdm

######## PROBLEM INSTANCE ########

class ProblemInstance:
    
    def __init__(self, dataSource, noiseGenerator):
        self.dataSource = dataSource
        self.noiseGenerator = noiseGenerator
    
    def get_sensitivity(self, rankingAlg, rankability_metrics, n_trials=100, progress_bar=True):
        # Load in the initial D matrix and get a ranking without noise
        D = self.dataSource.init_D()
        perfect_ranking = rankingAlg.rank(D)
        
        # Get P of original D matrix using most efficient algorithm
        k, details = pyrankability.hillside.bilp(D, max_solutions=1, num_random_restarts=10)
        
        # Compute each considered rankability metric
        rs = [metric.compute(k, details) for metric in rankability_metrics]
        
        # Setup the progress bar if needed
        if progress_bar:
            range_iter = tqdm(range(n_trials), ascii=True)
        else:
            range_iter = range(n_trials)
        
        taus = []
        for trial_index in range_iter:
            D_noisy = self.noiseGenerator.apply_noise(D)
            noisy_ranking = rankingAlg.rank(D_noisy)
            tau, pval = stats.kendalltau(perfect_ranking, noisy_ranking)
            taus.append(tau)
        
        return rs, taus

    
######## RANKABILITY METRICS ########
    
class RankabilityMetric:
    def compute(self, k, details):
        # Child classes should compute their rankability metric from k and P
        raise NotImplemented("Don't use the generic NoiseGenerator class")

class RatioToMaxMetric(RankabilityMetric):
    def compute(self, k, details):
        P = details["P"]
        n = len(P[0])
        return 1.0 - (k*len(P) / ((n**2 - n)/2*math.factorial(n)))
    
class PDiversityMetric(RankabilityMetric):
    
    # This function found at: https://stackoverflow.com/a/48916127
    def kendall_w(self, expt_ratings):
        if expt_ratings.ndim!=2:
            raise 'ratings matrix must be 2-dimensional'
        m = expt_ratings.shape[0] # number of raters
        n = expt_ratings.shape[1] # number of items
        denom = m**2*(n**3-n)
        rating_sums = np.sum(expt_ratings, axis=0)
        S = n*np.var(rating_sums)
        return 12*S/denom
    
    def compute(self, k, details):
        P = details["P"]
        n = len(P[0])
        return 1 - ((k / ((n*n - n)/2)) * (1-self.kendall_w(np.array(P))))


######## NOISE GENERATORS ########
    
class NoiseGenerator:
    def apply_noise(self, D):
        # Child classes should return numpy array of D_tilde
        raise NotImplemented("Don't use the generic NoiseGenerator class")

class PercentageFlipNoise:
    
    def __init__(self, noisePercentage):
        self.noisePercentage = noisePercentage
    
    def apply_noise(self, D):
        D_noisy = np.copy(D)
        n = len(D_noisy)
        num_flips = (np.square(n) - n) * self.noisePercentage
        unique_elems = set()
        for flip in range(int(num_flips)):
            i, j = random.sample(range(n), 2)
            while ((i, j) in unique_elems): i, j = random.sample(range(n), 2)
            unique_elems.add((i, j))
            D_noisy[i][j] = 1 - D_noisy[i][j]
        return D_noisy

    
######## DATA SOURCES ########
    
class DataSource:
    def init_D(self):
        # Child classes should return numpy array
        raise NotImplemented("Don't use the generic DataSource class")

class PerfectBinarySource(DataSource):
    def __init__(self, n):
        self.n = n
    
    def init_D(self):
        D = np.zeros((self.n,self.n), dtype=int)
        D[np.triu_indices(self.n,1)] = 1
        return D        


######## RANKING ALGORITHMS ########

class RankingAlgorithm:
        
    def rank(D):
        # Child classes should return numpy array of ranking vector
        raise NotImplemented("Don't use the generic RankingAlgorithm class")
        
class LOPRankingAlgorithm(RankingAlgorithm):
    def rank(self, D):
        k, details = pyrankability.hillside.bilp(D,max_solutions=1)
        # This could return the full P set or randomly sample from it rather
        # than reporting the first it finds.
        return details["P"][0]
    
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
    
class MarkovChainRankingAlgorithm(RankingAlgorithm):
    def rank(self, D):
        V = np.transpose(D.astype(float))
        wins = [sum(D[i]) for i in range(0,D.shape[0])]
        losses = [sum(np.transpose(D)[i]) for i in range(0,D.shape[0])]
        totalevents = [wins[i] + losses[i] for i in range(0,D.shape[0])]
        print("totalevents: ", totalevents)
        maxevents = max(totalevents)
        print(V)
        for i in range(V.shape[0]):
            if sum(V[i]) != 0:
                V[i] = np.divide(V[i], sum(V[i]))
                V[i][i] = maxevents - sum(V[i])
                V[i] = np.divide(V[i], maxevents)
        print(V)
        eigenvals, eigenvecs = np.linalg.eig(V)
        print(eigenvals)
        print(eigenvecs)
        indiciesreverse = eigenvals.argsort()[::-1]
        print(eigenvecs[indiciesreverse[0]])
    
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

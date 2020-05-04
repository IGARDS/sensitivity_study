import numpy as np
import math
import random
import sys
import itertools
sys.path.insert(0,"/home/egoldfar/rankability_toolbox/pyrankability")
#sys.path.append("/home/jwaschur/rankabiity_toolbox")
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
        print(D)
        k, details = pyrankability.hillside.bilp(D, num_random_restarts=10)
        print(k)
        print(details)
        
        # Compute each considered rankability metric
        rs = [metric.compute(k, P) for metric in rankability_metrics]
        
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
    
    def get_sensitivity_sample_P(self, rankingAlg, rankability_metrics, n_trials=100, progress_bar=True):
        # Load in the initial D matrix and get a ranking without noise
        D = self.dataSource.init_D()
        perfect_ranking = rankingAlg.rank(D)
        
        # Sample set P of D matrix
        print(D)
        k, details = pyrankability.lop.lp(D)
        print(k)
        P, info = pyrankability.lop.find_P_from_x(D, k, details)
        
        # Compute each considered rankability metric
        rs = [metric.compute(k, P) for metric in rankability_metrics]
        
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
    def compute(self, k, P):
        # Child classes should compute their rankability metric from k and P
        raise NotImplemented("Don't use the generic RankabilityMetric class")

class RatioToMaxMetric(RankabilityMetric):
    def compute(self, k, P):
        n = len(P[0])
        print(k, P)
        return 1.0 - (k*len(P) / ((n**2 - n)/2*math.factorial(n)))
    
class PDiversityMetric(RankabilityMetric):
    # This function found at: https://stackoverflow.com/a/48916127
    def kendall_w(self, expt_ratings):
        print(expt_ratings)
        if expt_ratings.ndim!=2:
            raise 'ratings matrix must be 2-dimensional'
        m = expt_ratings.shape[0] # number of raters
        n = expt_ratings.shape[1] # number of items
        denom = m**2*(n**3-n)
        rating_sums = np.sum(expt_ratings, axis=0)
        S = n*np.var(rating_sums)
        return 12*S/denom
    
    def compute(self, k, P):
        n = len(P[0])
        return 1 - ((k / ((n*n - n)/2)) * (1-self.kendall_w(np.array(P))))

class L2DifferenceMetric(RankabilityMetric):
    #RankVectors should be an array of ranking vectors
    def MaxL2Difference(self, RankVectors):
        print(list(itertools.combinations(RankVectors, 2)))
        pair = max(list(itertools.combinations(RankVectors, 2)), key=lambda x: math.sqrt(np.dot(np.array(x[0]) - np.array(x[1]), np.array(x[0]) - np.array(x[1]))))
        return math.sqrt(np.dot(np.array(pair[0]) - np.array(pair[1]), np.array(pair[0]) - np.array(pair[1])))/len(RankVectors)
        
        

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
        D = np.full(shape=(self.n,self.n), fill_value=0, dtype=int)
        for i in range(self.n):
            for j in range(i+1, self.n):
                    D[j,i] = (self.n-i) + j
        return D        


######## RANKING ALGORITHMS ########

class RankingAlgorithm:
        
    def rank(D):
        # Child classes should return numpy array of ranking vector
        raise NotImplemented("Don't use the generic RankingAlgorithm class")
        
class LOPRankingAlgorithm(RankingAlgorithm):
    def rank(self, D):
        k, details = pyrankability.lop.lp(D)
        P, info = pyrankability.lop.find_P_from_x(D, k, details)
        # This could return the full P set or randomly sample from it rather
        # than reporting the first it finds.
        return P[0]
    
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
    
    #print(sys.path)
    #print(pyrankability.__file__)
    #k, details = pyrankability.hillside.bilp(completedominanceweight, num_random_restarts=10, find_pair=True)
    #print(k)
    #print(details)
    #print(pyrankability.hillside.bilp_two_most_distant(fivegood))
    
    testmatrix = PerfectBinarySource(10)
    perf = testmatrix.init_D()
    print(perf)
    
    def noise_D(D, noise_percent):
        Dcopy = D
        for i in range(D.shape[0]):
            for j in range(i+1, D.shape[0]):
                if  random.random() <= noise_percent:
                    temp = Dcopy[i,j]
                    Dcopy[i,j] = Dcopy[j,i]/2
                    Dcopy[j,i] = temp
        return Dcopy
    noisy = noise_D(perf, .533333)
    print(noisy)
    k, details = pyrankability.hillside.bilp(noisy, num_random_restarts=10, find_pair=True)
    print(k)
    print(details["P"])
    l2dm = L2DifferenceMetric()
    print(l2dm.MaxL2Difference(details["P"]))
    
#if __name__ == "__main__":
#   main()

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
    
class KendallWMetric(RankabilityMetric):
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

class MeanTauMetric(RankabilityMetric):
    # Two similar statistics exist for the use of mean tau
    #      "Hays"  -->  W_a defined by Hays (1960)
    # "Ehrenberg"  -->  W_t defined by Ehrenberg (1952)
    #
    # The two strategies are the same when m (# of rankings) is even
    # Hays' W_a is computed with m+1 instead of m when m is odd
    # This ensures that 0 is the minimum value at all values of m
    #
    # In keeping with the statistics literature I've read:
    #    m  <-- number of rankings (equivalent to p)
    #    n  <-- number of items to rank / length of rankings
    #    u  <-- mean kendall tau over all pairs of rankings
    def __init__(self, strategy="Hays"):
        strategy = strategy.lower()
        if strategy == "hays":
            self.get_W = self.get_W_t
        elif strategy == "ehrenberg":
            self.get_W = self.get_W_a
        else:
            raise ValueError("Unrecognized MeanTauMetric strategy: %s" % strategy)
    
    # Defined by Ehrenberg (1952) as a possible measure of concordance
    def get_W_t(m, u):
        return ((m - 1.0) * u + 1.0) / m
    
    # Defined by Hays (1960) so that a minimum of 0 is always possible for all values of m
    def get_W_a(m, u):
        if m & 1 == 1:
            m += 1
        return ((m - 1.0) * u + 1.0) / m
    
    def compute(self, k, details):
        P = details["P"]
        p = len(P)
        if p == 1:
            return 1.0
        if p == 2:
            u, _ = stats.kendalltau(P[0], P[1])
        else:
            u = 0.0
            for r1 in range(p):
                for r2 in range(r1, p):
                    u += stats.kendalltau(P[r1], P[r2])[0]
            u /= (p*(p-1)/2)
        n = len(P[0])
        return 1 - (k / ((n*n - n)/2) * (1.0 - self.get_W(p, u))) # should be *(1-W)?
        

######## NOISE GENERATORS ########

class NoiseGenerator:
    def apply_noise(self, D):
        # Child classes should return numpy array of D_tilde
        raise NotImplemented("Don't use the generic NoiseGenerator class")

class BinaryFlipNoise(NoiseGenerator):
    
    def __init__(self, noisePercentage):
        self.noisePercentage = noisePercentage
    
    def apply_noise(self, D):
        D_noisy = np.copy(D)
        n = len(D_noisy)
        num_flips = (np.square(n) - n) * self.noisePercentage
        unique_elems = set()
        for flip in range(int(num_flips)):
            i, j = random.sample(range(n), 2)
            # May consider using another method to avoid this resampling loop
            # which becomes very expensive as noisePercentage->1.0 and len(D)->large
            while ((i, j) in unique_elems): i, j = random.sample(range(n), 2)
            unique_elems.add((i, j))
            D_noisy[i][j] = 1 - D_noisy[i][j]
        return D_noisy        

class SwapNoise(NoiseGenerator):
    # Flips (swaps edge weights for (i,j) and (j,i)) for given percentage
    # of relationships. 100% noise corresponds to returning D.T
    
    def __init__(self, noisePercentage):
        self.noisePercentage = noisePercentage
    
    def apply_noise(self, D):
        D_noisy = np.copy(D)
        n = len(D_noisy)
        num_swaps = int((n*n - n)/2 * self.noisePercentage)
        if self.noisePercentage > 0.5:
            D_noisy = D_noisy.T
            num_swaps = (n*n - n)/2 - num_swaps
        # Get indices of upper-triangular elements
        i_arr, j_arr = np.triu_indices(n,1)
        num_offdiag = len(i_arr)
        indices = random.sample(range(num_offdiag), num_swaps)
        for idx in indices:
            i, j = i_arr[idx], j_arr[idx]
            # Reverse this relationship
            temp = D_noisy[i,j]
            D_noisy[i][j] = D_noisy[j][i]
            D_noisy[j][i] = temp
        return D_noisy

class BootstrapResamplingNoise(NoiseGenerator):
    
    def __init__(self, noisePercentage):
        self.noisePercentage = noisePercentage
    
    def apply_noise(self, D):
        D_noisy = np.copy(D)
        n = len(D_noisy)
        num_resampled = int((n*n - n) * self.noisePercentage)
        # Get indices of all off-diagonal elements
        i_arr, j_arr = np.where(~np.eye(D.shape[0],dtype=bool))
        
        # Select the elements to be resampled and the bootstrap samples to use
        num_offdiag = len(i_arr)
        indices = random.sample(range(num_offdiag), num_resampled) # Without replacement
        bootstrap = np.random.randint(num_offdiag, size=num_resampled) # With replacement
        
        for idx, rel_idx in enumerate(indices):
            # rel_idx is the index of the off-diagonal relationship to be replaced
            # idx is the index of the bootstrap sample to replace it with
            bs_rel_idx = bootstrap[idx]
            # Replace the chosen off-diag element of D_noisy using the
            # off-diag element chosen from D
            D_noisy[i_arr[rel_idx], j_arr[rel_idx]] = D[i_arr[bs_rel_idx], j_arr[bs_rel_idx]]
        
        return D_noisy

class NormalResamplingNoise(NoiseGenerator):
    
    def __init__(self, noisePercentage):
        self.noisePercentage = noisePercentage
    
    def apply_noise(self, D):
        D_noisy = np.copy(D)
        n = len(D_noisy)
        num_resampled = int((n*n - n) * self.noisePercentage)
        
        # Get indices of all off-diagonal elements
        i_arr, j_arr = np.where(~np.eye(D.shape[0],dtype=bool))
        
        # Fit a normal distribution to all off-diagonal elements
        mu, sigma = stats.norm.fit(D[i_arr, j_arr])
        
        # Select the elements to be resampled
        indices = random.sample(range(len(i_arr)), num_resampled) # Without replacement
        
        for rel_idx in indices:
            # rel_idx is the index of the off-diagonal relationship to be replaced
            # Replace the chosen off-diag element of D_noisy using a
            # sample from the normal distribution fit to the data
            D_noisy[i_arr[rel_idx], j_arr[rel_idx]] = stats.norm.rvs(loc=mu, scale=sigma)
        
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

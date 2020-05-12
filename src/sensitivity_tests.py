import numpy as np
import math
import random
import sys
import itertools
sys.path.append("~/rankability_toolbox")
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
        
        k, details = pyrankability.hillside.bilp(D, num_random_restarts=10)
        
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
        P = list(set(details["P"]))
        return self._compute(k, P)
    
    def _compute(self, k, P):
        # Child classes should compute their rankability metric from k and P
        raise NotImplemented("Don't use the generic RankabilityMetric class")


class RatioToMaxMetric(RankabilityMetric):
    def compute(self, k, P):
        n = len(P[0])
        return 1.0 - (k*len(P) / ((n**2 - n)/2 * math.factorial(n)))
    

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
    
    def _compute(self, k, P):
        n = len(P[0])
        return 1 - ((k / (n**3 - n**2)) * (1-self.kendall_w(np.array(P))))


class L2DifferenceMetric(RankabilityMetric):
    
    def __init__(self, strategy="max"):
        strategy=strategy.lower()
        self.strategy = strategy
        if strategy == "max":
            self.get_dist_stat = self.get_max_dist
        elif strategy == "mean":
            self.get_dist_stat = self.get_mean_dist
        else:
            raise ValueError("Unrecognized L2DifferenceMetric strategy: %s" % strategy)
    
    def get_mean_dist(self, P):
        p = len(P)
        mean_dist = 0.0
        for r1 in range(p):
            for r2 in range(r1, p):
                mean_dist += np.linalg.norm(np.array(P[r1]) - np.array(P[r2]))
        
        if p == 1:
            return mean_dist
        else:
            return mean_dist / (p*(p-1)/2.0)
    
    def get_max_dist(self, P):
        p = len(P)
        max_dist = 0.0
        for r1 in range(p):
            for r2 in range(r1, p):
                dist = np.linalg.norm(np.array(P[r1]) - np.array(P[r2]))
                if dist > max_dist:
                    max_dist = dist
        return max_dist
    
    def _compute(self, k, P):
        p = len(P)
        n = len(P[0])
        dist = self.get_dist_stat(P)
        return 1.0 - (k / (n**3 - n**2) * (dist / np.sqrt((n / 3) * (n**2 - 1))))
    

class MeanTauMetric(RankabilityMetric):
    # Two similar statistics exist for the use of mean tau
    #      "hays"  -->  W_a defined by Hays (1960)
    # "ehrenberg"  -->  W_t defined by Ehrenberg (1952)
    #
    # The two strategies are the same when m (# of rankings) is even
    # Hays' W_a is computed with m+1 instead of m when m is odd
    # This ensures that 0 is the minimum value at all values of m
    #
    # In keeping with the statistics literature I've read:
    #    m  <-- number of rankings (equivalent to p)
    #    n  <-- number of items to rank / length of rankings
    #    u  <-- mean kendall tau over all pairs of rankings
    
    def __init__(self, strategy="hays"):
        strategy = strategy.lower()
        self.strategy = strategy
        if strategy == "hays":
            self.get_W = self.get_W_a
        elif strategy == "ehrenberg":
            self.get_W = self.get_W_t
        else:
            raise ValueError("Unrecognized MeanTauMetric strategy: %s" % strategy)
    
    # Defined by Ehrenberg (1952) as a possible measure of concordance
    def get_W_t(self, m, u):
        return ((m - 1.0) * u + 1.0) / m
    
    # Defined by Hays (1960) so that a minimum of 0 is always possible for all values of m
    def get_W_a(self, m, u):
        if m & 1 == 1:
            m += 1
        return ((m - 1.0) * u + 1.0) / m
    
    def _compute(self, k, P):
        p = len(P)
        if p == 1:
            return 1.0
        if p == 2:
            u, _ = stats.kendalltau(P[0], P[1])
        else:
            u = 0.0
            for r1 in range(p):
                for r2 in range(r1+1, p):
                    tau, _ = stats.kendalltau(P[r1], P[r2])
                    u += tau
            u /= p*(p-1)/2
        n = len(P[0])
        w_stat = self.get_W(p, u)
        return 1 - (k / (n**3 - n**2) * (1.0 - w_stat))
        

######## NOISE GENERATORS ########
    
class NoiseGenerator:
    
    def apply_noise(self, D):
        # Child classes should return numpy array of D_tilde
        raise NotImplemented("Don't use the generic NoiseGenerator class")
        
class IdentityNoise(NoiseGenerator):
    def apply_noise(self, D):
        return D

class BinaryFlipNoise(NoiseGenerator):
    def __init__(self, noisePercentage):
        self.noisePercentage = noisePercentage
    
    def apply_noise(self, D):
        D_noisy = np.copy(D)
        n = len(D_noisy)
        num_flips = (np.square(n) - n) * self.noisePercentage
        unique_elems = set()
        for flip in range(int(num_flips)):
            i, j = random.sample(range(n), 2) #Ensures that i and j are distinct
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
            num_swaps = int((n*n - n)/2 - num_swaps)
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
    
    def __init__(self, noisePercentage, clip_range=False):
        self.noisePercentage = noisePercentage
        self.clip_range = clip_range
    
    def apply_noise(self, D):
        D_noisy = np.copy(D)
        n = len(D_noisy)
        num_resampled = int((n*n - n) * self.noisePercentage)
        
        # Get indices of all off-diagonal elements
        i_arr, j_arr = np.where(~np.eye(D.shape[0],dtype=bool))
        offdiags = D[i_arr, j_arr]
        
        # Get the range of the off-diagonal elements if clipping
        if self.clip_range:
            D_min = np.amin(offdiags)
            D_max = np.amax(offdiags)
        
        # Fit a normal distribution to all off-diagonal elements
        mu, sigma = stats.norm.fit(offdiags)
        
        # Select the elements to be resampled
        indices = random.sample(range(len(i_arr)), num_resampled) # Without replacement
        
        for rel_idx in indices:
            # rel_idx is the index of the off-diagonal relationship to be replaced
            # Replace the chosen off-diag element of D_noisy using a
            # sample from the normal distribution fit to the data
            sample = stats.norm.rvs(loc=mu, scale=sigma)
            if self.clip_range:
                sample = np.clip(sample, D_min, D_max)
            D_noisy[i_arr[rel_idx], j_arr[rel_idx]] = sample
        
        return D_noisy


######## DATA SOURCES ########
    
class DataSource:
    
    def init_D(self):
        # Child classes should return numpy array
        raise NotImplemented("Don't use the generic DataSource class")
        
                      
class LOLib(DataSource):
    def __init__(self, file_name):
        self.n = 0
        self.file_name = file_name
        
    def init_D(self):
        with open("src/lolib_data/" + self.file_name, 'r') as f:
            unparsed = f.read().splitlines()
        self.n = int(unparsed[0]) #0th element is dim of D
        elements = [row.split(' ') for row in unparsed[1:]]
        D = []
        for row in elements:
            D.append([int(e) for e in row if e != ''])
        return np.array(D)
    
        
class PerfectBinarySource(DataSource):
    
    def __init__(self, n):
        self.n = n
    
    def init_D(self):     
        D = np.zeros((self.n,self.n), dtype=int)
        D[np.triu_indices(self.n,1)] = 1
        return D


class PerfectWeightedSource(DataSource):
    
    def __init__(self, n, scale=1):
        self.n = n
        self.scale = scale
    
    def init_D(self):
        D = np.zeros((self.n,self.n), dtype=float)
        i_arr, j_arr = np.triu_indices(self.n,1)
        for idx in range(len(i_arr)):
            i, j = i_arr[idx], j_arr[idx]
            D[i, j] = self.scale * (self.n-i+j)
        return D


class SynthELOTournamentSource(DataSource):
    # Simulates a tournament played by competitors with
    # normally distributed ELO scores
    # This ELO system uses a logistic curve to predict win probability
    
    def __init__(self, n, n_games=5, comp_var=200, elo_scale=400):
        # n --> the number of competitors
        # n_games --> the number of games played between every pair of competitors
        # comp_var --> the variance of the normal distribution competitor ELOs are drawn from
        #     a smaller comp_var/elo_scale ratio will produce a very rankable matrix
        #     a larger comp_var/elo_scale ratio will produce a very unrankable matrix
        # elo_scale --> the scale parameter for the ELO system
        self.n = n
        self.n_games = n_games
        self.comp_var = comp_var
        self.elo_scale = elo_scale
    
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    def init_D(self):
        ELOs = sorted(stats.norm.rvs(scale=self.comp_var, size=self.n), reverse=True)
        D = np.zeros((self.n,self.n), dtype=float)
        i_arr, j_arr = np.triu_indices(self.n,1)
        for idx in range(len(i_arr)):
            i, j = i_arr[idx], j_arr[idx]
            scaled_elo_diff = (ELOs[i] - ELOs[j])/self.elo_scale
            prob_i_win = self.sigmoid(scaled_elo_diff)
            i_wins = np.random.binomial(n=self.n_games, p=prob_i_win)
            j_wins = self.n_games - i_wins
            D[i, j] = i_wins
            D[j, i] = j_wins
        return D


######## RANKING ALGORITHMS ########

class RankingAlgorithm:
    
    def rank(D):
        # Child classes should return numpy array of ranking vector
        raise NotImplemented("Don't use the generic RankingAlgorithm class")


class LOPRankingAlgorithm(RankingAlgorithm):
    
    def rank(self, D):
        k, details = pyrankability.hillside.bilp(D)
        # This could return the full P set or randomly sample from it rather
        # than reporting the first it finds.
        return details["P"][0]


class ColleyRankingAlgorithm(RankingAlgorithm):
    def rank(self, D):
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
    #k, details = pyrankability.hillside.bilp(noisy, num_random_restarts=10, find_pair=True)
    #print(k)
    #print(details["P"])
    #print(l2dm.MaxL2Difference(details["P"]))
    
    eloTournament = SynthELOTournamentSource(16, 5, 80, 800)
    smalleloTournament = SynthELOTournamentSource(4, 5, 80, 800)
    l2dm = L2DifferenceMetric()
    
    eloMatrix = eloTournament.init_D()
    smalleloMatrix = smalleloTournament.init_D()
    print(eloMatrix)
    #k, details = pyrankability.hillside.bilp_two_most_distant(eloMatrix)
    #print(details["perm_x"],details["perm_y"])
    #print(details["x"],details["y"])
    k, details = pyrankability.hillside.bilp(eloMatrix, num_random_restarts=10, find_pair=True)
    print(k)
    print(set(details["P"]))
    print(l2dm.MaxL2Difference(details["P"]))
    k, details = pyrankability.hillside.bilp(smalleloMatrix, num_random_restarts=10, find_pair=True)
    print(k)
    print(set(details["P"]))
    print(l2dm.MaxL2Difference(details["P"]))
    

    

#if __name__ == "__main__":
#   main()

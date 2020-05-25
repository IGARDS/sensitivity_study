import numpy as np
import math
import random
import sys
import itertools
#sys.path.append("~/rankability_toolbox")
#import pyrankability
from pyrankability_dev.rank import solve
from pyrankability_dev.search import solve_pair_min_tau
import json
from tqdm import tqdm
from utilities import *

######## STATISTICS OF P ########
# Functions of P which package certain similar statistics
# together for sake of efficiency. These functions will probably
# replace the current "RankabilityMetric" classes since running
# cross-validated linear models seems to be the better alternative
# to hand-crafting metrics.

# This function found at: https://stackoverflow.com/a/48916127
def kendall_w(expt_ratings):
    expt_ratings = np.array(expt_ratings)
    if expt_ratings.ndim!=2:
        raise 'ratings matrix must be 2-dimensional'
    m = expt_ratings.shape[0] # number of raters
    n = expt_ratings.shape[1] # number of items
    denom = m**2*(n**3-n)
    rating_sums = np.sum(expt_ratings, axis=0)
    S = n*np.var(rating_sums)
    return "kendall_w", 12*S/denom

def p_len(P):
    return "p_lowerbound", len(P)

def l2_dist_stats(P):
    p = len(P)
    max_dist = 0.0
    mean_dist = 0.0
    for r1 in range(p):
        for r2 in range(r1+1, p):
            dist = np.linalg.norm(np.array(P[r1]) - np.array(P[r2]))
            mean_dist += dist
            if dist > max_dist:
                max_dist = dist
                
    if p > 1:
        mean_dist /= (p*(p-1)/2.0)
    
    return ["max_L2_dist", "mean_L2_dist"], [max_dist, mean_dist]

def tau_stats(P):
    p = len(P)
    min_tau = 1.0
    mean_tau = 0.0
    for r1 in range(p):
        for r2 in range(r1+1, p):
            tau = kendall_tau(P[r1], P[r2])
            mean_tau += tau
            if tau < min_tau:
                min_tau = tau
                
    if p > 1:
        mean_tau /= (p*(p-1)/2.0)
    else:
        mean_tau = 1.0
    
    return ["min_tau", "mean_tau"], [min_tau, mean_tau]

def get_P_stats(P):
    # Ensure P is a set, then make addressable as a list
    P = list(set(P))
    results = {}
    for func in [kendall_w, p_len, l2_dist_stats, tau_stats]:
        stat_names, stat_values = func(P)
        if type(stat_names) == list:
            for i in range(len(stat_names)):
                results[stat_names[i]] = stat_values[i]
        else:
            results[stat_names] = stat_values
    return results

######## RANKABILITY METRICS ########
    
class RankabilityMetric:
    
    def compute(self, k, details):
        P = list(set(details["P"]))
        return self._compute(k, P)
    
    def _compute(self, k, P):
        # Child classes should compute their rankability metric from k and P
        raise NotImplemented("Don't use the generic RankabilityMetric class")


class RatioToMaxMetric(RankabilityMetric):
    def _compute(self, k, P):
        n = len(P[0])
        return 1.0 - (k*len(P) / ((n**2 - n)/2 * math.factorial(n)))
    

class KendallWMetric(RankabilityMetric):
    
    def _compute(self, k, P):
        n = len(P[0])
        _, w = kendall_w(P)
        return 1 - ((k / (n**3 - n**2)) * (1 - w))


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
            for r2 in range(r1+1, p):
                mean_dist += np.linalg.norm(np.array(P[r1]) - np.array(P[r2]))
        
        if p == 1:
            return mean_dist
        else:
            return mean_dist / (p*(p-1)/2.0)
    
    def get_max_dist(self, P):
        p = len(P)
        max_dist = 0.0
        for r1 in range(p):
            for r2 in range(r1+1, p):
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
            u = kendall_tau(P[0], P[1])
        else:
            u = 0.0
            for r1 in range(p):
                for r2 in range(r1+1, p):
                    tau = kendall_tau(P[r1], P[r2])
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
    
    def __str__(self):
        return "IdentityNoise()"

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
    
    def __str__(self):
        return "BinaryFlipNoise({})".format(self.noisePercentage)


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
    
    def __str__(self):
        return "SwapNoise({})".format(self.noisePercentage)


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
    
    def __str__(self):
        return "BootstrapResamplingNoise({})".format(self.noisePercentage)


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
    
    def __str__(self):
        return "NormalResamplingNoise({})".format(self.noisePercentage)


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
    
    def __str__(self):
        return "LOLib({})".format(self.file_name)
    
        
class PerfectBinarySource(DataSource):
    
    def __init__(self, n):
        self.n = n
    
    def init_D(self):     
        D = np.zeros((self.n,self.n), dtype=int)
        D[np.triu_indices(self.n,1)] = 1
        return D
    
    def __str__(self):
        return "PerfectBinarySource({})".format(self.n)


class PerfectWeightedSource(DataSource):
    
    def __init__(self, n, scale=1):
        self.n = n
        self.scale = scale
    
    def init_D(self):
        D = np.zeros((self.n,self.n), dtype=float)
        i_arr, j_arr = np.triu_indices(self.n,1)
        for idx in range(len(i_arr)):
            i, j = i_arr[idx], j_arr[idx]
            D[i, j] = self.scale * (-i+j)
        return D
    
    def __str__(self):
        return "PerfectWeightedSource({},{})".format(self.n, self.scale)


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
    
    def __str__(self):
        return "SynthELOTournamentSource({},{},{},{})".format(self.n, self.n_games, self.comp_var, self.elo_scale)


class UniformRandomSource(DataSource):
    # Every off-diagonal element of the matrix is drawn independently from a uniform
    # distribution over the range of integers [0, max_val].
    
    def __init__(self, n, max_val=1):
        # n --> the number of items
        # max_val --> the maximum value that can appear as an element in the matrix
        self.n = n
        self.max_val = max_val
    
    def init_D(self):
        D = np.random.randint(low=0, high=self.max_val+1, size=(self.n,self.n), dtype=int)
        for i in range(self.n):
            D[i,i] = 0
        return D
    
    def __str__(self):
        return "UniformRandomSource({},{})".format(self.n, self.max_val)


######## RANKING ALGORITHMS ########

class RankingAlgorithm:
    
    def rank(D):
        # Child classes should return numpy array of ranking vector
        raise NotImplemented("Don't use the generic RankingAlgorithm class")


class LOPRankingAlgorithm(RankingAlgorithm):
    
    def rank(self, D):
        k, details = solve(D)
        # This could return the full P set or randomly sample from it rather
        # than reporting the first it finds.
        return details["P"][0]
    
    def __str__(self):
        return "LOPRankingAlgorithm"


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
        retvec = [r[i][1]-1 for i in range(len(r)-1, -1, -1)]
        return retvec
    
    def __str__(self):
        return "ColleyRankingAlgorithm"


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
        retvec = [r[i][1]-1 for i in range(len(r)-1, -1, -1)]
        return retvec
    
    def __str__(self):
        return "MasseyRankingAlgorithm"


class MarkovChainRankingAlgorithm(RankingAlgorithm):
    def rank(self, D):
        #f = np.vectorize(lambda x: 1 if x > 0 else 0)
        '''for i in range(D.shape[0]):
            for j in range(i, D.shape[1]):
                if D[i][j] > D[j][i] and i != j:
                    D[i][j] = 1
                    D[j][i] = 0
                elif D[i][j] < D[j][i] and i != j:
                    D[j][i] = 1
                    D[i][j] = 0
        '''
        V = np.transpose(D.astype(float))
        n = D.shape[0]
        wins = [sum(D[i]) for i in range(0,D.shape[0])]
        losses = [sum(np.transpose(D)[i]) for i in range(0,D.shape[0])]
        totalevents = [wins[i] + losses[i] for i in range(0,D.shape[0])]
        #print("totalevents: ", totalevents)
        maxevents = max(totalevents)
        #print(V)
        for i in range(V.shape[0]):
            if sum(V[i]) != 0:
                V[i] = np.divide(V[i], sum(V[i]))
            else:
                V[i] = np.zeros(n)
                V[i][i] = 1
        print(V)
        eigenvals, eigenvecs = np.linalg.eig(V)
        #print("eigenvals:\n", eigenvals)
        #for i in range(len(eigenvecs)):
        #    print(eigenvecs[i])
        #print("eigenvecs:\n", eigenvecs)
        print("Max eigenvalue/vector:", max(eigenvals), eigenvecs[np.argmax(eigenvals)]/np.linalg.norm(eigenvecs[np.argmax(eigenvals)]))
        if True in np.iscomplex(eigenvecs[np.argmax(eigenvals)]):
            print("Complex rating vector")
            return
        return np.argsort(eigenvecs[np.argmax(eigenvals)])
    
    def __str__(self):
        return "MarkovChainRankingAlgorithm"
    
class MarkovModifiedRankingAlgorithm(RankingAlgorithm):
    def rank(self, D):
        V = D.astype(float)
        n = D.shape[0]
        wins = [sum(D[i]) for i in range(0,D.shape[0])]
        losses = [sum(np.transpose(D)[i]) for i in range(0,D.shape[0])]
        totalevents = [wins[i] + losses[i] for i in range(0,D.shape[0])]
        maxevents = max(totalevents)
        #print(V)
        for i in range(V.shape[0]):
            if sum(V[i]) != 0:
                V[i] = np.divide(V[i], sum(V[i]))
            else:
                V[i] = np.zeros(n)
                V[i][i] = 1
        #print(V)
        return np.argsort(V.sum(axis=0))
        
    def __str__(self):
        return "MarkovModifiedRankingAlgorithm"

    
####### SOME HELPFUL GLOBALS #######

ALL_RANKING_ALGS = [LOPRankingAlgorithm(), MasseyRankingAlgorithm(), ColleyRankingAlgorithm()]
LOW_INTENSITY_NOISE_GENS = [SwapNoise(0.05), BinaryFlipNoise(0.05)]

######## PROBLEM INSTANCE ########

class ProblemInstance:
    
    def __init__(self, dataSource):
        self.dataSource = dataSource
        self._D = None
    
    def get_D(self, refresh=False):
        if self._D is None or refresh:
            self._D = self.dataSource.init_D()
        return self._D
    
    def get_optimal_rankings(self,
                             model="lop",
                             num_random_restarts=0):
        if model in ["lop", "hillside"]:
             return solve(self.get_D(), method=model, num_random_restarts=num_random_restarts)
        else:
            raise ValueError("Unrecognized model name '{}' should be one of 'lop' or 'hillside'".format(model))
    
    def get_most_distant_optimal_rankings(self,
                                          model="lop"):
        if model in ["lop","hillside"]:
            return solve_pair_min_tau(self.get_D(), method=model, verbose=False)
        else:
            raise ValueError("Unrecognized model name '{}' should be one of 'lop' or 'hillside'".format(model))
    
    def collect_data(self,
                     ranking_algorithms=ALL_RANKING_ALGS,
                     noise_generators=LOW_INTENSITY_NOISE_GENS,
                     model="lop",
                     num_random_restarts=200,
                     n_sensitivity_trials=50):
        D = self.get_D(refresh=True)
        k, details = self.get_optimal_rankings(model=model,
                                               num_random_restarts=num_random_restarts)
        P = details["P"]
        k_furthest, details_furthest = self.get_most_distant_optimal_rankings(model=model)
        if details_furthest is not None:
            P.append(details_furthest["perm_x"])
            P.append(details_furthest["perm_y"])
        data = get_P_stats(P)
        data["k"] = k
        if model == "lop":
            data["degree_of_linearity"] = k / np.sum(D)
        else:
            k_lop, _ = self.get_optimal_rankings(model="lop", num_random_restarts=0)
            data["degree_of_linearity"] = k_lop / np.sum(D)
        data["model"] = model
        data["D"] = json.dumps(D.tolist())
        data["Source"] = str(self.dataSource)
        data["n_items"] = D.shape[0]
        data["P"] = str(list(set(P)))
        data["P_repeats"] = str(P)
        
        for rankingAlg in ranking_algorithms:
            for noiseGenerator in noise_generators:
                taus = self.get_sensitivity(rankingAlg,
                                            noiseGenerator,
                                            model=model,
                                            n_trials=n_sensitivity_trials)
                sensitivities = (1.0 - np.array(taus)) / 2.0
                mean_tau_name = "mean_sensitivity({},{})".format(str(rankingAlg), str(noiseGenerator))
                data[mean_tau_name] = np.mean(sensitivities)
                std_tau_name = "std_sensitivity({},{})".format(str(rankingAlg), str(noiseGenerator))
                data[std_tau_name] = np.std(sensitivities)
        
        return data
    
    def get_sensitivity(self,
                        rankingAlg,
                        noiseGenerator,
                        model="lop",
                        n_trials=100,
                        progress_bar=False,
                        refresh=False):
        # Load in the initial D matrix and get a ranking without noise
        D = self.get_D(refresh)
        perfect_ranking = rankingAlg.rank(D)
        
        # Setup the progress bar if needed
        if progress_bar:
            range_iter = tqdm(range(n_trials), ascii=True)
        else:
            range_iter = range(n_trials)
        
        taus = []
        for trial_index in range_iter:
            D_noisy = noiseGenerator.apply_noise(D)
            noisy_ranking = rankingAlg.rank(D_noisy)
            tau = kendall_tau(perfect_ranking, noisy_ranking)
            taus.append(tau)
        
        return taus

######## MAIN, FOR DEBUG ONLY ########
    
def main():
    mcra = MarkovChainRankingAlgorithm()
    mmra = MarkovModifiedRankingAlgorithm()
    
    testmatrix = PerfectBinarySource(10)
    perf = testmatrix.init_D()
    
    eloTournament = SynthELOTournamentSource(10, 5, 260, 800)
    eloMatrix = eloTournament.init_D()
    eloTournament5 = SynthELOTournamentSource(5, 5, 260, 800)
    eloMatrix5 = eloTournament5.init_D()
    eloTournament2 = SynthELOTournamentSource(3, 5, 260, 800)
    eloMatrix2 = eloTournament2.init_D()
    eloTournament4 = SynthELOTournamentSource(4, 5, 260, 800)
    eloMatrix4 = eloTournament4.init_D()
    
    fivegood = np.array([[0,1,1,1,0],
                        [0,0,1,1,1],
                        [0,1,0,1,1],
                        [0,0,0,0,1],
                        [3,3,3,3,0]])
    
    np.set_printoptions(formatter={'float': lambda x: str(x)})
    
    
    print(eloMatrix5)
    print(mmra.rank(eloMatrix5))
    print(eloMatrix)
    print(mmra.rank(eloMatrix))
    print(eloMatrix2)
    print(mmra.rank(eloMatrix2))
    print(eloMatrix4)
    print(mmra.rank(eloMatrix4))
    
    

    

#if __name__ == "__main__":
#   main()

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
        # TODO: flip noisePercentage% of relations
        raise NotImplemented("Not Implemented yet")

class DataSource:
    def init_D(self):
        # Child classes should return numpy array
        raise NotImplemented("Don't use the generic DataSource class")
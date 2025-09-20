import numpy as np


class ThompsonStrategyBandit:
    def __init__(self):
        self.strategies = ['HYPERPARAMETERS', 'ALGORITHM_SELECTION', 'feature_selection', 'scaling_method',
                           'encoding_method', 'class_balancing']
        self.alpha = np.ones(len(self.strategies))
        self.beta = np.ones(len(self.strategies))

    def select_strategy(self):
        samples = [np.random.beta(self.alpha[i], self.beta[i]) for i in range(len(self.strategies))]
        return self.strategies[np.argmax(samples)]

    def update(self, strategy, improvement):
        idx = self.strategies.index(strategy)
        if improvement > 0:
            self.alpha[idx] += 1
        else:
            self.beta[idx] += 1
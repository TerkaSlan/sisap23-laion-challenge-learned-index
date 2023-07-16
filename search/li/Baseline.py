from li.Logger import Logger
from li.utils import pairwise_cosine
import time
import numpy as np


class Baseline(Logger):
    """ Baseline class for the learned index challenge,
        used for testing purposes with data volume < 1M.
    """
    def __init__(self):
        pass

    def search(self, queries, data, k=10):
        """ Searches for the k nearest neighbors of the queries in the data in a bruteforce way. """
        s = time.time()
        anns = pairwise_cosine(data, queries).T
        nns = anns.argsort()[:, :k] + 1
        dists = np.sort(anns)[:, :k]
        return dists, nns, time.time() - s

    def build(self, data):
        s = time.time()
        self.logger.info('No build method implemented for baseline.')
        return time.time() - s

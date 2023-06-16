from li.Logger import Logger
from li.utils import pairwise_cosine
import time


class Baseline(Logger):

    def __init__(self):
        self.pq = []

    def search(self, query_idx, queries, data, k=10):
        s = time.time()
        query = queries.iloc[[query_idx]]
        anns = pairwise_cosine(query, data)[0].argsort()[:k]
        return anns, time.time() - s

    def build(self, data):
        s = time.time()
        self.logger('No build method implemented for baseline.')
        return time.time() - s

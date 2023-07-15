import h5py
import pandas as pd
import pickle
from tqdm import tqdm
from li.utils import pairwise_cosine
import time
import logging
import numpy as np
import os
from scipy import sparse


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)-5.5s][%(name)-.20s] %(message)s'
)
LOG = logging.getLogger(__name__)

def increase_max_recursion_limit():
    """ Increases the maximum recursion limit.
    Source: https://stackoverflow.com/a/16248113
    """
    import sys
    import resource
    resource.setrlimit(resource.RLIMIT_STACK, (2**29, -1))
    sys.setrecursionlimit(10**6)


def search(loaded_queries, blmi, data, loaded_gt, clip_data, loaded_queries_seq, n_queries=100, stop_conditions=1_000_000, k=10):
    res_all = []
    for sc in stop_conditions:
        LOG.info(f'Searching for {n_queries} queries with stop condition {sc}')
        for i, query in tqdm(enumerate(loaded_queries[:n_queries])):
            s = time.time()
            pred_leaf_nodes = blmi.search(query, stop_condition=sc)[0]
            object_ids = []
            for pred_leaf_node in pred_leaf_nodes:
                leaf_node = blmi.nodes.get(pred_leaf_node)
                if leaf_node is not None:
                    object_ids.extend(leaf_node.object_ids)
            e = time.time()
            navigation_t = e - s
            s = time.time()
            data_part_sparse = sparse.csr_matrix(clip_data.loc[object_ids])
            query_part_sparse = sparse.csr_matrix(loaded_queries_seq[i])
            dists = pairwise_cosine(query_part_sparse, data_part_sparse)
            e = time.time()
            search_t = e - s
            anns_found = np.intersect1d(
                loaded_gt[i][:10],
                np.array(data.loc[object_ids].index[np.argsort(dists)[:, :k][0]])[:10]
            ).shape
            res_all.append({
                'query': i,
                'stop_condition': sc,
                'anns_found': anns_found[0],
                'navigation_t': navigation_t,
                'search_t': search_t,
            })
    return res_all


if __name__ == '__main__':
    LOG.info(f'Loading pca32 data')
    data_path = '../data/pca32v2/10M/dataset.h5'
    f = h5py.File(data_path, 'r')
    loaded_data = f['pca32'][:, :]
    data = pd.DataFrame(loaded_data)
    data.index += 1

    LOG.info(f'Loading queries')
    base_path = '../data/pca32v2/10M/'
    queries_path = f'{base_path}/query.h5'
    f2 = h5py.File(queries_path, 'r')
    #loaded_queries = f2['emb'][:, :]
    loaded_queries = f2['pca32'][:, :]

    base_path = '../data/clip768v2/10M/'
    queries_path = f'{base_path}/query.h5'
    f2 = h5py.File(queries_path, 'r')
    #loaded_queries = f2['emb'][:, :]
    loaded_queries_seq = f2['emb'][:, :]

    LOG.info(f'Loading clip data')
    data_path = '../data/clip768v2/10M/dataset.h5'
    f = h5py.File(data_path, 'r')
    loaded_clip_data = f['emb'][:, :]
    loaded_clip_data = pd.DataFrame(loaded_clip_data)
    loaded_clip_data.index += 1

    LOG.info(f'Loading GT')
    gt_path = f'../data/groundtruth-10M.h5'
    f3 = h5py.File(gt_path, 'r')
    loaded_gt = f3['knns'][:, :]

    def check_file_exists(filename):
        return os.path.isfile(filename)

    # for all in ../models/
    for f in [f for f in os.listdir('../../models/') if f.endswith('.pkl')]:
        filename = f'../../models/{f}'
        if check_file_exists(f'{filename}-search.csv'):
            LOG.info(f'Skipping {filename} because {filename}-search.csv already exists.')
            continue
        else:
            # create empty csv for now
            pd.DataFrame(list()).to_csv(f'{filename}-search.csv')
    
        LOG.info(f'Loading {filename}...')
        with open(f'{filename}', 'rb') as f:
            blmi = pickle.load(f)

        increase_max_recursion_limit()
        res_all = search(
            loaded_queries, blmi, data, loaded_gt, loaded_clip_data, loaded_queries_seq, n_queries=50, stop_conditions=[1_000_000]
        )

        LOG.info(f'Saving results as `{filename}-search.csv`...')
        df = pd.DataFrame(res_all)
        df.to_csv(f'{filename}-search.csv', index=False)

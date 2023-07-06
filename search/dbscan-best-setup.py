import h5py
import pandas as pd
import pickle
from tqdm import tqdm
from li.utils import pairwise_cosine
import time
import logging
import numpy as np
import os
from li.BaseLMI import cluster_kmeans_faiss, cluster_kmedoids
from li.BaseLMI import BaseLMI
prepare_data_cluster_kmedoids = BaseLMI.prepare_data_cluster_kmedoids
collect_predictions_kmedoids = BaseLMI.collect_predictions_kmedoids
from li.model import NeuralNetwork, data_X_to_torch, data_to_torch
import argparse
from datetime import datetime
from sklearn.cluster import DBSCAN


def get_current_datetime() -> str:
    """
    Formats current datetime into a string.

    Returns
    ----------
    str
        Created datetime.
    """
    return datetime.now().strftime('%Y-%m-%d--%H-%M-%S')


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)-5.5s][%(name)-.20s] %(message)s'
)
LOG = logging.getLogger(__name__)


def run(
    eps,
    min_samples,
    leaf_size,
    p,
    save=True
):

    timestamp = get_current_datetime()
    LOG.info(
        f'Running with: eps={eps}, min_samples={min_samples}, leaf_size={leaf_size},'
        f', p={p}'
    )
    LOG.info(f'Loading data')
    base_data_path = f'/auto/brno12-cerit/nfs4/home/tslaninakova/sisap-challenge/repo/data'
    data_path = f'{base_data_path}/pca32v2/10M/dataset.h5'
    f = h5py.File(data_path, 'r')
    loaded_data = f['pca32'][:, :]
    data = pd.DataFrame(loaded_data)
    data.index += 1

    LOG.info(f'Clustering')
    clusters = DBSCAN(
        eps=eps, min_samples=min_samples, metric='cosine', leaf_size=leaf_size, p=p
    ).fit(data.sample(100_000, random_state=2023).values)

    unique_result = np.unique(clusters.labels_, return_counts=True)

    if save:
        LOG.info(f'Saving results')
        results_path = f'../results-dbscan/'
        os.makedirs(results_path, exist_ok=True)
        two_top = unique_result[1][0] + unique_result[1][1:].max()
        results_file = f'{results_path}/{timestamp}-{str(unique_result[1][0])}-{str(unique_result[1][1:].max())}-{two_top}.csv'
        results = pd.DataFrame({
            'eps': [eps],
            'min_samples': [min_samples],
            'leaf_size': [leaf_size],
            'p': [p],
            'n_clusters': [unique_result[0].shape[0]],
            'without_cluster_size': [unique_result[1][0]],
        })
        results.to_csv(results_file, index=False)
        LOG.info(f'Results saved to {results_file}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--leaf-size",
        default=10,
        type=int
    )
    parser.add_argument(
        "--eps",
        default=0.1,
        type=float
    )
    parser.add_argument(
        "--min-samples",
        default=10,
        type=int
    )
    parser.add_argument(
        "--p",
        default=0.001,
        type=float
    )

    args = parser.parse_args()
    run(
        eps=args.eps,
        min_samples=args.min_samples,
        leaf_size=args.leaf_size,
        p=args.p
    )
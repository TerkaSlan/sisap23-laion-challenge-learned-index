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
from sklearn import preprocessing

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
    kind,
    key,
    size='100K',
    k=10,
    n_categories=10,
    epochs=100,
    lr=0.1,
    model_type=None,
    save=True
):

    timestamp = get_current_datetime()
    LOG.info(
        f'Running with: kind={kind}, key={key}, size={size},'
        f' n_categories={n_categories}, epochs={epochs}'
        f', model_type={model_type}'
    )

    size_mapping = {
        "100K": 100_000,
        "300K": 300_000,
        "10M": 10_000_000
    }

    kind_mapping = {
        "clip768v2": 768,
        "pca32v2": 32,
        "pca96v2": 96
    }

    emb_mapping = {
        "clip768v2": 'emb',
        "pca32v2": 'pca32',
        "pca96v2": 'pca96'
    }

    base_data_path = f'/auto/brno12-cerit/nfs4/home/tslaninakova/sisap-challenge/repo/data'

    LOG.info(f'Loading {kind} data')
    data_path = f'{base_data_path}/{kind}/{size}/dataset.h5'
    f = h5py.File(data_path, 'r')
    loaded_data = f[emb_mapping[kind]][:, :]

    loaded_data = preprocessing.normalize(loaded_data)


    data = pd.DataFrame(loaded_data)
    data.index += 1

    LOG.info(f'Loading queries')
    base_path = f'{base_data_path}/{kind}/{size}/'
    queries_path = f'{base_path}/query.h5'
    f2 = h5py.File(queries_path, 'r')
    #loaded_queries = f2['emb'][:, :]
    loaded_queries = f2[emb_mapping[kind]][:, :]

    loaded_queries = preprocessing.normalize(loaded_queries)


    search_kind = 'clip768v2'

    LOG.info(f'Loading GT')
    gt_path = f'{base_data_path}/groundtruth-{size}.h5'
    f3 = h5py.File(gt_path, 'r')
    loaded_gt = f3['knns'][:, :]

    LOG.info(f'Clustering with K-Means and data of shape: {data.shape}')
    _, basic_clustering = cluster_kmeans_faiss(data, n_clusters=n_categories)
    #data = data.sample(frac=0.1)
    #LOG.info(f'Clustering with K-Medoids and data of shape: {data.shape}')
    """
    (
        data_part,
        data_predict,
        data_part_index,
        data_all
    ) = prepare_data_cluster_kmedoids(
        data.values, n_categories, data.index.values.tolist()
    )
    labels = cluster_kmedoids(data_part, n_clusters=n_categories)
    """
    LOG.info(f'Instantiating NN')
    nn = NeuralNetwork(
        input_dim=data.shape[1], output_dim=n_categories, lr=lr, model_type=model_type
    )
    prev = 0
    for partition in np.arange(100_000, 10_100_000, 100_000):
        data_x, data_y = data_to_torch(loaded_data[prev:partition], basic_clustering[prev:partition])
        #data_x, data_y = data_to_torch(data_part, labels)
        LOG.info(f'Starting training')
        losses = nn.train(data_x, data_y, epochs=epochs, logger=LOG)
        """
        pred_positions = nn.predict(data_x)
        if data_predict is not None:
            pred_positions = collect_predictions_kmedoids(
                nn,
                data_predict,
                pred_positions,
                data_part_index,
                data_all
            )
            data['category'] = pred_positions
        """
        data['category'] = basic_clustering

        LOG.info(f'Evaluating')
        res = nn.predict_proba(data_X_to_torch(loaded_queries))

        n_cats_covered = []
        n_objects_covered = []

        for i in tqdm(range(10_000)):
            overall_sum = 0
            overall_objects_sum = 0
            argsorted = np.argsort(res[0][i])[::-1]
            idx = 0
            while overall_sum < 9:
                overall_sum += np.sum(data.loc[loaded_gt[i][:10]].category == argsorted[idx])
                overall_objects_sum += np.sum(basic_clustering == argsorted[idx])
                #overall_objects_sum += np.sum(pred_positions == argsorted[idx])
                idx += 1
            n_cats_covered.append(idx)
            n_objects_covered.append(overall_objects_sum)

        mean_cats_covered = np.mean(np.array(n_cats_covered))
        mean_objects_covered = np.mean(np.array(n_objects_covered))
        LOG.info(f'mean_cats_covered={mean_cats_covered}, mean_objects_covered={mean_objects_covered}')

        if save:
            def save_nn(nn, timestamp):
                nn_path = f'../models/{timestamp}.pkl'
                with open(nn_path, 'wb') as f:
                    pickle.dump(nn, f)
                LOG.info(f'NN saved to {nn_path}')

            LOG.info(f'Saving results')
            results_path = f'../results/'
            os.makedirs(results_path, exist_ok=True)
            results_file = f'{results_path}/{timestamp}-{mean_objects_covered}-{prev}-{partition}.csv'
            results = pd.DataFrame({
                'kind': [kind],
                'key': [key],
                'size': [size],
                'n_categories': [n_categories],
                'epochs': [epochs],
                'lr': [lr],
                'model_type': [model_type],
                'mean_cats_covered': [mean_cats_covered],
                'mean_objects_covered': [mean_objects_covered],
            })
            results.to_csv(results_file, index=False)
            LOG.info(f'Results saved to {results_file}')
        prev = partition

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="pca32v2"
    )
    parser.add_argument(
        "--emb",
        default="emb"
    )
    parser.add_argument(
        "--size",
        default="10M"
    )
    parser.add_argument(
        "--epochs",
        default=10,
        type=int
    )
    parser.add_argument(
        "--lr",
        default=0.1,
        type=float
    )
    parser.add_argument(
        "--n-categories",
        default=100,
        type=int
    )
    parser.add_argument(
        "--model-type",
        default='MLP'
    )

    args = parser.parse_args()
    run(
        kind=args.dataset,
        key=args.emb,
        size=args.size,
        epochs=args.epochs,
        lr=args.lr,
        n_categories=args.n_categories,
        model_type=args.model_type
    )
import argparse
import h5py
import numpy as np
import pandas as pd
from sklearn import preprocessing
import os
import time
import logging
from li.Baseline import Baseline
from li.LearnedIndex import LearnedIndex
from li.utils import save_as_pickle, prepare, store_results
from li.model import data_X_to_torch

np.random.seed(2023)

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
    index_type='baseline',
    n_buckets_perc=None,
    n_categories=None,
    epochs=100,
    model_type='MLP',
    lr=0.1,
    preprocess=False,
    save=False
):
    n_buckets_perc = [int((b/100)*n_categories) for b in n_buckets_perc]
    n_buckets_perc = list(set([b for b in n_buckets_perc if b > 0]))
    print(n_buckets_perc, n_categories)
    LOG.info(
        f'Running with: kind={kind}, key={key}, size={size},'
        f' n_buckets_perc={n_buckets_perc}, n_categories={n_categories},'
        f' epochs={epochs}, lr={lr}, model_type={model_type},'
        f' preprocess={preprocess}, save={save}'
    )

    prepare(kind, size)

    data = np.array(h5py.File(os.path.join("data", kind, size, "dataset.h5"), "r")[key])
    queries = np.array(h5py.File(os.path.join("data", kind, size, "query.h5"), "r")[key])
    if preprocess:
        data = preprocessing.normalize(data)
        queries = preprocessing.normalize(queries)

    n, d = data.shape
    LOG.info(f'Loaded downloaded data, shape: n={n}, d={d}')
    LOG.info(f'Loaded downloaded queries, shape: queries={queries.shape}')

    if index_type == 'baseline':
        baseline = Baseline()
        build_t = baseline.build(data)
        LOG.info(f'Build time: {build_t}')
        dists, nns, search_t = baseline.search(
            queries=queries,
            data=data,
            k=k
        )
        identifier = 'li-baseline'
    elif index_type == 'learned-index':
        s = time.time()
        # ---- data to pd.DataFrame ---- #
        data = pd.DataFrame(data)
        data.index += 1

        kind_search = 'clip768v2'
        key_search = 'emb'
        if kind != kind_search:
            LOG.info('Loading data to be used in search')
            prepare(kind_search, size)
            data_search = np.array(
                h5py.File(os.path.join("data", kind_search, size, "dataset.h5"), "r")[key_search]
            )
            # ---- data_search to pd.DataFrame ---- #
            data_search = pd.DataFrame(data_search)
            data_search.index += 1
            queries_search = np.array(
                h5py.File(os.path.join("data", kind_search, size, "query.h5"), "r")[key_search]
            )
            n, d = data_search.shape
            LOG.info(f'Loaded downloaded data, shape: n={n}, d={d}')
            LOG.info(f'Loaded downloaded queries, shape: queries={queries_search.shape}')
        else:
            data_search = data
            queries_search = queries
        # ---- instantiate the index ---- #
        li = LearnedIndex()
        # ---- build the index ---- #
        pred_categories, build_t = li.build(
            data,
            n_categories=n_categories,
            epochs=epochs,
            lr=lr
        )
        e = time.time()
        LOG.info(f'Pure build time: {build_t}')
        LOG.info(f'Overall build time: {e-s}')

        if save:
            save_filename = (
                f'./models/{kind}-{size}-ep={epochs}-lr={lr}-cat={n_categories}'
                f'-model={model_type}-prep={preprocess}'
            )
            LOG.info(f'Saving as {save_filename}')
            save_as_pickle(f'{save_filename}.pkl', li)

        for bucket in n_buckets_perc:
            s = time.time()
            LOG.info(f'Searching with {bucket} buckets')
            if bucket > 1:
                dists, nns = li.search(
                    data_navigation=data,
                    queries_navigation=queries,
                    data_search=data_search,
                    queries_search=queries_search,
                    pred_categories=pred_categories,
                    n_buckets=bucket,
                    k=k,
                    use_threshold=True
                )
            else:
                _, pred_proba_categories = li.model.predict_proba(
                    data_X_to_torch(queries)
                )
                data['category'] = pred_categories
                dists, nns = li.search_single(
                    data_navigation=data,
                    data_search=data_search,
                    queries_search=queries_search,
                    pred_categories=pred_proba_categories[:, 0],
                    k=k
                )
            search_t = time.time() - s
            LOG.info(f'Search time: {search_t}')

            short_identifier = 'learned-index'
            identifier = (
                f'{short_identifier}-{kind}-{size}-ep={epochs}-lr={lr}-cat='
                f'{n_categories}-model={model_type}-buck={bucket}'
            )
            store_results(
                os.path.join(
                    "result/",
                    kind,
                    size,
                    f"{identifier}.h5"
                ),
                short_identifier.capitalize(),
                kind,
                dists,
                nns,
                build_t,
                search_t,
                identifier,
                size
            )
    else:
        raise Exception(f'Unknown index type: {index_type}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="pca96v2"
    )
    parser.add_argument(
        "--emb",
        default="pca96"
    )
    parser.add_argument(
        "--size",
        default="10M"
    )
    parser.add_argument(
        "--k",
        default=10,
    )
    parser.add_argument(
        "--n-categories",
        default=122,
        type=int,
        help='Number of categories (= buckets) to create'
    )
    parser.add_argument(
        "--epochs",
        default=210,
        type=int,
        help='Number of epochs to train the model for'
    )
    parser.add_argument(
        "--model-type",
        default='MLP-3',
        type=str,
        help='Model type to use for the learned index'
    )
    parser.add_argument(
        "--lr",
        default=0.009,
        type=float,
        help='Learning rate'
    )
    parser.add_argument(
        '-bp',
        '--buckets-perc',
        nargs='+',
        default=[4],
        help='Percentage of the most similar buckets to look for the candidate answer in'
    )
    parser.add_argument(
        "--preprocess",
        default=True,
        type=bool,
        help='Whether to normalize the data or not'
    )
    parser.add_argument(
        "--save",
        default=False,
        type=bool,
        help='Whether to save the model or not'
    )
    args = parser.parse_args()

    assert args.size in ['100K', '300K', '10M', '30M', '100M']

    run(
        args.dataset,
        args.emb,
        args.size,
        args.k,
        'learned-index',
        [int(b) for b in args.buckets_perc],
        args.n_categories,
        args.epochs,
        args.model_type,
        args.lr,
        args.preprocess,
        args.save
    )

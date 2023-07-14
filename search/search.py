import argparse
import h5py
import numpy as np
import pandas as pd
from sklearn import preprocessing
import os
import time
from pathlib import Path
from urllib.request import urlretrieve
import logging
from li.Baseline import Baseline
from li.LearnedIndex import LearnedIndex
from li.utils import save_as_pickle
from li.model import data_X_to_torch

np.random.seed(2023)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)-5.5s][%(name)-.20s] %(message)s'
)
LOG = logging.getLogger(__name__)


def download(src, dst):
    if not os.path.exists(dst):
        os.makedirs(Path(dst).parent, exist_ok=True)
        LOG.info('downloading %s -> %s...' % (src, dst))
        urlretrieve(src, dst)


def prepare(kind, size):
    url = "https://sisap-23-challenge.s3.amazonaws.com/SISAP23-Challenge"
    task = {
        "query": f"{url}/public-queries-10k-{kind}.h5",
        "dataset": f"{url}/laion2B-en-{kind}-n={size}.h5",
    }

    for version, url in task.items():
        target_path = os.path.join("data", kind, size, f"{version}.h5")
        download(url, target_path)
        assert os.path.exists(target_path), f"Failed to download {url}"


def store_results(dst, algo, kind, dists, anns, buildtime, querytime, params, size):
    os.makedirs(Path(dst).parent, exist_ok=True)
    f = h5py.File(dst, 'w')
    f.attrs['algo'] = algo
    f.attrs['data'] = kind
    f.attrs['buildtime'] = buildtime
    f.attrs['querytime'] = querytime
    f.attrs['size'] = size
    f.attrs['params'] = params
    f.create_dataset('knns', anns.shape, dtype=anns.dtype)[:] = anns
    f.create_dataset('dists', dists.shape, dtype=dists.dtype)[:] = dists
    f.close()


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
        dists, nns, search_t, inference_t, search_single_t, seq_search_t = baseline.search(
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
            save_filename = f'./models/{kind}-{size}-ep={epochs}-lr={lr}-cat={n_categories}-model={model_type}-prep={preprocess}-{os.environ["PBS_JOBID"]}'
            LOG.info(f'Saving as {save_filename}')
            save_as_pickle(f'{save_filename}.pkl', li)

        for bucket in n_buckets_perc:
            LOG.info(f'Searching with {bucket} buckets')
            if bucket > 1:
                dists, nns, search_t, inference_t, search_single_t, seq_search_t, pure_seq_search_t, sort_t = li.search(
                    data_navigation=data,
                    queries_navigation=queries,
                    data_search=data_search,
                    queries_search=queries_search,
                    pred_categories=pred_categories,
                    n_buckets=bucket,
                    k=k,
                    use_threshold=True
                )
                LOG.info('Inference time: %s', inference_t)
                LOG.info('Search time: %s', search_t)
                LOG.info('Search single time: %s', search_single_t)
                LOG.info('Sequential search time: %s', seq_search_t)
                LOG.info('Pure sequential search time: %s', pure_seq_search_t)
                LOG.info('Sort time: %s', sort_t)
            else:
                s = time.time()
                _, pred_proba_categories = li.model.predict_proba(
                    data_X_to_torch(queries)
                )
                data['category'] = pred_categories
                dists, nns, t_all, t_pairwise, t_pure_pairwise, t_sort = li.search_single(
                    data_navigation=data,
                    data_search=data_search,
                    queries_search=queries_search,
                    pred_categories=pred_proba_categories[:, 0],
                    k=k
                )
                search_t = time.time() - s
                LOG.info(f't_all: {t_all}')
                LOG.info(f't_pairwise: {t_pairwise}')
                LOG.info(f't_pure_pairwise: {t_pure_pairwise}')
                LOG.info(f't_sort: {t_sort}')

            short_identifier = 'learned-index'
            identifier = f'{short_identifier}-{kind}-{size}-ep={epochs}-lr={lr}-cat={n_categories}-model={model_type}-buck={bucket}-{os.environ["PBS_JOBID"]}'
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
        default="pca32v2"
    )
    parser.add_argument(
        "--emb",
        default="pca32"
    )
    parser.add_argument(
        "--size",
        default="100K"
    )
    parser.add_argument(
        "--k",
        default=10,
    )
    parser.add_argument(
        "--n-categories",
        default=200,
        type=int
    )
    parser.add_argument(
        "--epochs",
        default=50,
        type=int
    )
    parser.add_argument(
        "--model-type",
        default='MLP',
        type=str
    )
    parser.add_argument(
        "--lr",
        default=0.1,
        type=float
    )
    parser.add_argument(
        '-b',
        '--n-buckets',
        nargs='+',
        default=[2, 3, 4] #, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30]
    )
    parser.add_argument(
        '-bp',
        '--buckets-perc',
        nargs='+',
        default=[1, 3, 5, 10, 20] #, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30]
    )
    parser.add_argument(
        "--preprocess",
        default=False,
        type=bool
    )
    parser.add_argument(
        "--save",
        default=True,
        type=bool
    )
    args = parser.parse_args()

    assert args.size in ["100K", "300K", "10M", "30M", "100M"]

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

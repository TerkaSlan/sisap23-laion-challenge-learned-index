import argparse
import h5py
import numpy as np
import pandas as pd
import os
import time
from pathlib import Path
from urllib.request import urlretrieve
import logging
from li.Baseline import Baseline
from li.LearnedIndex import LearnedIndex


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
    n_buckets=None,
    n_categories=None,
    epochs=100,
    lr=0.1
):
    LOG.info(
        f'Running with: kind={kind}, key={key}, size={size}'
        f'n_buckets={n_buckets}, n_categories={n_categories}'
    )

    prepare(kind, size)

    data = np.array(h5py.File(os.path.join("data", kind, size, "dataset.h5"), "r")[key])
    queries = np.array(h5py.File(os.path.join("data", kind, size, "query.h5"), "r")[key])
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

        dists, nns, search_t = li.search(
            data_navigation=data,
            queries_navigation=queries,
            data_search=data_search,
            queries_search=queries_search,
            pred_categories=pred_categories,
            n_buckets=n_buckets,
            k=k
        )
        LOG.info(f'Search time: {search_t}')
        identifier = 'learned-index'
        store_results(
            os.path.join(
                "result/",
                kind,
                size,
                f"{identifier}.h5"
            ),
            identifier.capitalize(),
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
        default=100,
        type=int
    )
    parser.add_argument(
        "--epochs",
        default=50,
        type=int
    )
    parser.add_argument(
        "--lr",
        default=0.1,
        type=int
    )
    parser.add_argument(
        "--n-buckets",
        default=10,
        type=int
    )
    args = parser.parse_args()

    assert args.size in ["100K", "300K", "10M", "30M", "100M"]

    run(
        args.dataset,
        args.emb,
        args.size,
        args.k,
        'learned-index',
        args.n_buckets,
        args.n_categories,
        args.epochs,
        args.lr
    )

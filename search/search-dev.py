import argparse
import h5py
import numpy as np
import os
from pathlib import Path
from urllib.request import urlretrieve
import logging
from li.Baseline import Baseline
from li.LearnedIndex10M import LearnedIndex
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
import torch
from tqdm import tqdm
import gc

def pairwise_cosine(x, y):
    r = cosine_similarity(x, y)
    return r


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


def run(kind, key, size="100K", k=10, index_type='baseline', n_buckets=None, n_categories=None, save=True):
    LOG.info(f'Running with: kind={kind}, key={key}, size={size}, n_buckets={n_buckets}, n_categories={n_categories}')

    prepare(kind, size)

    if index_type == 'baseline':
        data = np.array(h5py.File(os.path.join("data", kind, size, "dataset.h5"), "r")[key])
        queries = np.array(h5py.File(os.path.join("data", kind, size, "query.h5"), "r")[key])
        n, d = data.shape
        LOG.info(f'Loaded downloaded data, shape: n={n}, d={d}')
        LOG.info(f'Loaded downloaded queries, shape: queries={queries.shape}')
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
        size_mapping = {
            "100K": 100_000,
            "300K": 300_000,
            "10M": 10_000_000
        }

        kind_mapping = {
            "clip768v2": 768,
            "pca32v2": 32
        }


        f = h5py.File(os.path.join("data", kind, size, "dataset.h5"), "r")
        LOG.info(f'Instantiating LearnedIndex with dataset_shape={size_mapping[size]}, {kind_mapping[kind]}, n_categories={n_categories}')
        li = LearnedIndex(
            dataset_shape=(size_mapping[size], kind_mapping[kind]),
            n_categories=n_categories
        )
        model, build_t = li.build(f[key])
        if save:
            save_dir = f'../models/{kind}-{key}-{size}' 
            LOG.info(f'Saving into {save_dir}')
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), f'{save_dir}-best-model.pt')

        LOG.info(f'Build time: {build_t}')
        queries = np.array(h5py.File(os.path.join("data", kind, size, "query.h5"), "r")[key])
        LOG.info(f'Loaded downloaded queries, shape: queries={queries.shape}')
        if type(n_buckets) != list:
            n_buckets = [n_buckets]
        for b in n_buckets:
            dists, nns, search_t = li.search(
                queries=queries,
                k=k,
                n_buckets=b
            )
            identifier = f'li-index-{b}'
            LOG.info(f'dists={dists.shape}, nns={nns.shape} search time: {search_t}')

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
        default="pca32v2" # clip768v2
    )
    parser.add_argument(
        "--emb",
        default="pca32" # emb
    )
    parser.add_argument(
        "--size",
        default="10M"
    )
    parser.add_argument(
        "--k",
        default=10,
    )
    """
    parser.add_argument(
        "--n-buckets",
        default=1,
        type=int
    )
    """
    parser.add_argument(
        '-b',
        '--n-buckets',
        nargs='+',
        default=[1, 2, 5, 10, 15, 20, 30, 50, 100, 200, 300, 500]
    )

    parser.add_argument(
        "--n-categories",
        default=1000,
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
        save=True
    )

    #run("clip768v2", "emb", args.size, args.k, 'learned-index', args.n_buckets)
    """
    run(
        "pca32v2",
        "pca32",
        '100K',
        args.k,
        'learned-index',
        args.n_buckets,
        args.n_categories,
        save=True
    )

    run(
        "clip768v2",
        "emb",
        '100K',
        args.k,
        'learned-index',
        args.n_buckets,
        args.n_categories,
        save=True
    )

    run(
        "clip768v2",
        "emb",
        '300K',
        args.k,
        'learned-index',
        args.n_buckets,
        args.n_categories,
        save=True
    )

    run(
        "clip768v2",
        "emb",
        '10M',
        args.k,
        'learned-index',
        args.n_buckets,
        args.n_categories,
        save=True
    )
    """
import argparse
import h5py
import numpy as np
import os
from pathlib import Path
from urllib.request import urlretrieve
import logging
from li.Baseline import Baseline


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


def run(kind, key, size="100K", k=30):
    LOG.info(f'Running: kind={kind}, key={key}, size={size}')

    prepare(kind, size)

    data = np.array(h5py.File(os.path.join("data", kind, size, "dataset.h5"), "r")[key])
    queries = np.array(h5py.File(os.path.join("data", kind, size, "query.h5"), "r")[key])
    n, d = data.shape
    LOG.info(f'Loading downloaded data, shape: n={n}, d={d}')
    LOG.info(f'Loading downloaded queries, shape: queries={queries.shape}')

    baseline = Baseline()
    build_t = baseline.build(data)
    LOG.info(f'Build time: {build_t}')
    # ef search(self, query_idx, queries, data, k=10):
    dists, nns, search_t = baseline.search(
        queries=queries,
        data=data,
        k=10
    )
    LOG.info(f'dists={dists.shape}, nns={nns.shape} serach time: {search_t}')

    identifier = 'li-baseline'
    store_results(
        os.path.join(
            "result/",
            kind,
            size,
            f"{identifier}.h5"
        ),
        "LI-BASELINE", kind, dists, nns, build_t, search_t, identifier, size
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size",
        default="100K"
    )
    parser.add_argument(
        "--k",
        default=30,
    )

    args = parser.parse_args()

    assert args.size in ["100K", "300K", "10M", "30M", "100M"]

    run("clip768v2", "emb", args.size, args.k)

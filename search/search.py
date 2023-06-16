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


'''
def store_results(dst, algo, kind, D, I, buildtime, querytime, params, size):
    os.makedirs(Path(dst).parent, exist_ok=True)
    f = h5py.File(dst, 'w')
    f.attrs['algo'] = algo
    f.attrs['data'] = kind
    f.attrs['buildtime'] = buildtime
    f.attrs['querytime'] = querytime
    f.attrs['size'] = size
    f.attrs['params'] = params
    f.create_dataset('knns', I.shape, dtype=I.dtype)[:] = I
    f.create_dataset('dists', D.shape, dtype=D.dtype)[:] = D
    f.close()
'''


def run(kind, key, size="100K", k=30):
    LOG.info(f'Running: kind={kind}, key={key}, size={size}')

    prepare(kind, size)

    data = np.array(h5py.File(os.path.join("data", kind, size, "dataset.h5"), "r")[key])
    queries = np.array(h5py.File(os.path.join("data", kind, size, "query.h5"), "r")[key])
    n, d = data.shape
    LOG.info(f'Loaded downloaded data, shape: n={n}, d={d}')
    LOG.info(f'Loaded downloaded queries, shape: queries={queries.shape}')

    baseline = Baseline()
    build_t = baseline.build(data)
    LOG.info(f'Build time: {build_t}')
    # ef search(self, query_idx, queries, data, k=10):
    results, search_t = baseline.search(
        query_idx=0,
        queries=queries,
        data=data,
        k=10
    )
    LOG.info(f'Results shape: {results.shape}, serach time: {search_t}')

    '''
    nlist = 128 # number of clusters/centroids to build the IVF from

    if kind.startswith("pca"):
        index_identifier = f"IVF{nlist},Flat"
        index = faiss.index_factory(d, index_identifier)
    elif kind.startswith("hamming"):
        index_identifier = f"BIVF{nlist},Flat" # use binary IVF index
        d = 64 * d # one chunk contains 64 bits
        index = faiss.index_binary_factory(d, index_identifier)
        # create view to interpret original uint64 as 8 chunks of uint8
        data = np.array(data).view(dtype="uint8")
        queries = np.array(queries).view(dtype="uint8")
    else:
        # if kind == "clip768":
        # convert vectors from float16 to float32
        # normalize vectors
        # dot product / angle as distance (1-cosine)
        raise Exception(f"unsupported input type {kind}")

    print(f"Training index on {data.shape}")
    start = time.time()
    index.train(data)
    index.add(data)
    elapsed_build = time.time() - start
    print(f"Done training in {elapsed_build}s.")
    assert index.is_trained

    for nprobe in [1, 2, 5, 10, 20, 50, 100]:
        print(f"Starting search on {queries.shape} with nprobe={nprobe}")
        start = time.time()
        index.nprobe = nprobe
        D, I = index.search(queries, k)
        elapsed_search = time.time() - start
        print(f"Done searching in {elapsed_search}s.")

        I = I + 1 # FAISS is 0-indexed, groundtruth is 1-indexed

        identifier = f"index=({index_identifier}),query=(nprobe={nprobe})"

        store_results(
            os.path.join(
                "result/",
                kind,
                size,
                f"{identifier}.h5"
            ),
            "faissIVF", kind, D, I, elapsed_build, elapsed_search, identifier, size
        )
    '''


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

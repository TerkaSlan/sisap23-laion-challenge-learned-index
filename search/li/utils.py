from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
import os
from pathlib import Path
import h5py
from urllib.request import urlretrieve


def pairwise_cosine(x, y):
    return 1-cosine_similarity(x, y)


def pairwise_cosine_threshold(x, y, threshold, cat_idxs, k=10):
    """ Evaluates the distances between queries and the objects,
    disregards the objects that are further from each query than what is specified in `threshold`
    """
    s = time.time()
    result = 1-cosine_similarity(x, y)
    t_pure_seq_search = time.time() - s
    # create an array of consistent shapes
    thresh_consistent = np.repeat(threshold[cat_idxs, np.newaxis], result.shape[1], 1)
    relevant_dists = np.where(result < thresh_consistent)
    # filter the relevant object ids
    try:
        relevant_object_ids = np.unique(relevant_dists[1])
        max_idx = relevant_object_ids.shape[0]
        if max_idx == 0:
            return None, t_pure_seq_search
    except ValueError:
        # There is no distance below the threshold, we can return
        return None, t_pure_seq_search
    max_idx = max_idx if max_idx > k else k
    # output array filled with some large value
    output_arr = np.full(shape=(result.shape[0], max_idx), fill_value=10_000, dtype=float)
    # create indexes to store the relevant distances
    mapping = dict(zip(relevant_object_ids, np.arange(relevant_object_ids.shape[0])))
    # tried also with np.vectorize, wasn't faster
    output_arr_2nd_dim = np.array([mapping[x] for x in relevant_dists[1]])
    to_be_added = result[relevant_dists[0], relevant_dists[1]]
    # populate the output array
    output_arr[relevant_dists[0], output_arr_2nd_dim] = to_be_added
    return output_arr, relevant_object_ids, t_pure_seq_search


def save_as_pickle(filename: str, obj):
    """
    Saves an object as a pickle file.
    Expects that the directory with the file exists.

    Parameters
    ----------
    filename : str
        Path to the file to write to.
    obj: object
        The object to save.
    """
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def download(src, dst):
    # copied form https://github.com/sisap-challenges/sisap23-laion-challenge-faiss-example
    if not os.path.exists(dst):
        os.makedirs(Path(dst).parent, exist_ok=True)
        print('downloading %s -> %s...' % (src, dst))
        urlretrieve(src, dst)


def prepare(kind, size):
    # copied form https://github.com/sisap-challenges/sisap23-laion-challenge-faiss-example
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
    # copied form https://github.com/sisap-challenges/sisap23-laion-challenge-faiss-example
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

import argparse
import h5py
import numpy as np
import os
from pathlib import Path
from urllib.request import urlretrieve
import logging
from li.Baseline import Baseline
import pandas as pd
#from li.LearnedIndex10M import LearnedIndex
from li.BulkLMI import BulkLMI
from li.utils import pairwise_cosine, save_as_pickle
import torch
from datetime import datetime
import time
from tqdm import tqdm
from sklearn import preprocessing
from li.BaseLMI import cluster_kmeans_faiss
from li.model import NeuralNetwork, data_X_to_torch, data_to_torch, LIDataset_Single


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
    lr=0.1,
    model_type=None,
    perc_train=0.5,
    stop_condition=100_000,
    batch_size=256,
    n_levels=2,
    save=True
):
    LOG.info(
        f'Running with: kind={kind}, key={key}, size={size}, stop_condition={stop_condition}'
        f' n_buckets={n_buckets}, n_categories={n_categories}, epochs={epochs}, n_levels={n_levels}'
        f', batch_size={batch_size}, model_type={model_type}, perc_train={perc_train}, lr={lr}'
    )

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
        '''
        elif index_type == 'learned-index':
            timestamp = get_current_datetime()
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

            f = h5py.File(os.path.join("data", kind, size, "dataset.h5"), "r")
            LOG.info(
                f'Instantiating LearnedIndex with dataset_shape={size_mapping[size]}'
                f', {kind_mapping[kind]}, n_categories={n_categories}'
            )
            li = LearnedIndex(
                dataset_shape=(size_mapping[size], kind_mapping[kind]),
                n_categories=n_categories
            )
            model, build_t = li.build(f[key], epochs, lr=lr, model_size=model_size)
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
                identifier = f'li-index-{b}-{timestamp}'
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
        '''
    elif index_type == 'clip-simple':
        assert kind == 'clip768v2'
        timestamp = get_current_datetime()
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
            "pca32v2": 'pca32'
        }
        LOG.info('Preparing -- loading data for training')
        f = h5py.File(os.path.join("data", kind, size, "dataset.h5"), "r")
        loaded_data_build = f[emb_mapping[kind]][:, :]

        s = time.time()
        loaded_data_build = preprocessing.normalize(loaded_data_build)
        data = pd.DataFrame(loaded_data_build)
        data.index += 1

        LOG.info(f'Clustering into {n_categories}')
        kmeans, result = cluster_kmeans_faiss(data, n_clusters=n_categories)
        LOG.info(f'Preparing NN')

        nn = NeuralNetwork(
            input_dim=data.shape[1], output_dim=n_categories, lr=lr, model_type=model_type
        )

        #data['category'] = result
        LOG.info(f'np.unique k-means partitioning: {np.unique(result, return_counts=True)}')
        dataset = LIDataset_Single(data, result)
                
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(data.index.values.tolist())
        )
        LOG.info(f'Training for {epochs} epochs')

        losses = nn.train_batch(train_loader, epochs=epochs, logger=LOG)
        if save:
            save_dir = f'../models/{os.environ["PBS_JOBID"]}{kind}-{key}-{size}-{n_categories}-{perc_train}-{epochs}-{lr}-{model_type}.pkl'
            LOG.info(f'Saving as {save_dir}')
            #Path(save_dir).mkdir(parents=True, exist_ok=True)
            save_as_pickle(f'{save_dir}.pkl', nn)

        LOG.info(f'losses: {losses}')
        build_t = time.time() - s
        LOG.info('Preparing -- loading queries for navigation')
        f = h5py.File(os.path.join("data", kind, size, "query.h5"), "r")
        queries = f[emb_mapping[kind]][:, :]
        queries = preprocessing.normalize(queries)
        LOG.info('Starting search')
        s = time.time()
        #res_all = []
        anns = []
        dists_all = []
        data['category_nn'] = nn.predict(data_X_to_torch(data))
        probs, classes = nn.predict_proba(data_X_to_torch(queries))

        nns = np.zeros((queries.shape[0], k), dtype=np.uint32)
        dists = np.zeros((queries.shape[0], k), dtype=np.float32)
        for class_ in np.unique(classes[:, 0]):
            cat_idxs = np.where(classes[:, 0] == class_)[0]
            bucket_obj_indexes = data.query(f'category_nn == {class_}', engine='python').index
            seq_search_dists = pairwise_cosine(queries[cat_idxs], data.loc[bucket_obj_indexes].drop(['category_nn'], axis=1))
            ann_relative = seq_search_dists.argsort()[:, :k]
            nns[cat_idxs] = np.array(bucket_obj_indexes)[ann_relative] + 1
            dists[cat_idxs] = np.take_along_axis(seq_search_dists, ann_relative, axis=1)
        search_t = time.time() - s
        identifier = f'clip-single-index-build_32-search_clip-{stop_condition}-{timestamp}-{os.environ["PBS_JOBID"]}'
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


    elif index_type == 'learned-index-simple':
        timestamp = get_current_datetime()
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
            "pca32v2": 'pca32'
        }
        search_kind = 'clip768v2'
 
        LOG.info('Preparing -- loading data for training')
        f = h5py.File(os.path.join("data", kind, size, "dataset.h5"), "r")
        loaded_data_build = f[emb_mapping[kind]][:, :]
        
        if loaded_data_build.shape[1] == 768:
            loaded_data_build = preprocessing.normalize(loaded_data_build)

        data = pd.DataFrame(loaded_data_build)
        data.index += 1
        data_s = data.sample(int(loaded_data_build.shape[0]*perc_train), random_state=2023)
        LOG.info(f'Instantiating BaseLMI and inserting the data of size {data_s.shape}')
        s = time.time()
        blmi = BulkLMI()
        blmi.insert(data_s)
        LOG.info('Running deepen (root)')
        info_df = pd.DataFrame([], columns=['op', 'time-taken', 'size', '#-objects'])
        info_df = blmi.deepen(
            blmi.nodes[(0, )],
            n_categories,
            epochs=epochs,
            lr=lr,
            model_type=model_type,
            batch_size=batch_size,
            info_df=info_df
        )
        n_levels -= 1
        while n_levels > 0:
            LOG.info(f'Running deepen (L{n_levels})')
            for i, leaf in enumerate(blmi.get_leaf_nodes_pos()):
                info_df = blmi.deepen(
                    blmi.nodes[leaf],
                    n_categories,
                    epochs=epochs,
                    lr=lr,
                    model_type=model_type,
                    batch_size=batch_size,
                    info_df=info_df
                )
            n_levels -= 1

        data_to_insert = data.loc[np.setdiff1d(data.index, data_s.index)]
        LOG.info(f'Inserting the rest of the data (size {data_to_insert.shape})')
        if data_to_insert.shape[0] > 0:
            blmi.insert(data_to_insert)
        build_t = time.time() - s
        LOG.info(f'Build time: {build_t}')
        if save:
            save_dir = f'../models/{kind}-{key}-{size}-{n_categories}-{perc_train}-{epochs}-{lr}-{model_type}.pkl'
            LOG.info(f'Saving as {save_dir}')
            #Path(save_dir).mkdir(parents=True, exist_ok=True)
            save_as_pickle(f'{save_dir}.pkl', blmi)

        ## build is ended
        LOG.info('Preparing -- loading queries for navigation')
        f = h5py.File(os.path.join("data", kind, size, "query.h5"), "r")
        loaded_queries_navigation = f[emb_mapping[kind]][:, :]
        
        if kind == 'clip768v2':
            loaded_queries_navigation = preprocessing.normalize(loaded_queries_navigation)

        if kind != search_kind:
            LOG.info('Preparing -- loading queries for search')
            f = h5py.File(os.path.join("data", search_kind, size, "query.h5"), "r")
            loaded_queries_seq_search = f[emb_mapping[search_kind]][:, :]
            LOG.info('Preparing -- loading data for search')
            f = h5py.File(os.path.join("data", search_kind, size, "dataset.h5"), "r")
            loaded_data_search = f[emb_mapping[search_kind]][:, :]
            loaded_data_search = pd.DataFrame(loaded_data_search)
            loaded_data_search.index += 1
        else:
            loaded_data_search = loaded_data_build
            loaded_data_search = pd.DataFrame(loaded_data_search)
            loaded_data_search.index += 1
            loaded_queries_seq_search = loaded_queries_navigation

        LOG.info('Starting search')
        s = time.time()
        #res_all = []
        anns = []
        dists_all = []
        for i, query in tqdm(enumerate(loaded_queries_navigation), position=0, leave=True):
            pred_leaf_nodes = blmi.search(query, stop_condition=stop_condition)[0]
            object_ids = []
            for pred_leaf_node in pred_leaf_nodes:
                leaf_node = blmi.nodes.get(pred_leaf_node)
                if leaf_node is not None:
                    object_ids.extend(leaf_node.object_ids)

            dists = pairwise_cosine(
                [loaded_queries_seq_search[i]],
                loaded_data_search.loc[object_ids]
            )
            nns_idxs = np.argsort(dists)[:, :k][0]
            anns.append(data.loc[object_ids].index[nns_idxs].tolist())
            dists_all.append(dists[0][nns_idxs].tolist())
            #res_all.extend(bucket_ids)
        dists = np.array(dists_all)
        nns = np.array(anns)
        search_t = time.time() - s
        identifier = f'blmi-index-build_32-search_clip-{stop_condition}-{timestamp}-{os.environ["PBS_JOBID"]}'
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
        default="pca32v2"
    )
    parser.add_argument(
        "--emb",
        default="emb"
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
        "--n-levels",
        default=1,
        type=int
    )
    parser.add_argument(
        "--stop_condition",
        default=20_000,
        type=int
    )
    parser.add_argument(
        "--batch_size",
        default=256,
        type=int
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
        default=10,
        type=int
    )

    parser.add_argument(
        "--model-type",
        default='MLP'
    )

    parser.add_argument(
        "--perc-train",
        default=1.0,
        type=float
    )

    parser.add_argument(
        "--index-type",
        default='learned-index-simple',
        type=str
    )

    args = parser.parse_args()

    assert args.size in ["100K", "300K", "10M", "30M", "100M"]
    
    """

    kind,
    key,
    size='100K',
    k=10,
    index_type='baseline',
    n_buckets=None,
    n_categories=None,
    epochs=100,
    lr=0.1,
    model_type=None,
    perc_train=0.5,
    stop_condition=100_000,
    batch_size=256,
    n_levels=2,
    save=True
    """
    
    
    run(
        kind=args.dataset,
        key=args.emb,
        size=args.size,
        k=args.k,
        index_type=args.index_type,
        n_buckets=args.n_buckets,
        n_categories=args.n_categories,
        epochs=args.epochs,
        lr=args.lr,
        model_type=args.model_type,
        perc_train=args.perc_train,
        stop_condition=args.stop_condition,
        batch_size=args.batch_size,
        n_levels=args.n_levels,
        save=True
    )

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

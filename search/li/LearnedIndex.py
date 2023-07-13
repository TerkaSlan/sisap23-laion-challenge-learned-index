import numpy as np
from li.Logger import Logger
from li.utils import pairwise_cosine
import time
import torch
import torch.utils.data
from li.model import NeuralNetwork, data_X_to_torch, LIDataset
import faiss
from tqdm import tqdm
import numpy as np

torch.manual_seed(2023)
np.random.seed(2023)

class LearnedIndex(Logger):

    def __init__(self):
        self.pq = []
        self.model = None

    def search(
        self,
        data_navigation,
        queries_navigation,
        data_search,
        queries_search,
        pred_categories,
        n_buckets=1,
        k=10
    ):
        """ Search for k nearest neighbors for each query in queries.

        Parameters
        ----------
        queries : np.array
            Queries to search for.
        data : np.array
            Data to search in.
        n_buckets : int
            Number of most similar buckets to search in.
        k : int
            Number of nearest neighbors to search for.

        Returns
        -------
        dists : np.array
            Array of shape (queries.shape[0], k) with distances to nearest neighbors for each query.
        anns : np.array
            Array of shape (queries.shape[0], k) with nearest neighbors for each query.
        time : float
            Time it took to search.
        """
        assert self.model is not None, 'Model is not trained, call `build` first.'
        s = time.time()
        _, pred_proba_categories = self.model.predict_proba(
            data_X_to_torch(queries_navigation)
        )
        t_inference = time.time() - s
        anns_final = None
        dists_final = None
        # sorts the predictions of a bucket for each query, ordered by lowest probability
        data_navigation['category'] = pred_categories

        # iterates over the predicted buckets starting from the most similar (index -1)
        t_all_buckets = 0
        t_all_pairwise = 0
        t_all_sort = 0
        for bucket in range(n_buckets):
            dists, anns, t_all, t_pairwise, t_sort = self.search_single(
                data_navigation,
                data_search,
                queries_search,
                pred_proba_categories[:, bucket]
            )
            t_all_buckets += t_all
            t_all_pairwise += t_pairwise
            t_all_sort += t_sort
            if anns_final is None:
                anns_final = anns
                dists_final = dists
            else:
                # stacks the results from the previous sorted anns and dists
                # *_final arrays now have shape (queries.shape[0], k*2)
                anns_final = np.hstack((anns_final, anns))
                dists_final = np.hstack((dists_final, dists))
                # gets the sorted indices of the stacked dists
                idx_sorted = dists_final.argsort(kind='stable', axis=1)[:, :k]
                # indexes the final arrays with the sorted indices
                # *_final arrays now have shape (queries.shape[0], k)
                idx = np.ogrid[tuple(map(slice, dists_final.shape))]
                idx[1] = idx_sorted
                dists_final = dists_final[tuple(idx)]
                anns_final = anns_final[tuple(idx)]

                assert anns_final.shape == dists_final.shape == (queries_search.shape[0], k)

        return dists_final, anns_final, time.time() - s, t_inference, t_all_buckets, t_all_pairwise, t_all_sort

    def search_single(
        self,
        data_navigation,
        data_search,
        queries_search,
        pred_categories,
        k=10,
        threshold_dist=None
    ):
        """ Search for k nearest neighbors for each query in queries.

        Parameters
        ----------
        queries : np.array
            Queries to search for.
        data : np.array
            Data to search in.
        k : int
            Number of nearest neighbors to search for.

        Returns
        -------
        anns : np.array
            Array of shape (queries.shape[0], k) with nearest neighbors for each query.
        final_dists_k : np.array
            Array of shape (queries.shape[0], k) with distances to nearest neighbors for each query.
        time : float
            Time it took to search.
        """
        s_all = time.time()
        nns = np.zeros((queries_search.shape[0], k), dtype=np.uint32)
        dists = np.zeros((queries_search.shape[0], k), dtype=np.float32)

        if 'category' in data_search.columns:
            data_search = data_search.drop('category', axis=1, errors='ignore')

        t_pairwise = 0
        t_sort = 0
        for cat, g in tqdm(data_navigation.groupby('category')):
            cat_idxs = np.where(pred_categories == cat)[0]
            bucket_obj_indexes = g.index
            if bucket_obj_indexes.shape[0] != 0 and cat_idxs.shape[0] != 0:
                s = time.time()
                # TODO: Add filter, filter will be different for every query
                # OR pass nns, dists from previous buckets
                seq_search_dists = pairwise_cosine(
                    queries_search[cat_idxs],
                    data_search.loc[bucket_obj_indexes]
                )
                t_pairwise += time.time() - s
                s = time.time()
                ann_relative = seq_search_dists.argsort(kind='quicksort')[
                    :, :k if k < seq_search_dists.shape[1] else seq_search_dists.shape[1]
                ]
                t_sort += time.time() - s
                if bucket_obj_indexes.shape[0] < k:
                    # pad to `k` if needed
                    pad_needed = (k - bucket_obj_indexes.shape[0]) // 2 + 1
                    bucket_obj_indexes = np.pad(np.array(bucket_obj_indexes), pad_needed, 'edge')[:k]
                    ann_relative = np.pad(ann_relative[0], pad_needed, 'edge')[:k].reshape(1, -1)
                    seq_search_dists = np.pad(seq_search_dists[0], pad_needed, 'edge')[:k].reshape(1, -1)
                    _, i = np.unique(seq_search_dists, return_index=True)
                    duplicates_i = np.setdiff1d(np.arange(k), i)
                    # assign a large number such that the duplicated value gets replaced
                    seq_search_dists[0][duplicates_i] = 10_000

                nns[cat_idxs] = np.array(bucket_obj_indexes)[ann_relative]
                dists[cat_idxs] = np.take_along_axis(seq_search_dists, ann_relative, axis=1)
        t_all = time.time() - s_all
        return dists, nns, t_all, t_pairwise, t_sort

    def build(self, data, n_categories=100, epochs=100, lr=0.1, model_type='MLP'):
        """ Build the index.

        Parameters
        ----------
        data : np.array
            Data to build the index on.

        Returns
        -------
        time : float
            Time it took to build the index.
        """
        s = time.time()
        # ---- cluster the data into categories ---- #
        _, labels = self.cluster(data, n_categories)

        # ---- train a neural network ---- #
        dataset = LIDataset(data, labels)
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=256,
            sampler=torch.utils.data.SubsetRandomSampler(
                data.index.values.tolist()
            )
        )
        nn = NeuralNetwork(
            input_dim=data.shape[1],
            output_dim=n_categories,
            lr=lr,
            model_type=model_type
        )
        nn.train_batch(train_loader, epochs=epochs, logger=self.logger)
        # ---- collect predictions ---- #
        self.model = nn
        return nn.predict(data_X_to_torch(data)), time.time() - s

    def cluster(
        self,
        data,
        n_clusters,
        n_redo=10,
        spherical=True,
        int_centroids=True,

    ):
        if data.shape[0] < 2:
            return None, np.zeros_like(data.shape[0])

        if data.shape[0] < n_clusters:
            n_clusters = data.shape[0] // 5
            if n_clusters < 2:
                n_clusters = 2

        kmeans = faiss.Kmeans(
            d=np.array(data).shape[1],
            k=n_clusters,
            verbose=True,
            #nredo=n_redo,
            #spherical=spherical,
            #int_centroids=int_centroids,
            #update_index=False,
            seed=2023
        )
        X = np.array(data).astype(np.float32)
        kmeans.train(X)

        return kmeans, kmeans.index.search(X, 1)[1].T[0]

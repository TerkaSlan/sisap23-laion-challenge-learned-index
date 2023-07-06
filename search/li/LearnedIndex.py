import numpy as np
from li.Logger import Logger
from li.utils import pairwise_cosine
import time
import torch
import torch.utils.data
from li.model import NeuralNetwork, data_X_to_torch, LIDataset
import faiss


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
        anns_final = None
        dists_final = None
        # sorts the predictions of a bucket for each query, ordered by lowest probability
        data_navigation['category'] = pred_categories

        # iterates over the predicted buckets starting from the most similar (index -1)
        for bucket in range(n_buckets):
            dists, anns = self.search_single(
                data_navigation,
                data_search,
                queries_search,
                pred_proba_categories[:, bucket]
            )
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

        return dists_final, anns_final, time.time() - s

    def search_single(
        self,
        data_navigation,
        data_search,
        queries_search,
        pred_categories,
        k=10
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
        nns = np.zeros((queries_search.shape[0], k), dtype=np.uint32)
        dists = np.zeros((queries_search.shape[0], k), dtype=np.float32)

        for cat in np.unique(pred_categories):
            cat_idxs = np.where(pred_categories == cat)[0]
            bucket_obj_indexes = data_navigation.query('category == @cat').index
            seq_search_dists = pairwise_cosine(
                queries_search[cat_idxs], data_search.loc[bucket_obj_indexes]
            )
            ann_relative = seq_search_dists.argsort()[:, :k]
            nns[cat_idxs] = np.array(bucket_obj_indexes)[ann_relative]
            dists[cat_idxs] = np.take_along_axis(seq_search_dists, ann_relative, axis=1)

        return dists, nns

    def build(self, data, n_categories=100, epochs=100, lr=0.1):
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
            model_type='MLP'
        )
        nn.train_batch(train_loader, epochs=epochs, logger=self.logger)
        # ---- collect predictions ---- #
        self.model = nn
        return nn.predict(data_X_to_torch(data)), time.time() - s

    def cluster(self, data, n_clusters):
        if data.shape[0] < 2:
            return None, np.zeros_like(data.shape[0])

        if data.shape[0] < n_clusters:
            n_clusters = data.shape[0] // 5
            if n_clusters < 2:
                n_clusters = 2

        kmeans = faiss.Kmeans(d=np.array(data).shape[1], k=n_clusters)
        X = np.array(data).astype(np.float32)
        kmeans.train(X)

        return kmeans, kmeans.index.search(X, 1)[1].T[0]

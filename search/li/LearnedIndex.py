import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from li.Logger import Logger
from li.utils import pairwise_cosine
import time


class LearnedIndex(Logger):

    def __init__(self):
        self.pq = []
        self.model = None

    def search(self, queries, data, n_buckets=2, k=10):
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
        data = pd.DataFrame(data)
        # assigns each object to its bucket (category)
        data_categories = pd.DataFrame(
            self.model.predict(data),
            index=data.index,
            columns=['category']
        )
        anns_final = None
        dists_final = None
        # sorts the predictions of a bucket for each query, ordered by lowest probability
        predicted_categories = np.argsort(self.model.predict_proba(queries), axis=1)

        # iterates over the predicted buckets starting from the most similar (index -1)
        for bucket in range(n_buckets):
            dists, anns = self.search_single(
                queries,
                data,
                data_categories,
                predicted_categories[:, -(bucket+1)]
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

                assert anns_final.shape == dists_final.shape == (queries.shape[0], k)

        return dists_final, anns_final, time.time() - s

    def search_single(self, queries, data, data_categories, predicted_categories, k=10):
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
        nns = np.zeros((queries.shape[0], k), dtype=np.uint32)
        dists = np.zeros((queries.shape[0], k), dtype=np.float32)
        for cat in np.unique(predicted_categories):
            cat_idxs = np.where(predicted_categories == cat)[0]
            bucket_obj_indexes = data_categories.query('category == @cat').index
            seq_search_dists = pairwise_cosine(queries[cat_idxs], data.loc[bucket_obj_indexes])
            ann_relative = seq_search_dists.argsort()[:, :k]
            nns[cat_idxs] = np.array(bucket_obj_indexes)[ann_relative] + 1
            dists[cat_idxs] = np.take_along_axis(seq_search_dists, ann_relative, axis=1)

        return dists, nns

    def build(self, data):
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
        data = pd.DataFrame(data)
        labels_df = self.get_train_labels(data)
        time_labels = time.time() - s
        s2 = time.time()
        model = self.train_index(data, labels_df)
        time_train = time.time() - s2
        self.model = model
        self.logger.info(
            f'Collecting labels took: {time_labels}, training took: {time_train}.'
        )
        return time.time() - s

    def create_category(self, data, random_state_offset):
        """ Create categories for multi-class classification.

        Parameters
        ----------
        data : np.array
            Data to create categories on.
        random_state_offset : int
            Random state offset to use.

        Returns
        -------
        obj_id : int
            Object id of the main object in the category.
        cat : np.array
            Array with object ids and distances to the main object.
        """
        main_datapoint = data.sample(1, random_state=2023+random_state_offset)
        distances = pairwise_cosine(main_datapoint.values, data)
        idxs = distances[0].argsort()[1:1001]
        return (
            main_datapoint.index[0],
            np.vstack([idxs, distances[0][idxs]])
        )

    def get_train_labels(self, data, n_categories=100):
        """ Get labels for training.

        Parameters
        ----------
        data : np.array
            Data to get labels for.
        n_categories : int
            Number of categories to create.

        Returns
        -------
        df_all : pd.DataFrame
            DataFrame with labels.
        """
        categories = []
        main_objs = []
        for i in range(n_categories):
            obj_id, cat = self.create_category(data, random_state_offset=i)
            main_objs.append(obj_id)
            categories.append(cat)

        df_all = pd.DataFrame(np.empty(0, dtype=np.uint32))
        for cat_id, c in enumerate(categories):
            df_ = pd.DataFrame(c.T)
            df_[2] = cat_id
            df_all = pd.concat([df_all, df_])

        df_all = df_all.rename(
            columns={
                0: 'object_id', 1: 'dist', 2: 'category_id'
            }
        ).sort_values(
            'dist', ascending=True
        ).drop_duplicates(
            'object_id', keep='first'
        )
        return df_all

    def train_index(self, data, labels_df):
        """ Train the index.

        Parameters
        ----------
        data : np.array
            Data to train the index on.
        labels_df : pd.DataFrame
            DataFrame with labels.

        Returns
        -------
        model : sklearn.linear_model.LogisticRegression
            Trained model.
        """
        X = data.loc[labels_df.object_id.astype(int)]
        y = labels_df.category_id.astype(int)
        model = LogisticRegression(random_state=2023, max_iter=500).fit(X, y)
        return model

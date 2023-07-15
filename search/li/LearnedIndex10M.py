import numpy as np
from li.Logger import Logger
from li.utils import pairwise_cosine
import time
from sklearn.metrics import accuracy_score
import gc
from sklearn.metrics.pairwise import cosine_similarity
from li.model import get_device, Model, data_to_torch, data_X_to_torch, predict_proba, ModelBigger
import torch


def get_random_indexes(size, dataset_size, seed):
    rng = np.random.default_rng(seed)
    random_idxs = rng.choice(range(dataset_size), size=(size), replace=False)
    random_idxs = np.sort(random_idxs)
    return random_idxs


class LearnedIndex(Logger):

    def __init__(self, dataset_shape, n_categories=1000):
        self.pq = []
        self.model = None
        self.dataset_size = dataset_shape[0]
        self.dataset_dim = dataset_shape[1]
        self.n_categories = n_categories
        self.dists_per_category = self.dataset_size // self.n_categories
        self.seed = 2023
        self.device = get_device()

    def search(self, queries, n_buckets=2, k=10):
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
        # assigns each object to its bucket (category)
        anns_final = None
        dists_final = None
        # sorts the predictions of a bucket for each query, ordered by lowest probability
        predicted_categories = np.argsort(
            predict_proba(
                self.model,
                self.device,
                data_X_to_torch(queries)
            ), axis=1
        )

        # iterates over the predicted buckets starting from the most similar (index -1)
        for bucket in range(n_buckets):
            dists, anns = self.search_single(
                queries,
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

    def search_single(self, queries, predicted_categories, k=10):
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
            bucket_obj_indexes = np.where(self.predictions == cat)[0]
            seq_search_dists = pairwise_cosine(queries[cat_idxs], self.all_data[bucket_obj_indexes])
            ann_relative = seq_search_dists.argsort()[:, :k]
            nns[cat_idxs] = np.array(bucket_obj_indexes)[ann_relative] + 1
            dists[cat_idxs] = np.take_along_axis(seq_search_dists, ann_relative, axis=1)

        return dists, nns

    def build(self, f, n_epochs=1000, lr=0.01, model_size=None):
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
        self.logger.info(
            'Getting train data.'
        )
        train_data, train_labels, training_indexes = self.get_train_data(f)
        time_labels = time.time() - s
        s2 = time.time()
        self.logger.info(
            'Training the index model.'
        )
        train_predictions, model = self.train_index(
            train_data, train_labels, epochs=n_epochs, model_size=model_size
        )
        self.logger.info(
            'Trained the index model, collected train preds.'
        )
        self.logger.info(
            f'Accuracy: {accuracy_score(train_labels, train_predictions)}.'
        )
        all_data = f[:, :]
        self.logger.info(
            'loaded all data for prediction'
        )
        pred_data_t = data_X_to_torch(all_data)
        self.logger.info(
            'predicting with all data'
        )
        test_predictions = predict(model, get_device(), pred_data_t)
        self.logger.info(
            f'collected test predictions: {test_predictions.shape}'
        )
        print(f'train_predictions.shape: {train_predictions.shape}')
        print(f'test_predictions.shape: {test_predictions.shape}')
        time_train = time.time() - s2
        self.model = model

        self.all_data = all_data
        self.predictions = test_predictions
        self.logger.info(
            f'Collecting labels took: {time_labels}, training took: {time_train}.'
        )
        return self.model, time.time() - s

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

    def get_train_data(self, f):
        """ Get data and labels for training.
        """
        final_dists_per_category = self.dists_per_category // 10

        training_indexes_all = np.empty((0, final_dists_per_category))
        training_data = np.empty((0, self.dataset_dim))
        prev = 0

        pivot_indexes = get_random_indexes(self.n_categories, self.dataset_size, self.seed)
        self.seed += 1
        assert pivot_indexes.shape == (self.n_categories, )
        self.logger.info(
            'Loading pivot data.'
        )
        pivot_data = f[pivot_indexes, :]
        self.logger.info(
            'Loaded pivot_data.'
        )
        assert pivot_data.shape == (self.n_categories, self.dataset_dim), \
            f'{(self.n_categories, self.dataset_dim)} != {pivot_data.shape}'

        self.logger.info(
            'Starting labels collecting loop.'
        )
        for batch, pivot in zip(
            range(self.dists_per_category, self.dataset_size+1, self.dists_per_category),
            pivot_data
        ):
            loaded_data = f[prev:batch, :]
            dists = np.argsort(cosine_similarity([pivot], loaded_data)[0])
            training_data_indexes = dists[-final_dists_per_category:]
            assert training_data_indexes.shape == (final_dists_per_category, )

            training_indexes_all = np.vstack((training_indexes_all, prev+training_data_indexes))
            training_data = np.vstack((training_data, loaded_data[training_data_indexes]))
            del loaded_data
            gc.collect()
            prev = batch
        self.logger.info(
            'Finished labels collecting loop.'
        )

        labels = np.array([
            np.array([
                i for _ in range(final_dists_per_category)
            ]) for i in range(self.n_categories)
        ])
        labels = labels.reshape(labels.shape[0]*labels.shape[1])
        self.logger.info(
            'Created labels.'
        )
        exp_out = self.n_categories*final_dists_per_category
        assert training_data.shape == (
            exp_out, self.dataset_dim
        ), f'{training_data.shape} != {(exp_out, self.dataset_dim)}'
        assert labels.shape == (training_indexes_all.shape[0]*training_indexes_all.shape[1], ),\
            f'{labels.shape} != {training_indexes_all.shape[0]*training_indexes_all.shape[1]}'

        return training_data, labels, training_indexes_all

    def train_index(self, data, labels, epochs=100, lr=0.01, model_size=None):
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
        if model_size is None:
            model = Model(input_dim=self.dataset_dim, output_dim=self.n_categories)
        else:
            model = ModelBigger(input_dim=self.dataset_dim, output_dim=self.n_categories)
        loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        device = get_device()
        data_X, data_y = data_to_torch(data, labels)
        self.logger.info(
            'Prepared everything, starting training.'
        )
        _, model = train(
            data_X,
            data_y,
            model,
            optimizer,
            device,
            loss,
            epochs=epochs,
            logger=self.logger
        )
        self.logger.info(
            'Finished training, collecting predictions.'
        )
        predictions = predict(model, device, data_X)
        self.logger.info(
            'Collected training predictions.'
        )
        return predictions, model

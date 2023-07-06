import numpy as np
import pandas as pd
import time
from enum import Enum
from typing import List, Union, Tuple

from li.model import NeuralNetwork, data_X_to_torch, data_to_torch, LIDataset
from li.Tree import InnerNode, LeafNode
from li.Logger import Logger
from li.utils import pairwise_cosine
import kmedoids
import gc
import faiss
from scipy import sparse
from sklearn.metrics import pairwise_distances
import torch
import torch.utils.data


def cluster_kmedoids(data: pd.DataFrame, n_clusters=10, max_iter=100, LOG=None) -> List[int]:
    """ Cluster the data using k-medoids and return the k-medoids object and the labels.
    Used for cosine distance using kmedoids:
        - Fast k-medoids Clustering in Rust and Python (https://doi.org/10.21105/joss.04183)

    Parameters
    ----------
    data : np.array
        The data to be clustered.
    n_clusters : int, optional
        The number of clusters, the default is 10.

    Returns
    -------
    Tuple
        The k-medoids object and the labels.
    """
    labels_all = []
    if data.shape[0] > n_clusters:
        if LOG is not None:
            LOG.info(f'Running pairwise_cosine on {data.shape}, this might take a while')
        data = sparse.csr_matrix(data)
        if LOG is not None:
            LOG.info(f'Created sparse matrix')
        #dists = pairwise_cosine(data, data, dense_output=False)
        dists = pairwise_distances(data, metric='cosine')
        if LOG is not None:
            LOG.info(f'Running kmedoids on {data.shape}, this might take a while')
        fp = kmedoids.fasterpam(dists, n_clusters, max_iter=max_iter)
        labels_all = np.array(fp.labels, dtype=np.int64)
        del dists
        gc.collect()
    else:
        labels_all = np.array([0] * data.shape[0])

    return labels_all


def cluster_kmeans_faiss(data: pd.DataFrame, n_clusters=10) -> Tuple[faiss.Kmeans, List[int]]:
    """ Cluster the data using k-means and return the k-means object and the labels.
    Used for euclidean distance using faiss (https://github.com/facebookresearch/faiss)
    Parameters
    ----------
    data : np.array
        The data to be clustered.
    n_clusters : int, optional
        The number of clusters, the default is 10.

    Returns
    -------
    Tuple[faiss.KMeans, np.array]
        The faiss k-means object and the labels.
    """
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


class NodeType(Enum):
    """ Enum class for node types."""
    INNER_NODE = InnerNode.__name__
    LEAF_NODE = LeafNode.__name__


class BaseLMI(Logger):
    """ The base LMI class """
    def __init__(
        self
    ):
        self.nodes = {}
        self.pq = []

    def search(
        self,
        query,
        stop_condition=None,
        stop_condition_leaf=None,
        stop_condition_time=None
    ) -> Tuple:
        """ Searches for `query` in the LMI. Recursively searches through the tree
        (starting from root), fills out priority queue (sorted by probability
        == similarity to the query) and checks for a given stop condition.
        One of the stop conditions needs to be provided.

        Parameters
        ----------
        query : np.array
            The query to search for.
        stop_condition : int, optional
            The number of objects to search for, by default None
        stop_condition_leaf : int, optional
            The number of leaf nodes to search for, by default None
        stop_condition_time : int, optional
            The time in seconds to search for, by default None

        Returns
        -------
        Tuple
            A Tuple of (candidate_answer, prob_distributions, n_objects, time)
        """
        assert not (
            stop_condition is None and stop_condition_leaf is None and stop_condition_time is None
        ), "At least one stop condition needs to be provided."

        self.pq = []
        candidate_answer = []
        prob_distrs = []
        start_time = time.time()
        self.n_objects = 0
        if isinstance(query, pd.core.series.Series):
            query = query.values
        query = data_X_to_torch(query)

        def predict_positions(parent_position: Tuple[int], query: np.array) -> Tuple[np.array]:
            """ Recursive function following the branch based on highest node prediction
            probability until a leaf node is met.
            BASE CASE: Stop condition is met or priority queue is empty

            Parameters
            ----------
            parent_position : Tuple[int]
                The node being the parent of the branch to inspect.
            query : np.array
                The query to search for.

            Returns
            -------
            Tuple
                A Tuple of (candidate_answer, prob_distributions, n_objects, time)
            """
            def predict_positions_of_child_nodes(
                parent_position: Tuple[int],
                query: np.array
            ) -> Tuple[np.array]:
                """ Predicts the position of the child nodes of `parent_position` based on `query`.
                Recursively calls predict_positions with child nodes of `parent_position` if
                priority queue is not empty.

                Parameters
                ----------
                parent_position : Tuple[int]
                    The node being the parent of the branch to inspect.
                query : np.array
                    The query to search for.

                Returns
                -------
                Tuple
                    A Tuple of (candidate_answer, prob_distributions, n_objects, time)
                """
                # (2.1) Collect predictions from the parent node's model
                probs, positions = self.nodes[parent_position].nn.predict_proba(query)
                positions = [parent_position + tuple((position, )) for position in positions]
                prob_distr = np.array([positions, probs], dtype=object).T

                # (2.2) Extend the priority queue with the predictions
                if type(self.pq) is list:
                    self.pq = prob_distr
                else:
                    self.pq = np.concatenate([self.pq, prob_distr], axis=0)
                    # kind='stable' for an almost sorted array
                    self.pq = self.pq[self.pq[:, 1].argsort(kind='stable')[::-1]]

                # (2.3) Recursively call predict_positions with the child nodes of `parent_position`
                top_entry = self.pq[0]
                self.pq = self.pq[1:] if len(self.pq) > 1 else []

                # BASE CASE
                if len(self.pq) == 0:
                    return candidate_answer, prob_distrs, self.n_objects, time.time() - start_time
                prob_distrs.append(top_entry[1])
                return predict_positions(top_entry[0], query)

            # (1) If node is a leaf node, return the candidate answer.
            current_node = self.nodes.get(parent_position)
            if not hasattr(current_node, 'nn'):
                candidate_answer.append(parent_position)
                if current_node is not None:
                    self.n_objects += len(current_node.objects)

                # (1.1) Check if any of the stop conditions is met.
                if stop_condition is not None and self.n_objects >= stop_condition:
                    return candidate_answer, prob_distrs, self.n_objects, time.time() - start_time
                if stop_condition_leaf is not None and (
                  len(candidate_answer) == stop_condition_leaf or
                  self.n_leaf_nodes < stop_condition_leaf
                ):
                    return candidate_answer, prob_distrs, self.n_objects, time.time() - start_time
                if stop_condition_time is not None and \
                   (time.time() - start_time) >= stop_condition_time:
                    return candidate_answer, prob_distrs, self.n_objects, time.time() - start_time
                else:
                    # (1.2) Check if priority queue is not empty
                    if len(self.pq) == 0:
                        return (
                            candidate_answer,
                            prob_distrs,
                            self.n_objects,
                            time.time() - start_time
                        )
                    # (1.3) Continue with the most probable node in the priority queue.
                    top_entry = self.pq[0]
                    self.pq = self.pq[1:] if len(self.pq) > 1 else []
                    prob_distrs.append(top_entry[1])
                    return predict_positions(top_entry[0], query)
            else:
                # (2) If node is not a leaf node, move with the search to its child nodes.
                return predict_positions_of_child_nodes(parent_position, query)

        self.pq = []
        if isinstance(query, pd.core.series.Series):
            query = query.values

        # Start the recursive search from the root node ((0,)).
        return predict_positions((0,), query)

    def get_n_of_objects(self) -> int:
        """ Returns the number of objects in the LMI."""
        str_df = self.dump_structure()
        return str_df[str_df['type'] == NodeType.LEAF_NODE.value]['children'].sum()

    def get_leaf_nodes_pos(self) -> List[Tuple[int]]:
        """ Returns the positions of all leaf nodes."""
        str_df = self.dump_structure()
        return [i for i in str_df[str_df['type'] == NodeType.LEAF_NODE.value].index]

    def get_n_leaf_nodes(self) -> int:
        """ Returns the number of leaf nodes in the LMI."""
        return len([
            len(node) for node in self.nodes.values()
            if node.__class__.__name__ == NodeType.LEAF_NODE.value
        ])

    def insert(self, objects: pd.DataFrame, callback=None):
        """ Inserts `objects` to the leaf node with the highest probabilistic response.

        Parameters
        ----------
        objects : pd.DataFrame
            The objects to insert.
        callback : Callable, optional
            A callback function used to check inconsistencies for LMI, by default None
        """
        def predict_positions(position: Tuple[int], data: pd.DataFrame, ids: List[int]):
            """ Follows the branch based on highest node prediction probability until
            a leaf node is met. Inserts data to the leaf node.

            Parameters
            ----------
            position : Tuple[int]
                The node being the parent of the branch to inspect.
            data : pd.DataFrame
                The data to insert.
            ids : List[int]
                The ids of the data to insert.
            """
            def predict_positions_of_child_nodes(position: Tuple[int], data: pd.DataFrame):
                """ Predicts the position of the child nodes of `position` based on `data`.
                Recursively calls predict_positions with child nodes of `position`

                Parameters
                ----------
                position : Tuple[int]
                    The node being the parent of the branch to inspect.
                data : pd.DataFrame
                    The data to insert.
                """
                pred_positions = self.nodes[position].nn.predict(data_X_to_torch(data))
                for pred_position_cat in np.unique(pred_positions):
                    predict_positions(
                        position + tuple((pred_position_cat,)),
                        data[np.where(pred_positions == pred_position_cat)],
                        ids[np.where(pred_positions == pred_position_cat)]
                    )

            # found the leaf node to insert data to
            if position not in self.nodes:
                self.create_child_nodes(
                    [position[-1]],
                    position[:-1],
                    data,
                    ids,
                    [position[-1]]
                )
                pass
            elif not hasattr(self.nodes[position], 'nn'):
                self.nodes[position].insert_objects(data, ids)
                if callback:
                    callback(self.nodes[position])
            else:
                predict_positions_of_child_nodes(position, data)

        if len(self.nodes) == 0:
            node = LeafNode((0,), self.ln_cap_min, self.ln_cap_max)
            node.insert_objects(objects.values, objects.index)
            self.insert_node(node)

        else:
            predict_positions((0,), objects.values, objects.index)

        self.n_leaf_nodes = self.get_n_leaf_nodes()

    def create_child_nodes(
        self,
        new_child_positions: np.array,
        parent_node_position: Tuple[int],
        objects: pd.DataFrame,
        object_ids: List[int],
        pred_positions: np.array,
        unique_pred_positions=None
    ):
        """ Creates new leaf nodes based on number of new positions, inserts their respective data
        objects.

        Parameters
        ----------
        new_child_positions : np.array
            The positions of the new leaf nodes.
        parent_node_position : Tuple[int]
            The position of the parent node.
        objects : pd.DataFrame
            The data objects to insert.
        object_ids : List[int]
            The ids of the data objects to insert.
        pred_positions : np.array
            The predicted positions of the data objects.
        unique_pred_positions : np.array, optional
            The unique predicted positions of the data objects, by default None
        """
        if unique_pred_positions is None:
            unique_pred_positions = new_child_positions

        for position_cat in new_child_positions:
            new_leaf_node = LeafNode(
                parent_node_position + tuple((position_cat, )),
                self.ln_cap_min,
                self.ln_cap_max
            )
            if position_cat in unique_pred_positions:
                current_cat_index = np.where(pred_positions == position_cat)
                accessed_mapping = map(objects.__getitem__, list(current_cat_index[0]))
                objs_to_insert = list(accessed_mapping)
                accessed_mapping = map(object_ids.__getitem__, list(current_cat_index[0]))
                obj_ids_to_insert = list(accessed_mapping)
                self.logger.debug(
                    f'Creating new leaf node: {parent_node_position + tuple((position_cat, ))}' +
                    f' for {parent_node_position}, inserting {len(objs_to_insert)} objects into' +
                    ' this new leaf node'
                )
                new_leaf_node.insert_objects(
                    objs_to_insert,
                    obj_ids_to_insert
                )
            self.insert_node(new_leaf_node)

    @staticmethod
    def train_model(
        objects: np.array,
        labels: List[int],
        n_clusters: int,
        epochs=100,
        model_type='MLP',
        lr=0.1,
        batch_size=256,
        logger=None
    ) -> Tuple:
        """ Trains a new NeuralNetwork, collects the predictions.

        Parameters
        ----------
        objects : np.array
            Data objects to train the model on.
        labels : List[int]
            Labels of the data objects.
        n_clusters : int
            Number of classes to train the model on.
        epochs : int, optional
            Number of epochs to train the model, by default 500


        Returns
        -------
        Tuple
            A Tuple of (Trained NeuralNetwork, predictions, losses)
        """
        #data_X, data_y = data_to_torch(objects, labels)
        """
        class_w = data_X_to_torch(
            (1 - pd.Series(labels).value_counts().sort_index() / len(labels)).values
        )
        class_w = None if class_w.shape[0] == 1 else class_w
        """
        class_w = None
        nn = NeuralNetwork(
            input_dim=len(objects[0]), output_dim=n_clusters, model_type=model_type, class_weight=class_w,
            lr=lr
        )
        dataset = LIDataset(objects, labels)
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(np.arange(len(labels)))
        )
        try:
            #losses = nn.train(data_X, data_y, epochs=epochs, logger=logger)
            losses = nn.train_batch(train_loader, epochs=epochs, logger=logger)
        except RuntimeError as e:
            print(f'RuntimeError: {e} -- caught, training without class weights')
            nn = NeuralNetwork(
                input_dim=len(objects[0]), output_dim=n_clusters, model_type=model_type
            )
            #losses = nn.train(data_X, data_y, epochs=epochs)
            losses = nn.train_batch(train_loader, epochs=epochs)

        return nn, nn.predict(data_X_to_torch(objects)), losses[-1]

    @staticmethod
    def get_objects(leaf_nodes: List[LeafNode]) -> Tuple[int, List[int]]:
        """ Retrieves all the objects from `leaf_nodes`.

        Parameters
        ----------
        leaf_nodes : List[LeafNode]
            Leaf nodes to retrieve the objects from.

        Returns
        -------
        Tuple[int, List[int]]
            A Tuple of (objects, object_ids)
        """
        objects = np.concatenate(
            [leaf_node.objects for leaf_node in leaf_nodes if len(leaf_node.objects) > 0],
            axis=0
        )
        object_ids = np.concatenate(
            [leaf_node.object_ids for leaf_node in leaf_nodes if len(leaf_node.object_ids) > 0],
            axis=0
        )
        return objects, object_ids

    def delete_nodes(
        self,
        leaf_nodes: List[Union[LeafNode, InnerNode]],
        remove_nn=True,
        callback=None
    ):
        """ Deletes `leaf_nodes` from the LMI.

        Parameters
        ----------
        leaf_nodes : List[Union[LeafNode, InnerNode]]
            The leaf nodes to delete.
        remove_nn : bool, optional
            If True, the node's NeuralNetwork is removed, by default True
        callback : Callable, optional
            A callback function used to check inconsistencies for LMI, by default None
        """
        for i in range(len(leaf_nodes)-1, -1, -1):
            leaf_node = leaf_nodes[i]
            if callback:
                callback(leaf_node)
            self.remove_node(leaf_node, remove_nn=remove_nn)

    def find_object_by_id(self, object_id: int) -> Tuple[int]:
        """ Returns the position of the leaf node containing `object_id` or None if not found.

        Parameters
        ----------
        object_id : int
            The id of the object to search for.

        Returns
        -------
        Tuple[int]
            The position of the leaf node or None if not found.
        """
        for _, node in self.nodes.items():
            if node.is_leaf_node() and object_id in node.object_ids:
                return node.position
        return None

    def dump_structure(self) -> pd.DataFrame:
        """ Create a dataframe of LMI's structure.

        Returns
        -------
        pd.DataFrame
            A dataframe of LMI's structure.
        """
        struct_df = pd.DataFrame(
            np.array([
                list(self.nodes.keys()),
                [node.__class__.__name__ for node in list(self.nodes.values())],
                [len(node) for node in list(self.nodes.values())]
            ], dtype=object),
            ['key', 'type', 'children']
        ).T
        struct_df = struct_df.set_index(['key'])
        return struct_df


    def get_parent_node(self, position: Tuple[int]) -> Union[Tuple[int], None]:
        """ Returns the parent node based on `position` or None if the node has
        no parent (is root).

        Parameters
        ----------
        position : Tuple[int]
            The position of the node to get the parent of.

        Returns
        -------
        Union[Tuple[int], None]
            The parent node or None if the node has no parent.
        """
        if len(position[:-1]) != 0:
            return self.nodes.get(position[:-1])
        else:
            return None

    def insert_node(self, node: InnerNode, should_check=False, callback=None):
        """ Puts `node` to LMI's `nodes` list and increases its parent's `children` count.

        Parameters
        ----------
        node : InnerNode
            The node to insert.
        should_check : bool, optional
            If True, inconsistencies are checked, by default True
        callback : Callable, optional
            A callback function used to check inconsistencies for LMI, by default None
        """
        self.nodes[node.position] = node
        if should_check:
            if callback:
                callback(node)

        self.logger.debug(f'Inserted node `{node.position}` into LMI')

        parent_node = self.get_parent_node(node.position)
        if parent_node is not None:
            parent_node.children.append(node)

    @staticmethod
    def prepare_data_cluster_kmedoids(
        data_partition: pd.DataFrame, n_children: int, object_ids: pd.Index
    ) -> Tuple[pd.DataFrame]:
        """ Prepares the data for clustering with K-Medoids.

        Parameters
        ----------
        data_partition : pd.DataFrame
            The data to prepare.
        n_children : int
            The number of children to split the data into.
        object_ids : pd.Index
            The ids of the data objects.

        Returns
        -------
        Tuple[pd.DataFrame]
            A Tuple of (data_partition, data_predict, data_partition_index)
        """
        limit = data_partition.shape[0]
        if data_partition.shape[0] >= 1000:
            # make it 1-10% of the data
            limit = int(data_partition.shape[0] * 0.01)
            increment = 0
            #  but always at least 8 data points per one category on average
            while limit <= n_children*8:
                limit = int(data_partition.shape[0] * (0.01 + increment))
                increment += 0.01
        if data_partition.shape[0] > limit:
            data_all = pd.DataFrame(data_partition, index=object_ids)
            data_train_partition = data_all.sample(limit, random_state=2022)
            data_predict = data_all.loc[data_all.index.difference(data_train_partition.index)]
            data_train_partition_index = data_train_partition.index
            data_train_partition = data_train_partition.values
            assert data_train_partition.shape[0] + data_predict.shape[0] == data_all.shape[0], \
                f'Inconsistent shapes: data_train_partition: {data_train_partition}, ' \
                f'data_predict: {data_predict}, data_all: {data_all}'
            data_partition = data_train_partition
        else:
            data_predict = None
            data_all = None
            data_train_partition_index = None
        return data_partition, data_predict, data_train_partition_index, data_all

    @staticmethod
    def collect_predictions_kmedoids(
        nn: NeuralNetwork,
        data_predict: pd.DataFrame,
        pred_positions: np.array,
        data_partition_index: pd.Index,
        data_all: pd.DataFrame,
        logger=None
    ):
        """ Collects predictions for the kmedoids training.

        Parameters
        ----------
        nn : NeuralNetwork
            The NeuralNetwork to collect the predictions from.
        data_predict : pd.DataFrame
            The data to collect the predictions from.
        pred_positions : np.array
            The predicted positions of the data objects.
        data_partition_index : pd.Index
            The ids of the data objects.

        Returns
        -------
        np.array
            The predictions.
        """
        if data_predict.shape[0] > 1_183_514:
            increment = data_predict.shape[0] // 8
            res_all = []
            for i in range(8):
                res = nn.predict(
                    data_X_to_torch(data_predict.iloc[i*increment:(i+1)*increment])
                )
                res_all.append(res)
            pred_positions_second = np.hstack(res_all)
        else:
            pred_positions_second = nn.predict(data_X_to_torch(data_predict.values))

        if logger is not None:
            logger.info(f'pred_positions_second: {pred_positions_second.shape}')
        data_all['labels'] = np.nan
        def make_consistent_shapes(pred_positions_second, data_predict):
            if pred_positions_second.shape[0] < data_predict.shape[0]:
                pred_positions_second = np.concatenate(
                    [pred_positions_second, np.array([pred_positions_second[-1]])]
                )
                return make_consistent_shapes(pred_positions_second, data_predict)
            else:
                return pred_positions_second
        data_all.loc[data_predict.index, 'labels'] = make_consistent_shapes(
            pred_positions_second,
            data_predict
        )
        data_all.loc[data_partition_index, 'labels'] = pred_positions
        return data_all['labels'].values

    def deepen(
        self,
        leaf_node: LeafNode,
        n_children: int,
        info_df: pd.DataFrame,
        epochs=100,
        lr=0.1,
        model_type='MLP',
        batch_size=256,
        callback=None
    ):
        """ Performs the DEEPEN (split downwards) operation on `leaf_node`.

        Parameters
        ----------
        leaf_node : LeafNode
            The leaf node to perform the DEEPEN operation on. Will be turned into an inner node.
        n_children : int
            The number of children to split the `leaf_node` into.
        info_df : pd.DataFrame
            A dataframe to append the information about the operation to.
        distance_function : str
            The distance function to use for clustering.
        callback : Callable, optional
            A callback function used to check inconsistencies for LMI, by default None

        Returns
        -------
        pd.DataFrame
            A dataframe with the appended information about the operation.
        """
        if len(leaf_node.objects) <= 1:
            return info_df
        data_predict = None
        data_all = None
        n_objects = self.get_n_of_objects()
        self.logger.info(f'==== Deepen with {leaf_node.position}')

        # (1) Clustering
        s_partition = time.time()
        data_part = np.array(leaf_node.objects)
        # Not used with Faiss' K-Medoids
        """
        (
            data_part,
            data_predict,
            data_part_index,
            data_all
        ) = BaseLMI.prepare_data_cluster_kmedoids(
            data_part, n_children, leaf_node.object_ids
        )
        """
        #labels = np.random.randint(n_children, size=data_part.shape[0])
        #labels = cluster_kmedoids(data_part, n_clusters=n_children)
        _, labels = cluster_kmeans_faiss(data_part, n_clusters=n_children)

        t_partition = time.time() - s_partition
        self.logger.debug(
            f'==== Partitioned: {t_partition} | n. unique labels={len(np.unique(labels))}'
        )
        if info_df is not None:
            info_df.loc[len(info_df.index)] = [
                f'DEEPEN-PART-{leaf_node.position}-{n_children}',
                t_partition,
                np.NaN,
                n_objects
            ]
        else:
            self.logger.info('info_df is none')
        # (2) Train model
        s_train = time.time()
        nn, pred_positions, loss = BaseLMI.train_model(
            data_part,
            labels,
            n_children,
            epochs=epochs,
            lr=lr,
            model_type=model_type,
            batch_size=batch_size,
            logger=self.logger
        )
        self.logger.debug(f'Training loss: {loss}')
        # after-predict (for k-medoids, if there are too many datapoints)
        if data_predict is not None:
            pred_positions = BaseLMI.collect_predictions_kmedoids(
                nn,
                data_predict,
                pred_positions,
                data_part_index,
                data_all
            )

        t_train = time.time() - s_train
        if info_df is not None:
            info_df.loc[len(info_df.index)] = [
                f'DEEPEN-TRAIN-{leaf_node.position}-{n_children}',
                t_train,
                np.NaN,
                n_objects
            ]
        s_nodes_cleanup = time.time()
        position = leaf_node.position
        # (3) Delete the obsolete leaf node
        self.delete_nodes([leaf_node], remove_nn=False)

        # (4) Create new inner node
        node = InnerNode(position, nn, self.child_n_min, self.child_n_max)
        # we'll not check the inconsistencies just yet
        self.insert_node(node, should_check=False)

        # (5) Create new child nodes
        self.create_child_nodes(
            np.arange(n_children),
            node.position,
            leaf_node.objects,
            leaf_node.object_ids,
            pred_positions
        )

        t_nodes_cleanup = time.time() - s_nodes_cleanup
        info_df.loc[len(info_df.index)] = [
            f'DEEPEN-REST-{leaf_node.position}-{n_children}',
            t_nodes_cleanup,
            np.NaN,
            n_objects
        ]
        if callback:
            callback(node)
        return info_df

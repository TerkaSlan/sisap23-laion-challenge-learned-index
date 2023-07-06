import numpy as np
from typing import Union
from enum import Enum
from li.Tree import InnerNode, LeafNode
from li.BaseLMI import BaseLMI


class NodeType(Enum):
    INNER_NODE = InnerNode.__name__
    LEAF_NODE = LeafNode.__name__


class BulkLMI(BaseLMI):
    """ The Learned Metric Index implementation."""
    def __init__(
        self,
    ):
        """ The Static LMI implementation."""
        super().__init__()
        self.ln_cap_min = np.nan
        self.ln_cap_max = np.nan
        self.child_n_min = np.nan
        self.child_n_max = np.nan

    def remove_node(self, node: Union[InnerNode, LeafNode], remove_nn=None):
        """ Deletes `node` from LMI's `nodes` and decreases its parent's `children` count.

        Parameters
        ----------
        node : Union[InnerNode, LeafNode]
            Node to be deleted.
        """
        self.logger.debug(f'Removing node at `{node.position}`')
        parent_node = self.get_parent_node(node.position)
        if parent_node is not None:
            parent_node.children.remove(node)
        del self.nodes[node.position]

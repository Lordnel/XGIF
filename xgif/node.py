"""
node.py
-------
IsoNode: internal node and leaf representation for isolation trees.
"""

import numpy as np


class IsoNode:
    """
    A node in an isolation tree.

    Attributes
    ----------
    depth : int
        Depth of this node in the tree (root = 0).
    q : int or None
        Index of the splitting feature (None for leaves).
    p : float or None
        Split threshold (None for leaves).
    size : int
        Number of training instances that reached this node.
    left : IsoNode or None
        Left child (instances with x[q] < p).
    right : IsoNode or None
        Right child (instances with x[q] >= p).
    n_l : int
        Number of instances routed to the left child.
    n_r : int
        Number of instances routed to the right child.
    """

    def __init__(
        self,
        depth: int,
        p: float | None,
        q: int | None,
        size: int,
        left: "IsoNode | None",
        right: "IsoNode | None",
        n_l: int = 0,
        n_r: int = 0,
    ) -> None:
        self.depth = depth
        self.q     = q
        self.p     = p
        self.size  = size
        self.left  = left
        self.right = right
        self.n_l   = n_l
        self.n_r   = n_r

    def is_leaf(self) -> bool:
        """Return True if this node is a leaf (no children)."""
        return self.left is None and self.right is None

    @staticmethod
    def compute_iic(n: int, n_l: int, n_r: int, adjust: bool = True) -> float:
        """
        Compute the Induced Imbalance Coefficient (IIC) for a split.

        The IIC measures how strongly a split isolates a small subset of
        instances. A perfectly isolating split (one child has a single
        instance) receives the maximum score of 1.0. A perfectly balanced
        split receives the minimum score of 0.5. A useless split (all
        instances go to one child) receives 0.0.

        Parameters
        ----------
        n : int
            Total number of instances at the node.
        n_l : int
            Number of instances routed to the left child.
        n_r : int
            Number of instances routed to the right child.
        adjust : bool, default True
            If True, scale the raw ratio into [0.5, 1.0] to mitigate
            node-size effects (as in the original DIFFI paper).

        Returns
        -------
        float
            IIC value in [0.0, 1.0], or -1.0 if n <= 1 (undefined).
        """
        if n <= 1:
            return -1.0                     # undefined for trivial nodes
        if n_l == 0 or n_r == 0:
            return 0.0                      # useless split

        ratio = max(n_l, n_r) / n           # in (0.5, 1.0]

        if not adjust:
            return float(ratio)

        # Scale into [0.5, 1.0] relative to the min/max achievable ratios
        # given this node size, so that node size does not bias the score.
        lambda_min = (n // 2) / n           # worst non-useless split
        lambda_max = (n - 1) / n            # best split (isolating)

        denom = lambda_max - lambda_min
        if denom < 1e-12:
            return 1.0

        iic = ((ratio - lambda_min) / denom) * 0.5 + 0.5
        return float(np.clip(iic, 0.5, 1.0))

    def __repr__(self) -> str:
        leaf_str = "leaf" if self.is_leaf() else f"q={self.q}, p={self.p:.4f}"
        return f"IsoNode(depth={self.depth}, size={self.size}, {leaf_str})"
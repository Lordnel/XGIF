"""
tree.py
-------
IsoTree: a single isolation tree used by IsoForest.
"""

from __future__ import annotations
import numpy as np
from node import IsoNode

def _c(n: int) -> float:
    """
    Expected path length of an unsuccessful search in a Binary Search Tree
    with n nodes. Used to normalise anomaly scores.

        c(n) = 2 * H(n-1) - 2*(n-1)/n

    where H(k) ≈ ln(k) + 0.5772156649 (Euler–Mascheroni constant).

    Parameters
    ----------
    n : int
        Number of instances in the leaf (or subsample size ψ).

    Returns
    -------
    float
        Correction value. Returns 0 for n ≤ 1, 1 for n = 2.
    """
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (np.log(n - 1) + 0.5772156649) - (2 * (n - 1) / n)

class IsoTree:
    """
    A single isolation tree.

    In standard mode (xg=False), the tree follows the original Isolation
    Forest construction: uniform random feature selection, uniform random
    split threshold, and a fixed depth limit d_max = ceil(log2(ψ)).

    In guided mode (xg=True, XGIF), the tree is augmented with two
    attribution-guided mechanisms in the extended growth zone
    (d_max ≤ depth < d_max_ext):
      - Mechanism 1: adaptive growth — the tree extends beyond d_max only
        when a node is *not* attribution-pure (max ξ(x_i) ≥ τ).
      - Mechanism 2: GFI-guided feature selection — features are drawn
        from a softmax distribution over GFI scores instead of uniformly.

    Parameters
    ----------
    max_depth : int
        Standard depth limit d_max = ceil(log2(ψ)).
    xg : bool, default False
        Enable XGIF guided mechanisms.
    max_depth_extended : int or None
        Hard ceiling depth d_max_ext = ceil(ext * d_max).
        Defaults to max_depth when None (no extension).
    lfi_ : np.ndarray of shape (N, Q) or None
        Local Feature Importance matrix from DIFFI warm-start.
        Required when xg=True.
    tau : float or None
        Attribution-purity threshold τ. Required when xg=True.
    feature_probs_ : np.ndarray of shape (Q,) or None
        GFI-derived feature selection distribution P.
        Required when xg=True.
    global_std_ref : float or None
        Global normalization constant σ̄ = mean_i std_q(LFI[i, :]).
        Required when xg=True.
    """

    def __init__(
        self,
        max_depth: int,
        xg: bool = False,
        max_depth_extended: int | None = None,
        lfi_: np.ndarray | None = None,
        tau: float | None = None,
        feature_probs_: np.ndarray | None = None,
        global_std_ref: float | None = None,
    ) -> None:
        self.max_depth          = max_depth
        self.max_depth_extended = max_depth_extended
        self.xg                 = xg
        self.lfi_               = lfi_
        self.tau                = tau
        self.feature_probs_     = feature_probs_
        self.global_std_ref     = global_std_ref

        self.root          : IsoNode | None = None
        self.idx           : np.ndarray | None = None

    def _select_feature(self, X: np.ndarray, idx: np.ndarray) -> int:
        """Uniform random feature selection (standard iForest)."""
        return int(np.random.randint(X.shape[1]))

    def _get_split_value(
        self, X: np.ndarray, idx: np.ndarray, q: int
    ) -> float:
        """Uniform random split threshold in [min_q, max_q]."""
        vals = X[idx, q]
        return float(np.random.uniform(vals.min(), vals.max()))

    def _should_stop(
        self, X: np.ndarray, idx: np.ndarray, depth: int
    ) -> bool:
        """
        Decide whether to stop growing at this node.

        Standard mode (xg=False)
        -------------------------
        Stop when: size ≤ 1, depth ≥ d_max, or depth ≥ d_max_ext.

        XGIF mode (xg=True)
        --------------------
        - depth < d_max      : never stop early (preserve iForest behavior).
        - depth ≥ d_max_ext  : always stop (hard ceiling).
        - d_max ≤ depth < d_max_ext : stop only if the node is
          attribution-pure (max_i ξ(x_i) < τ).

        The attribution stability signal ξ(x_i) is defined as:
            ξ(x_i) = std_q(LFI[i, :]) / (σ̄ + ε)
        A node is attribution-pure when no instance has ξ ≥ τ.
        """
        # Hard stop: trivial node or hard ceiling
        if len(idx) <= 1:
            return True
        if depth >= self.max_depth_extended:
            return True

        # Standard mode: stop at d_max
        if not self.xg:
            return depth >= self.max_depth

        # XGIF mode
        # Shallow zone: always continue (preserve iForest randomness)
        if depth < self.max_depth:
            return False

        # Extended zone: stop if attribution-pure
        lfi_s  = self.lfi_[idx]
        signal = np.where(
            self.global_std_ref > 1e-12,
            lfi_s.std(axis=1) / self.global_std_ref,
            0.0,
        )
        return bool(signal.max() < self.tau)

    def _build(
        self, X: np.ndarray, idx: np.ndarray, depth: int
    ) -> IsoNode:
        """
        Recursively build the isolation tree.

        Parameters
        ----------
        X : np.ndarray of shape (N, Q)
            Full training matrix.
        idx : np.ndarray of shape (k,)
            Indices (into X) of instances in the current node.
        depth : int
            Current depth.

        Returns
        -------
        IsoNode
            Root of the subtree built from idx.
        """
        node = IsoNode(depth, None, None, len(idx), None, None)

        if self._should_stop(X, idx, depth):
            return node

        # Feature selection
        # Extended zone + XGIF: GFI-guided distribution P
        # Otherwise: uniform random
        if depth >= self.max_depth and self.xg:
            q = int(np.random.choice(X.shape[1], p=self.feature_probs_))
        else:
            q = self._select_feature(X, idx)

        node.q = q

        min_val = X[idx, q].min()
        max_val = X[idx, q].max()

        # All values identical along this feature: cannot split -> leaf
        if max_val <= min_val:
            return node

        p = self._get_split_value(X, idx, q)
        node.p = p

        mask_l = X[idx, q] < p
        idx_l  = idx[mask_l]
        idx_r  = idx[~mask_l]

        node.n_l = len(idx_l)
        node.n_r = len(idx_r)

        node.left  = self._build(X, idx_l, depth + 1)
        node.right = self._build(X, idx_r, depth + 1)

        return node

    def fit(self, X: np.ndarray, idx: np.ndarray) -> IsoNode:
        """
        Build the tree on subsample idx.

        Parameters
        ----------
        X : np.ndarray of shape (N, Q)
        idx : np.ndarray of shape (ψ,)
            Subsample indices drawn from {0, …, N-1}.

        Returns
        -------
        IsoNode
            Root of the built tree.
        """
        self.idx = idx.copy()
        self.root = self._build(X, idx, 0)
        return self.root

    def _path_length(self, x: np.ndarray, node: IsoNode) -> float:
        """
        Compute the path length of instance x in this tree.

        For a leaf of size s at depth d: h(x) = d + c(s).
        """
        if node.is_leaf():
            return node.depth + _c(node.size)
        if x[node.q] < node.p:
            return self._path_length(x, node.left)
        return self._path_length(x, node.right)

    def _decision_path_rec(
        self, x: np.ndarray, node: IsoNode, path: list
    ) -> None:
        """Recursively collect internal nodes along the path of x."""
        if node.is_leaf():
            return
        path.append(node)
        child = node.left if x[node.q] < node.p else node.right
        self._decision_path_rec(x, child, path)

    def get_decision_path(self, x: np.ndarray) -> list[IsoNode]:
        """Return the list of internal nodes visited by x."""
        path: list[IsoNode] = []
        self._decision_path_rec(x, self.root, path)
        return path

    def get_leaf_depth(self, x: np.ndarray) -> int:
        """Return the depth of the leaf node reached by x."""
        node = self.root
        while not node.is_leaf():
            node = node.left if x[node.q] < node.p else node.right
        return node.depth

    def route_rec(
        self,
        X: np.ndarray,
        idx: np.ndarray,
        node: IsoNode,
        result: dict,
    ) -> None:
        """
        Recursively record (n, n_l, n_r) for each node reached by idx.
        Used by DIFFI to compute IIC values for a specific subset.
        """
        if node.is_leaf() or len(idx) == 0:
            return
        mask_l = X[idx, node.q] < node.p
        idx_l  = idx[mask_l]
        idx_r  = idx[~mask_l]
        result[id(node)] = (len(idx), len(idx_l), len(idx_r))
        self.route_rec(X, idx_l, node.left,  result)
        self.route_rec(X, idx_r, node.right, result)

    def path_for_subset(
        self, X: np.ndarray, idx: np.ndarray, node: IsoNode
    ) -> dict[int, tuple[int, int, int]]:
        """
        Return a dict mapping node id -> (n, n_l, n_r) for all nodes
        reached by the instances in idx. Used by global_diffi.
        """
        result: dict[int, tuple[int, int, int]] = {}
        self.route_rec(X, idx, node, result)
        return result
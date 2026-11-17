"""
diffi.py
--------
DIFFI: Depth-based Isolation Forest Feature Importance.

Implements:
  - local_diffi       : LFI for a single instance
  - local_diffi_batch : LFI for all instances in X
  - global_diffi      : GFI for the whole dataset

Reference:
  Carletti, M., Terzi, M., & Susto, G.A. (2023).
  Interpretable Anomaly Detection with DIFFI: Depth-based feature
  importance of Isolation Forest.
  Engineering Applications of Artificial Intelligence, 119, 105730.
"""

from __future__ import annotations
import numpy as np
from xgif.node import IsoNode
from xgif.tree import IsoTree

def _collect_internal_nodes(root: IsoNode) -> list[IsoNode]:
    """
    Return all internal (non-leaf) nodes in the tree via DFS.

    Parameters
    ----------
    root : IsoNode
        Root of the isolation tree.

    Returns
    -------
    list[IsoNode]
    """
    nodes: list[IsoNode] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if not node.is_leaf():
            nodes.append(node)
            stack.append(node.left)
            stack.append(node.right)
    return nodes


def _compute_iic_for_subset(
    tree: IsoTree,
    X: np.ndarray,
    idx: np.ndarray,
    adjust: bool = True,
) -> dict[int, float]:
    """
    Compute the Induced Imbalance Coefficient (IIC) for each internal
    node of *tree*, with counts (n, n_l, n_r) derived from the subset
    of instances specified by *idx*.

    Nodes not visited by any instance in *idx* receive IIC = -1.0
    (sentinel value: excluded from CFI updates).

    Parameters
    ----------
    tree : IsoTree
    X : np.ndarray of shape (N, Q)
    idx : np.ndarray of shape (k,)
        Global indices of the instances to route through the tree.
    adjust : bool, default True
        Pass to IsoNode.compute_iic for node-size normalization.

    Returns
    -------
    dict mapping node id (int) → IIC (float)
    """
    if len(idx) == 0:
        return {}

    route   = tree.path_for_subset(X, idx, tree.root)
    iic_map: dict[int, float] = {}

    for node in _collect_internal_nodes(tree.root):
        nid = id(node)
        if nid in route:
            n, n_l, n_r = route[nid]
            iic_map[nid] = IsoNode.compute_iic(n, n_l, n_r, adjust)
        else:
            iic_map[nid] = -1.0     # node not visited by this subset

    return iic_map

def local_diffi(forest, x: np.ndarray) -> np.ndarray:
    """
    Compute the Local Feature Importance (LFI) for a single instance x.

    For each tree t, LFI is accumulated along the isolation path of x:
    each internal node v splitting on feature f_j contributes

        Δ_j = 1/h_t(x) - 1/h_max

    where h_t(x) is the leaf depth and h_max = d_max. The correction
    -1/h_max zeroes out contributions when x is not isolated (leaf at
    the maximum depth).

    Parameters
    ----------
    forest : IsoForest
        A trained IsoForest instance.
    x : np.ndarray of shape (Q,)
        A single instance.

    Returns
    -------
    np.ndarray of shape (Q,)
        LFI scores, one per feature.
    """
    Q     = x.shape[0]
    I_loc = np.zeros(Q)
    C_loc = np.zeros(Q)

    for tree in forest.trees:
        path  = tree.get_decision_path(x)
        depth = tree.get_leaf_depth(x)

        if depth == 0:
            continue

        delta = (1.0 / depth) - (1.0 / forest.max_depth)

        for node in path:
            I_loc[node.q] += delta
            C_loc[node.q] += 1

    lfi = np.divide(I_loc, C_loc, out=np.zeros_like(I_loc), where=C_loc > 0)
    return lfi


def local_diffi_batch(
    forest, X: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute LFI for all instances in X.

    Parameters
    ----------
    forest : IsoForest
    X : np.ndarray of shape (N, Q)

    Returns
    -------
    lfi : np.ndarray of shape (N, Q)
        LFI matrix; row i corresponds to instance X[i].
    ord_idx : np.ndarray of shape (N, Q)
        Column indices that sort each row of lfi in descending order
        (most important feature first).
    """
    lfi = np.vstack([local_diffi(forest, X[i]) for i in range(X.shape[0])])
    ord_idx = np.argsort(-lfi, axis=1)
    return lfi, ord_idx

def global_diffi(
    forest,
    X: np.ndarray,
    contamination: float = 0.1,
    adjust_iic: bool = True,
) -> np.ndarray:
    """
    Compute the Global Feature Importance (GFI) for a trained forest.

    GFI is defined as:

        GFI_j = (I_O,j / C_O,j) / (I_I,j / C_I,j)

    where I_O, I_I are Cumulative Feature Importances (CFI) accumulated
    over predicted outliers and inliers respectively, and C_O, C_I are
    feature usage counters that normalise for random feature selection
    frequency.

    For each tree:
      1. Instances in the bootstrap sample are classified as outliers /
         inliers using the global anomaly scores and a threshold set at
         the (1 - contamination) quantile.
      2. IIC values are computed for each internal node using only the
         outlier or inlier subset (separate IIC maps for O and I).
      3. CFIs are updated along the isolation path of each instance:

             ΔI_j = (1/h_t(x)) x IIC(v)   for node v splitting on f_j

    Parameters
    ----------
    forest : IsoForest
        A trained IsoForest instance.
    X : np.ndarray of shape (N, Q)
        The same data used to train forest (in-bag computation).
    contamination : float, default 0.1
        Assumed anomaly rate. Used to threshold anomaly scores.
    adjust_iic : bool, default True
        Apply node-size normalization to IIC values.

    Returns
    -------
    np.ndarray of shape (Q,)
        GFI scores. Higher values indicate features that strongly
        discriminate outliers from inliers.
    """
    Q = X.shape[1]

    I_O = np.zeros(Q)
    I_I = np.zeros(Q)
    C_O = np.zeros(Q)
    C_I = np.zeros(Q)

    # Global anomaly scores for thresholding
    scores_global = forest.anomaly_score(X)
    threshold     = np.percentile(scores_global, 100 * (1 - contamination))

    for tree in forest.trees:
        in_bag   = tree.idx
        scores_ib = scores_global[in_bag]

        is_outlier = scores_ib >= threshold
        idx_O_local = np.where(is_outlier)[0]
        idx_I_local = np.where(~is_outlier)[0]

        if len(idx_O_local) == 0 or len(idx_I_local) == 0:
            continue     # skip degenerate trees

        # Global indices (into X) for outliers and inliers in this tree
        idx_O = in_bag[idx_O_local]
        idx_I = in_bag[idx_I_local]

        # IIC maps computed separately for outlier and inlier subsets
        iic_O = _compute_iic_for_subset(tree, X, idx_O, adjust_iic)
        iic_I = _compute_iic_for_subset(tree, X, idx_I, adjust_iic)

        # Accumulate CFI for outliers
        for xi in X[idx_O]:
            path  = tree.get_decision_path(xi)
            depth = tree.get_leaf_depth(xi)
            if depth == 0:
                continue
            for node in path:
                iic_val = iic_O.get(id(node), -1.0)
                if iic_val < 0:
                    continue
                I_O[node.q] += (1.0 / depth) * iic_val
                C_O[node.q] += 1

        # Accumulate CFI for inliers
        for xi in X[idx_I]:
            path  = tree.get_decision_path(xi)
            depth = tree.get_leaf_depth(xi)
            if depth == 0:
                continue
            for node in path:
                iic_val = iic_I.get(id(node), -1.0)
                if iic_val < 0:
                    continue
                I_I[node.q] += (1.0 / depth) * iic_val
                C_I[node.q] += 1

    # Normalise CFIs by feature usage frequency, then compute GFI ratio
    fi_O = np.divide(I_O, C_O, out=np.zeros_like(I_O), where=C_O > 0)
    fi_I = np.divide(I_I, C_I, out=np.zeros_like(I_I), where=C_I > 0)
    gfi  = np.divide(fi_O, fi_I, out=np.zeros_like(fi_O), where=fi_I > 0)

    return gfi
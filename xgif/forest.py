"""
forest.py
---------
IsoForest: Isolation Forest with optional XGIF guided extension.
"""

from __future__ import annotations
import numpy as np
from xgif.tree import IsoTree, _c

class IsoForest:
    """
    Isolation Forest with optional explanation-guided extension (XGIF).

    When ext > 1.0, XGIF is activated: a warm-start phase trains a
    preliminary forest IF₀ and computes DIFFI attribution scores (GFI,
    LFI). These signals guide IF₁ tree construction through two mechanisms:
      - Mechanism 1: adaptive growth beyond d_max using attribution stability.
      - Mechanism 2: GFI-guided feature selection in the extended zone.

    When ext = 1.0 (default), the model reduces to standard Isolation Forest.

    Parameters
    ----------
    t : int, default 100
        Number of isolation trees.
    psi : int, default 256
        Subsample size ψ used for each tree.
    random_state : int, default 42
        Random seed for reproducibility.
    tau : float, default 0.6
        Attribution-purity threshold τ (XGIF only).
        A node is attribution-pure when max_i ξ(x_i) < τ.
    ext : float, default 1.0
        Extension factor ext. Sets d_max_ext = ceil(ext * d_max).
        Values > 1.0 activate XGIF; 1.0 = standard iForest.
    temp : float, default 0.6
        Softmax temperature T for the GFI feature selection distribution.
        Lower values concentrate probability on top-GFI features.
    """

    def __init__(
        self,
        t: int   = 100,
        psi: int = 256,
        random_state: int   = 42,
        tau:          float = 0.6,
        ext:          float = 1.0,
        temp:         float = 0.6,
    ) -> None:
        self.t            = t
        self.psi          = psi
        self.random_state = random_state
        self.tau          = tau
        self.ext          = max(ext, 1.0)   # ext < 1.0 makes no sense
        self.temp         = temp

        # XGIF is active when the extension factor strictly exceeds 1.0
        self.xg: bool = self.ext > 1.0

        # Populated by _run_warmstart when xg=True
        self.lfi_           : np.ndarray | None = None
        self.gfi_           : np.ndarray | None = None
        self.feature_probs_ : np.ndarray | None = None
        self.global_std_ref : float | None      = None

        # Populated by fit()
        self.trees            : list[IsoTree] = []
        self.estimators_depth : list[int]     = []
        self.max_depth        : int | None    = None

    def _run_warmstart(self, X: np.ndarray) -> None:
        """
        Train a preliminary forest IF₀ and compute DIFFI attribution scores.

        IF₀ uses the same (t, ψ) as IF₁ but a different random seed
        (random_state + 7) to ensure independence between the two forests.

        Populates:
            self.lfi_           : Local Feature Importance (N, Q)
            self.gfi_           : Global Feature Importance (Q,)
            self.global_std_ref : σ̄ = mean_i std_q(LFI[i, :])
            self.feature_probs_ : softmax(clip(GFI, 0) / temp)  shape (Q,)
        """
        from xgif.diffi import global_diffi, local_diffi_batch

        seed_if0 = self.random_state + 7
        if0 = IsoForest(t=self.t, psi=self.psi, random_state=seed_if0)
        if0.fit(X)

        self.lfi_, _   = local_diffi_batch(if0, X)
        self.gfi_      = global_diffi(if0, X, contamination=0.1, adjust_iic=True)
        self.global_std_ref = float(self.lfi_.std(axis=1).mean())
        self.feature_probs_ = _gfi_to_probs(self.gfi_, self.temp)

    def fit(self, X: np.ndarray) -> "IsoForest":
        """
        Train the isolation forest on X.

        Parameters
        ----------
        X : np.ndarray of shape (N, Q)
            Training data.

        Returns
        -------
        self
        """
        N = X.shape[0]

        self.psi       = min(self.psi, N)
        self.max_depth = int(np.ceil(np.log2(self.psi)))

        # XGIF: compute attribution signals before building IF₁
        if self.xg:
            self._run_warmstart(X)

        np.random.seed(self.random_state)
        rng = np.random.default_rng(self.random_state)

        max_depth_ext = int(self.max_depth * self.ext)

        for _ in range(self.t):
            idx = rng.choice(N, self.psi, replace=False)

            tree = IsoTree(
                max_depth          = self.max_depth,
                xg                 = self.xg,
                max_depth_extended = max_depth_ext,
                lfi_               = self.lfi_,
                tau                = self.tau,
                feature_probs_     = self.feature_probs_,
                global_std_ref     = self.global_std_ref,
            )
            tree.fit(X, idx)
            self.estimators_depth.append(tree.current_depth)
            self.trees.append(tree)

        return self
    
    def _average_path_length(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the mean path length E[h(x)] over all trees.

        Returns
        -------
        np.ndarray of shape (N,)
        """
        depths = np.array(
            [[tree._path_length(x, tree.root) for x in X]
             for tree in self.trees]
        )                                   # (t, N)
        return depths.mean(axis=0)          # (N,)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the anomaly score s(x) = 2^{-E[h(x)] / c(ψ)} for each x.

        Higher scores indicate stronger anomalies.

        Parameters
        ----------
        X : np.ndarray of shape (N, Q)

        Returns
        -------
        np.ndarray of shape (N,), values in (0, 1].
        """
        E_h = self._average_path_length(X)
        return np.power(2.0, -E_h / _c(self.psi))

    def predict(self, X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        """
        Predict binary labels (1 = anomaly, 0 = normal).

        The threshold is set at the (1 - contamination) quantile of
        anomaly scores on X.

        Parameters
        ----------
        X : np.ndarray of shape (N, Q)
        contamination : float, default 0.1
            Assumed proportion of anomalies in X.

        Returns
        -------
        np.ndarray of shape (N,), dtype int, values in {0, 1}.
        """
        scores    = self.anomaly_score(X)
        threshold = np.percentile(scores, 100 * (1 - contamination))
        return (scores >= threshold).astype(int)

def _gfi_to_probs(gfi: np.ndarray, temp: float) -> np.ndarray:
    """
    Convert Global Feature Importance scores to a probability distribution
    using a clipped softmax with temperature temp.

        P(q) = softmax(clip(GFI, 0) / temp)

    Negative GFI values (features more important for inliers) are clipped
    to zero before the softmax so they do not contribute to the distribution.
    If all scores are zero or negative, returns a uniform distribution.

    Parameters
    ----------
    gfi : np.ndarray of shape (Q,)
    temp : float
        Softmax temperature. Lower -> more concentrated on top features.

    Returns
    -------
    np.ndarray of shape (Q,), sums to 1.
    """
    scores = np.clip(gfi, 0, None)

    if scores.sum() < 1e-12:
        return np.ones(len(gfi)) / len(gfi)    # fallback: uniform

    scaled  = scores / temp
    scaled -= scaled.max()                      # numerical stability
    exp     = np.exp(scaled)
    return exp / exp.sum()
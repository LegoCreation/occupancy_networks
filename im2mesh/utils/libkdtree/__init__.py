"""KDTree backend selection with safe fallbacks.

The original Occupancy Networks code expects a compiled pykdtree extension.
On environments where it is not built, importing im2mesh would fail early.
This module keeps the same ``KDTree`` symbol while falling back to SciPy,
scikit-learn, or a minimal NumPy implementation.
"""

import numpy as np


try:
    # Fast path used by the original repository when the extension is built.
    from .pykdtree.kdtree import KDTree  # type: ignore
except Exception:  # pragma: no cover
    try:
        from scipy.spatial import cKDTree as _CKDTree

        class KDTree:  # noqa: D401
            """SciPy-backed KDTree with pykdtree-like query signature."""

            def __init__(self, data, leafsize=10):
                self._tree = _CKDTree(np.asarray(data), leafsize=leafsize)

            def query(
                self,
                x,
                k=1,
                eps=0,
                distance_upper_bound=np.inf,
                sqr_dists=False,
                mask=None,
                **kwargs,
            ):
                # ``mask`` is not supported by cKDTree; ignore for compatibility.
                dist, idx = self._tree.query(
                    np.asarray(x),
                    k=k,
                    eps=eps,
                    distance_upper_bound=distance_upper_bound,
                    **kwargs,
                )
                if sqr_dists:
                    dist = np.square(dist)
                return dist, idx

    except Exception:  # pragma: no cover
        try:
            from sklearn.neighbors import KDTree as _SKKDTree

            class KDTree:  # noqa: D401
                """scikit-learn-backed KDTree fallback."""

                def __init__(self, data, leafsize=40):
                    self._tree = _SKKDTree(np.asarray(data), leaf_size=leafsize)

                def query(
                    self,
                    x,
                    k=1,
                    eps=0,
                    distance_upper_bound=np.inf,
                    sqr_dists=False,
                    mask=None,
                    **kwargs,
                ):
                    # ``eps``, ``distance_upper_bound`` and ``mask`` are ignored
                    # as sklearn's API differs; this is a best-effort fallback.
                    dist, idx = self._tree.query(np.asarray(x), k=k)
                    if k == 1:
                        dist = dist[:, 0]
                        idx = idx[:, 0]
                    if sqr_dists:
                        dist = np.square(dist)
                    return dist, idx

        except Exception:  # pragma: no cover
            class KDTree:  # noqa: D401
                """Minimal NumPy fallback for environments without KDTree libs."""

                def __init__(self, data, leafsize=10):
                    self.data = np.asarray(data)

                def query(
                    self,
                    x,
                    k=1,
                    eps=0,
                    distance_upper_bound=np.inf,
                    sqr_dists=False,
                    mask=None,
                    **kwargs,
                ):
                    x = np.asarray(x)
                    diff = x[:, None, :] - self.data[None, :, :]
                    dist2 = np.sum(diff * diff, axis=-1)

                    if k == 1:
                        idx = np.argmin(dist2, axis=1)
                        dist2_min = dist2[np.arange(x.shape[0]), idx]
                        dist = dist2_min if sqr_dists else np.sqrt(dist2_min)
                        if np.isfinite(distance_upper_bound):
                            too_far = dist > distance_upper_bound
                            dist = dist.astype(np.float64, copy=True)
                            idx = idx.astype(np.int64, copy=True)
                            dist[too_far] = np.inf
                            idx[too_far] = self.data.shape[0]
                        return dist, idx

                    idx = np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]
                    row = np.arange(x.shape[0])[:, None]
                    chosen = dist2[row, idx]
                    order = np.argsort(chosen, axis=1)
                    idx = idx[row, order]
                    chosen = chosen[row, order]
                    dist = chosen if sqr_dists else np.sqrt(chosen)

                    if np.isfinite(distance_upper_bound):
                        too_far = dist > distance_upper_bound
                        dist = dist.astype(np.float64, copy=True)
                        idx = idx.astype(np.int64, copy=True)
                        dist[too_far] = np.inf
                        idx[too_far] = self.data.shape[0]

                    return dist, idx


__all__ = ["KDTree"]

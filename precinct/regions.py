"""Region formation (size-balanced k-means) and territory queries."""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree


def balanced_kmeans(P, K, seed, iters=15):
    """Deterministic size-balanced k-means: every camera that runs this with
    the same (positions, K, seed) obtains the same partition."""
    rng = np.random.default_rng(seed)
    M = len(P)
    if K <= 1:
        return np.zeros(M, int), P.mean(0, keepdims=True)
    cap = math.ceil(M / K)
    C = P[rng.choice(M, K, replace=False)].copy()
    kk = min(K, 12)
    for _ in range(iters):
        # capacity-constrained greedy assignment over each point's kk nearest centers
        d, j = cKDTree(C).query(P, k=kk)
        d, j = d.reshape(M, kk), j.reshape(M, kk)
        order = np.argsort(d, axis=None, kind="stable")
        lab = np.full(M, -1)
        cnt = np.zeros(K, int)
        for flat in order:
            i, t = divmod(int(flat), kk)
            k = j[i, t]
            if lab[i] >= 0 or cnt[k] >= cap:
                continue
            lab[i] = k
            cnt[k] += 1
        for i in np.flatnonzero(lab < 0):     # rare: all nearby centers full
            free = np.flatnonzero(cnt < cap)
            k = free[np.argmin(np.linalg.norm(C[free] - P[i], axis=1))]
            lab[i] = k
            cnt[k] += 1
        newC = np.array([P[lab == k].mean(0) if (lab == k).any() else C[k] for k in range(K)])
        if np.allclose(newC, C):
            break
        C = newC
    return lab, C


def choose_voters(P, lab, C, R):
    voters = {}
    for k in range(len(C)):
        idx = np.flatnonzero(lab == k)
        d = np.linalg.norm(P[idx] - C[k], axis=1)
        voters[k] = [int(i) for i in idx[np.argsort(d)][:R]]
    return voters


class DirSnapshot:
    """Immutable versioned view of the camera directory. Any node holding a
    snapshot can compute the territory (nearest live camera, labelled by
    region) and the border zones."""

    def __init__(self, version, cam_pos, cam_region, live, centers, voters, centroids):
        self.version = version
        self.cam_pos = cam_pos              # array [Mtot, 2]
        self.cam_region = cam_region        # array [Mtot]
        self.live = live                    # bool array [Mtot]
        self.centers = dict(centers)        # region -> node id
        self.voters = {k: list(v) for k, v in voters.items()}
        self.centroids = centroids
        self.live_ids = np.flatnonzero(live)
        self.tree = cKDTree(cam_pos[self.live_ids]) if len(self.live_ids) else None
        self.live_region = cam_region[self.live_ids]
        self._members = None

    def members(self, region):
        if self._members is None:
            mm = {}
            for i in self.live_ids:
                mm.setdefault(int(self.cam_region[i]), []).append(int(i))
            self._members = mm
        return self._members.get(region, [])

    def region_at(self, p):
        if self.tree is None:
            return -1
        _, j = self.tree.query(p)
        return int(self.live_region[j])

    def regions_at(self, P):
        if self.tree is None or len(P) == 0:
            return np.full(len(P), -1)
        _, j = self.tree.query(P)
        return self.live_region[j]

    def margin(self, p, region, k=8):
        """Signed depth of p inside `region`'s territory:
        dist(nearest live camera of another region) - dist(nearest of region)."""
        if self.tree is None:
            return -math.inf
        k = min(k, len(self.live_ids))
        d, j = self.tree.query(p, k=k)
        d, j = np.atleast_1d(d), np.atleast_1d(j)
        regs = self.live_region[j]
        mine = np.flatnonzero(regs == region)
        other = np.flatnonzero(regs != region)
        if len(mine) == 0:
            return -math.inf
        if len(other) == 0:
            return math.inf
        return float(d[other[0]] - d[mine[0]])

    def border_regions(self, p, region, radius):
        """Regions other than `region` with a live camera within `radius` of p."""
        if self.tree is None:
            return set()
        js = self.tree.query_ball_point(p, radius)
        return {int(self.live_region[j]) for j in js} - {region}

    def covered(self, P, r):
        """Mask of points within r of some live camera."""
        if self.tree is None or len(P) == 0:
            return np.zeros(len(P), bool)
        d, _ = self.tree.query(P, distance_upper_bound=r)
        return np.isfinite(d)

    def nearest_region_by_centroid(self, p):
        d = np.linalg.norm(self.centroids - p, axis=1)
        return int(np.argmin(d))

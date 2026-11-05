"""Ground-truth world: random-waypoint objects, cameras, and the detection model."""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


class FeatureBank:
    """Appearance prototypes z'. Synthetic unit vectors on the D-sphere; a
    fraction rho_la of objects are look-alikes whose prototype lies at angle
    theta_la from an earlier object's prototype. Optionally samples from a
    bank of precomputed encoder features (.npy, shape [n, D])."""

    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng
        self.bank = None
        if cfg.feature_file:
            b = np.load(cfg.feature_file).astype(np.float32)
            self.bank = _unit(b)
        self.recent = []

    def sample(self):
        c, rng = self.cfg, self.rng
        if self.bank is not None:
            z = self.bank[rng.integers(len(self.bank))]
        elif self.recent and rng.random() < c.rho_la:
            base = self.recent[rng.integers(len(self.recent))]
            u = rng.standard_normal(c.D).astype(np.float32)
            u -= u.dot(base) * base
            u /= np.linalg.norm(u)
            th = math.radians(c.theta_la_deg)
            z = math.cos(th) * base + math.sin(th) * u
        else:
            z = _unit(rng.standard_normal(c.D)).astype(np.float32)
        z = z.astype(np.float32)
        self.recent.append(z)
        if len(self.recent) > 512:
            self.recent.pop(0)
        return z


class World:
    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng
        self.L = cfg.L
        self.features = FeatureBank(cfg, rng)
        self.next_gt = 0
        cap = int(cfg.N_bar * 2) + 64
        self.pos = np.zeros((cap, 2))
        self.vel = np.zeros((cap, 2))
        self.dest = np.zeros((cap, 2))
        self.speed = np.zeros(cap)
        self.exiting = np.zeros(cap, bool)
        self.alive = np.zeros(cap, bool)
        self.gt = np.full(cap, -1, np.int64)
        self.proto = np.zeros((cap, cfg.D), np.float32)
        # arrival rate so that the steady-state mean is N_bar
        legs = 1.0 / cfg.p_exit + 1.0
        mean_leg = 0.5214 * self.L
        e_inv_v = math.log(cfg.v_max / cfg.v_min) / (cfg.v_max - cfg.v_min) if cfg.v_max > cfg.v_min else 1 / cfg.v_min
        lifetime = legs * mean_leg * e_inv_v
        self.lam_in = cfg.N_bar / lifetime
        for _ in range(int(round(cfg.N_bar))):
            self._spawn(self.rng.uniform(0, self.L, 2))

    # ---------------- objects ----------------
    def _slot(self):
        free = np.flatnonzero(~self.alive)
        if len(free) == 0:
            n = len(self.alive)
            for name in ("pos", "vel", "dest"):
                setattr(self, name, np.vstack([getattr(self, name), np.zeros((n, 2))]))
            self.speed = np.concatenate([self.speed, np.zeros(n)])
            self.exiting = np.concatenate([self.exiting, np.zeros(n, bool)])
            self.alive = np.concatenate([self.alive, np.zeros(n, bool)])
            self.gt = np.concatenate([self.gt, np.full(n, -1, np.int64)])
            self.proto = np.vstack([self.proto, np.zeros((n, self.cfg.D), np.float32)])
            return n
        return free[0]

    def _boundary_point(self):
        s = self.rng.integers(4)
        u = self.rng.uniform(0, self.L)
        return np.array([(u, 0.0), (u, self.L), (0.0, u), (self.L, u)][s])

    def _spawn(self, p):
        i = self._slot()
        c = self.cfg
        self.pos[i] = p
        self.dest[i] = self.rng.uniform(0, self.L, 2)
        self.speed[i] = self.rng.uniform(c.v_min, c.v_max)
        self.exiting[i] = False
        self.alive[i] = True
        self.gt[i] = self.next_gt
        self.next_gt += 1
        self.proto[i] = self.features.sample()
        self._set_vel(np.array([i]))

    def _set_vel(self, idx):
        d = self.dest[idx] - self.pos[idx]
        h = np.arctan2(d[:, 1], d[:, 0]) + self.rng.normal(0, self.cfg.heading_sigma, len(idx))
        self.vel[idx] = self.speed[idx, None] * np.stack([np.cos(h), np.sin(h)], 1)

    def step(self, dt):
        a = np.flatnonzero(self.alive)
        self.pos[a] += self.vel[a] * dt
        dist = np.linalg.norm(self.dest[a] - self.pos[a], axis=1)
        arrived = a[dist < self.speed[a] * dt]
        for i in arrived:
            if self.exiting[i]:
                self.alive[i] = False
            elif self.rng.random() < self.cfg.p_exit:
                self.dest[i] = self._boundary_point()
                self.exiting[i] = True
            else:
                self.dest[i] = self.rng.uniform(0, self.L, 2)
        a = np.flatnonzero(self.alive)
        self.pos[a] = np.clip(self.pos[a], 0, self.L)
        self._set_vel(a)
        for _ in range(self.rng.poisson(self.lam_in * dt)):
            self._spawn(self._boundary_point())

    def snapshot(self):
        a = np.flatnonzero(self.alive)
        return self.gt[a].copy(), self.pos[a].copy()

    # ---------------- detections ----------------
    def detect(self, cam_pos, cam_phase, cam_mask):
        """Detections of all cameras for the current frame.

        Returns per-pair arrays (cam, gt, t_offset, p, z, c). Each camera
        captures at t_k + phase_i; positions are extrapolated to that time."""
        c, rng = self.cfg, self.rng
        a = np.flatnonzero(self.alive)
        if len(a) == 0 or not cam_mask.any():
            return None
        cams = np.flatnonzero(cam_mask)
        otree = cKDTree(self.pos[a])
        ctree = cKDTree(cam_pos[cams])
        sp = ctree.sparse_distance_matrix(otree, c.r + c.v_max * c.dt, output_type="coo_matrix")
        ci, oi = cams[sp.row], a[sp.col]
        if len(ci) == 0:
            return None
        ph = cam_phase[ci]
        x = self.pos[oi] + self.vel[oi] * ph[:, None]
        d = np.linalg.norm(x - cam_pos[ci], axis=1)
        keep = d <= c.r
        pd = 1.0 - c.beta * (d / c.r) ** 2
        keep &= rng.random(len(d)) < pd
        ci, oi, ph, x, d = ci[keep], oi[keep], ph[keep], x[keep], d[keep]
        n = len(ci)
        if n == 0:
            return None
        conf = np.clip(1.0 - d / c.r + rng.normal(0, c.sigma_c, n), 0.0, 1.0)
        sp_ = c.sigma_p0 + c.sigma_p1 * (1 - conf)
        p = x + rng.standard_normal((n, 2)) * sp_[:, None]
        sz = (c.sigma_z0 + c.sigma_z1 * (1 - conf)).astype(np.float32)
        z = self.proto[oi] + rng.standard_normal((n, c.D), dtype=np.float32) * sz[:, None]
        order = np.argsort(ci, kind="stable")
        return (ci[order], self.gt[oi][order], ph[order], p[order], z[order], conf[order],
                x[order])


def place_cameras(cfg, rng, M=None):
    M = cfg.M if M is None else M
    L = cfg.L
    if cfg.placement == "uniform":
        return rng.uniform(0, L, (M, 2))
    g = int(math.ceil(math.sqrt(M)))
    s = L / g
    xs, ys = np.meshgrid(np.arange(g), np.arange(g))
    pts = (np.stack([xs.ravel(), ys.ravel()], 1)[:M] + 0.5) * s
    pts += rng.uniform(-0.3 * s, 0.3 * s, pts.shape)
    return np.clip(pts, 0, L)

"""All system and simulation parameters (Table "Main parameters" in the report)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict, replace


@dataclass
class Config:
    seed: int = 0
    # ---------------- scale ----------------
    M: int = 100                    # cameras
    L: float | None = None          # world side [m]; default 1000*sqrt(M/100)
    N_bar: float | None = None      # mean number of objects; default = M
    duration: float = 60.0          # simulated seconds
    # ---------------- objects ----------------
    v_min: float = 5.0
    v_max: float = 15.0
    heading_sigma: float = 0.05     # rad, per-step heading perturbation
    p_exit: float = 0.3
    # ---------------- cameras / detector ----------------
    r: float = 50.0
    placement: str = "jittered_grid"   # or "uniform"
    beta: float = 0.5
    p_timeout: float = 0.01
    sigma_c: float = 0.1
    sigma_p0: float = 0.5
    sigma_p1: float = 2.0
    D: int = 256
    sigma_z0: float = 0.02          # per-dimension appearance noise (unit-norm prototypes)
    sigma_z1: float = 0.04
    rho_la: float = 0.1             # fraction of look-alike objects
    theta_la_deg: float = 30.0      # angle between look-alike prototypes
    feature_file: str | None = None  # optional .npy bank of precomputed encoder features
    # ---------------- network & clocks ----------------
    delta0: float = 0.002
    delay_median: float = 0.003     # log-normal component median [s]
    delay_sigma: float = 0.6
    p_loss: float = 0.01
    theta_max: float = 0.05         # max clock offset [s]
    rho_max: float = 50e-6          # max clock drift
    # ---------------- time ----------------
    dt: float = 0.1
    W: float = 0.05                 # center deadline after step end
    W_a: float = 0.10               # aggregator deadline after center deadline
    T_sync: float = 5.0
    # ---------------- regions ----------------
    S: int = 20
    R: int = 3
    # ---------------- tracking ----------------
    alpha: float = 0.5
    gamma: float = 9.21
    s_min: float = 0.3
    s_reid: float = 0.5
    n_c: int = 2
    n_w: int = 3
    n_d: int = 5
    T_coast_max: float = 2.0
    T_lost: float = 10.0
    lam: float = 0.3
    sigma_a: float = 3.0            # Kalman process (acceleration) noise
    z_resend: float = 0.05          # cosine change that triggers re-sending z
    # ---------------- handoff ----------------
    handoff: bool = True
    h: float = 5.0
    m: float = 10.0
    T_h: float = 5.0
    T_reoffer: float = 0.5
    T_young: float = 1.0
    T_retry: float = 0.3
    # ---------------- raft / failures ----------------
    replication: bool = True
    election_timeout: float = 0.3
    raft_hb: float = 0.05
    T_snap: int = 50
    n_hb: int = 10
    T_hello: float = 0.5
    T_center_timeout: float = 1.5
    T_dir: float = 1.0
    T_empty: float = 5.0
    # ---------------- aggregator ----------------
    T_stale: float = 2.0
    dedup: bool = True
    C_dup: float = 0.3
    n_dup: int = 5
    dup_radius: float = 10.0
    global_reid: bool = True
    T_gl: float = 30.0              # aggregator vanished-pool lifetime
    # ---------------- computation cost ----------------
    cost_model: str = "measured"    # "measured" | "none"
    kappa: float = 1.0
    # ---------------- faults / churn ----------------
    faults: list = field(default_factory=list)   # [(t, kind, arg, ...)]
    member_crash_rate: float = 0.0  # crashes per camera per hour
    member_mttr: float = 10.0       # mean time to recovery [s] (0 = crash-stop)
    join_rate: float = 0.0          # cameras joining per second
    leave_rate: float = 0.0         # cameras leaving per second
    # ---------------- mode / output ----------------
    centralized: bool = False       # baseline: one region, one node, no Raft
    trace: str | None = None
    trace_centers: bool = True
    warmup: float = 5.0             # seconds excluded from metrics

    def __post_init__(self):
        if self.L is None:
            self.L = 1000.0 * math.sqrt(self.M / 100.0)
        if self.N_bar is None:
            self.N_bar = float(self.M)
        if self.centralized:
            # baseline: one node fuses everything; its report is never dropped
            # by an aggregator deadline, so its cost shows up as latency
            self.S = self.M
            self.R = 1
            self.replication = False
            self.W_a = max(self.W_a, 10.0)
        if not self.replication:
            self.R = 1

    @property
    def K(self) -> int:
        return max(1, math.ceil(self.M / self.S))

    def sigma_p(self, c):
        return self.sigma_p0 + self.sigma_p1 * (1.0 - c)

    def sigma_z(self, c):
        return self.sigma_z0 + self.sigma_z1 * (1.0 - c)

    def to_dict(self):
        return asdict(self)

    def with_(self, **kw) -> "Config":
        d = asdict(self)
        # let __post_init__ recompute derived sizes when M changes
        if "M" in kw:
            d.setdefault("L", None)
            if "L" not in kw:
                d["L"] = None
            if "N_bar" not in kw:
                d["N_bar"] = None
        d.update(kw)
        return Config(**d)

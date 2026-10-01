from __future__ import annotations

import os
os.environ["HUGGINGFACE_HUB_CACHE"] = ".hf_cache"
os.environ["HF_HUB_DISABLE_XET"] = "1"

from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

import math
import warnings
from pathlib import Path
from typing import Dict, Optional, Union, Callable

import numpy as np
import torch
from scipy.interpolate import interp1d, PchipInterpolator
from diffusers import DPMSolverSinglestepScheduler as _DefaultSingle
from diffusers import StableDiffusionPipeline
from scipy.ndimage import gaussian_filter1d

from xue_step_optim import NoiseScheduleVP, StepOptim

# ─────────────────────────────────────────────────────────────────────────────
# VP-SDE helpers
# ─────────────────────────────────────────────────────────────────────────────

def _alpha(lam: float) -> float:
    return 1.0 / math.sqrt(1.0 + math.exp(-2.0 * lam))

def _sigma_vp(lam: float) -> float:
    return 1.0 / math.sqrt(1.0 + math.exp(2.0 * lam))

# ─────────────────────────────────────────────────────────────────────────────
# φ builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_phi_fn(
    rho_s: np.ndarray,
    rho_vals: np.ndarray,
    quad_pts: int = 64,
    s_extend_factor: float = 3.0,
) -> Callable[[float, float], float]:
    rho_s    = np.asarray(rho_s,    dtype=np.float64)
    rho_vals = np.asarray(rho_vals, dtype=np.float64)
    rho_vals = np.clip(rho_vals, 1e-12, None)

    if rho_s[0] > 1e-10:
        rho_s    = np.concatenate([[0.0], rho_s])
        rho_vals = np.concatenate([[rho_vals[0]], rho_vals])

    log_rho    = np.log(rho_vals)
    rho_interp = PchipInterpolator(rho_s, log_rho, extrapolate=False)

    s_max  = rho_s[-1] * s_extend_factor
    n_fine = max(2000, int(s_max / (rho_s[-1] + 1e-12) * len(rho_s) * 4))
    s_fine = np.linspace(0.0, s_max, n_fine)

    log_rho_fine = rho_interp(s_fine)
    nan_mask = np.isnan(log_rho_fine)
    if nan_mask.any():
        last  = int(np.where(~nan_mask)[0][-1])
        slope = (log_rho_fine[last] - log_rho_fine[max(last - 1, 0)]) / (s_fine[1] - s_fine[0] + 1e-30)
        slope = min(slope, -1e-3)
        for idx in np.where(nan_mask)[0]:
            log_rho_fine[idx] = log_rho_fine[last] + slope * (s_fine[idx] - s_fine[last])
    rho_fine = np.exp(np.clip(log_rho_fine, -30.0, 0.0))

    ds     = s_fine[1] - s_fine[0]
    R_fine = np.zeros(n_fine)
    R_fine[1:] = np.cumsum(0.5 * (rho_fine[:-1] + rho_fine[1:]) * ds)
    R_interp = interp1d(s_fine, R_fine, kind="linear", bounds_error=False,
                        fill_value=(0.0, float(R_fine[-1])))
    _trapz = getattr(np, "trapezoid", None) or np.trapz

    def phi_fn(a: float, b: float) -> float:
        if a <= 0.0 or b <= 0.0:
            return 0.0
        a_int = min(a, b); b_int = max(a, b)
        u = np.linspace(0.0, a_int, quad_pts)
        return float(_trapz(R_interp(u) + R_interp(b_int - u), u))

    return phi_fn


def _build_phi_res_fn(
    rho_s: np.ndarray,
    rho_vals: np.ndarray,
    rho_infty: float,
    quad_pts: int = 64,
) -> Callable[[float, float], float]:
    phi_full = _build_phi_fn(rho_s, rho_vals, quad_pts)
    def phi_res_fn(a: float, b: float) -> float:
        return phi_full(a, b) - rho_infty * a * b
    return phi_res_fn


def build_g_alpha_fn(
    base_stats_path: str,
    kappa_npz_path:  str,
    alpha: float,
    smooth_sigma_lam: float = 0.4,
    kappa_key: str = "kappa_values",
):
    base     = np.load(base_stats_path)
    kap      = np.load(kappa_npz_path)
    lam_grid = base["lambda_grid"].astype(np.float64)
    g_vals   = base["g_values"].astype(np.float64)

    kappa_raw = (kap[kappa_key] if kappa_key in kap else kap["kappa"]).astype(np.float64)
    kappa_raw = np.clip(kappa_raw, 1e-3, None)

    if "lambda_grid" in kap:
        kap_lam = kap["lambda_grid"].astype(np.float64)
        # default: increasing
        sort_idx  = np.argsort(kap_lam)
        kap_lam   = kap_lam[sort_idx]
        kappa_raw = kappa_raw[sort_idx]
        uniq = np.concatenate([[True], np.diff(kap_lam) > 1e-10])
        kap_lam   = kap_lam[uniq]
        kappa_raw = kappa_raw[uniq]
        kappa_raw = np.interp(lam_grid, kap_lam, kappa_raw)
    else:
        if len(kappa_raw) != len(lam_grid):
            raise ValueError(
                f"kappa length {len(kappa_raw)} != lam_grid length {len(lam_grid)}, "
                "and no lambda_grid in kappa npz to interpolate."
            )

    # ── log-space smoothing  ─────────────────────────────────────────────────────
    dlam      = float(np.median(np.diff(lam_grid)))
    sigma_pts = max(1.0, smooth_sigma_lam / dlam)
    kappa_smooth = np.exp(
        gaussian_filter1d(np.log(kappa_raw), sigma=sigma_pts, mode="nearest")
    )

    # ── g_α ────────────────────────────────────────────────────────────────
    kappa_alpha = kappa_smooth ** alpha
    mean_ka     = float(np.mean(kappa_alpha))
    g_alpha     = g_vals * kappa_alpha / (mean_ka + 1e-12)

    sort_idx = np.argsort(lam_grid)
    lam_s    = lam_grid[sort_idx]
    g_s      = g_alpha[sort_idx]
    uniq     = np.concatenate([[True], np.diff(lam_s) > 1e-10])
    interp   = PchipInterpolator(lam_s[uniq], g_s[uniq], extrapolate=True)

    return lambda lam: float(interp(lam)), kappa_smooth
# ─────────────────────────────────────────────────────────────────────────────
# ε_k and Γ
# ─────────────────────────────────────────────────────────────────────────────

def _compute_epsilon(lam_prev, lam_curr, g_fn, r1=0.5):
    h     = lam_curr - lam_prev
    lam_s = lam_prev + r1 * h
    I_full = math.exp(-lam_prev) - math.exp(-lam_curr)
    I_half = math.exp(-lam_prev) - math.exp(-lam_s)
    al_p   = _alpha(lam_prev); al_s = _alpha(lam_s)
    a_k    = (1.0 - 1.0 / (2.0 * r1)) * al_p * I_full
    b_k    = -(al_s / (2.0 * r1)) * I_full
    phi_k  = 1.0 - al_p * I_half * g_fn(lam_prev)
    return a_k * g_fn(lam_prev) + b_k * g_fn(lam_s) * phi_k

def _compute_gamma(eps, lambdas):
    N = len(eps); Gamma = np.ones(N)
    for k in range(N - 2, -1, -1):
        Gamma[k] = Gamma[k + 1] * (1.0 + eps[k + 1])
    return Gamma

def _c_j(lam_prev, lam_curr, g_fn, r1):
    h     = lam_curr - lam_prev
    lam_s = lam_prev + r1 * h
    al_s  = _alpha(lam_s)
    I_full = math.exp(-lam_prev) - math.exp(-lam_curr)
    return (g_fn(lam_prev) / (2.0 * r1)) * al_s * I_full


# ─────────────────────────────────────────────────────────────────────────────
# Per-step cost components
# ─────────────────────────────────────────────────────────────────────────────

def _compute_A_j(lam_prev, lam_curr, sigma2_fn, g_fn, r1=0.5, n_quad=32):
    h = lam_curr - lam_prev; lam_s = lam_prev + r1 * h
    cj = _c_j(lam_prev, lam_curr, g_fn, r1)

    mu_full = np.linspace(lam_prev, lam_curr, n_quad)
    sig_full = np.sqrt(np.maximum([sigma2_fn(float(m)) for m in mu_full], 0.0))
    int_full = float(np.trapezoid(np.exp(-mu_full) * sig_full, mu_full))

    n_half = max(2, n_quad // 2)
    mu_half = np.linspace(lam_prev, lam_s, n_half)
    sig_half = np.sqrt(np.maximum([sigma2_fn(float(m)) for m in mu_half], 0.0))
    int_half = float(np.trapezoid(np.exp(-mu_half) * sig_half, mu_half))

    return int_full - cj * int_half


def _compute_V_res(lam_prev, lam_curr, g_fn, sigma2_fn, phi_res_fn, r1=0.5):
    h = lam_curr - lam_prev
    lam_bar = 0.5 * (lam_prev + lam_curr)
    cj = _c_j(lam_prev, lam_curr, g_fn, r1)
    prefactor = sigma2_fn(lam_bar) * math.exp(-2.0 * lam_prev)
    phi_h     = phi_res_fn(h, h)
    phi_r1h   = phi_res_fn(r1 * h, r1 * h)
    phi_r1h_h = phi_res_fn(r1 * h, h)
    Q_res = phi_h - 2.0 * cj * phi_r1h_h + cj**2 * phi_r1h
    return prefactor * max(Q_res, 0.0)


def _compute_D_j(lam_prev, lam_curr, g_fn, sigma2_gpp_fn, r1=0.5):
    h = lam_curr - lam_prev
    lam_bar = 0.5 * (lam_prev + lam_curr)
    # Taylor expansion of DPM-Solver-2 quadrature error (O(h^3) LTE):
    # δ_disc ≈ α_{t_j} · e^{-λ̄_j} · (1/6 - r1/4) · h^3 · g''(λ̄_j)
    # D_j = E[||δ_disc||^2/d] = α²_{t_j} · e^{-2λ̄_j} · (1/6 - r1/4)^2 · h^6 · σ²_{g''}
    coeff = (1.0/6.0 - r1/4.0) ** 2   # = 1/576 for r1=0.5
    return (math.exp(-2.0 * lam_bar) * sigma2_gpp_fn(lam_bar) * coeff * h**6)

# ─────────────────────────────────────────────────────────────────────────────
# Cost functional  ──  Piecewise kernel
# ─────────────────────────────────────────────────────────────────────────────

def _cost_functional(
    lambdas, g_fn, sigma2_fn, sigma2_gpp_fn,
    get_phi_res, ell_gpp,
    r1, n_quad, barrier_weight, w_rank1, w_vres, w_disc,
    rho_inf_fn: Callable,        # require：per-step sqrt(ρ_∞(λ_bar))
    banded_scale_fn: Callable,   # require：per-step 1+2·C_ρ(λ_bar)
):
    N     = len(lambdas) - 1
    h_arr = np.diff(lambdas)
    eps   = np.array([_compute_epsilon(lambdas[k-1], lambdas[k], g_fn, r1)
                      for k in range(1, N+1)])
    Gamma = _compute_gamma(eps, lambdas)

    A    = np.zeros(N)
    Vres = np.zeros(N)
    D    = np.zeros(N)
    banded = np.zeros(N)
    sqrt_rho = np.zeros(N)

    for k in range(N):
        lp, lc  = lambdas[k], lambdas[k+1]
        lam_bar = 0.5 * (lp + lc)
        phi_res_fn, rho_inf_local = get_phi_res(lam_bar)

        A[k]    = _compute_A_j(lp, lc, sigma2_fn, g_fn, r1, n_quad)
        Vres[k] = _compute_V_res(lp, lc, g_fn, sigma2_fn, phi_res_fn, r1)
        D[k]    = _compute_D_j(lp, lc, g_fn, sigma2_gpp_fn, r1)

        banded[k]   = float(banded_scale_fn(lam_bar))
        sqrt_rho[k] = math.sqrt(max(rho_inf_local, 0.0))

    alpha_tN2 = _alpha(lambdas[-1]) ** 2

    rank1_term = w_rank1 * float(np.dot(sqrt_rho * A, Gamma)) ** 2
    vres_term  = w_vres  * float(np.dot(banded * Vres, Gamma**2))
    disc_term  = w_disc  * float(np.dot(banded * D,    Gamma**2))

    main_cost = alpha_tN2 * (rank1_term + vres_term + disc_term)

    if barrier_weight > 0.0:
        h_uniform = (lambdas[-1] - lambdas[0]) / N
        return main_cost - barrier_weight * float(np.sum(np.log(h_arr / h_uniform)))
    return main_cost


# ─────────────────────────────────────────────────────────────────────────────
# Stats helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_rho_infty_and_ell(rho_s, rho_vals):
    drho     = np.abs(np.diff(rho_vals))
    rel_drho = drho / (np.maximum(np.abs(rho_vals[:-1]), 0.05))
    MIN_FLAT = max(5, len(rho_vals) // 10)
    plateau_start = None
    for i in range(len(rel_drho) - MIN_FLAT + 1):
        if np.all(rel_drho[i : i + MIN_FLAT] < 0.02):
            plateau_start = i; break
    if plateau_start is None:
        warnings.warn("[BornSchedule] No plateau found — extend rho_s range.", UserWarning)
        plateau_start = int(0.5 * len(rho_vals))

    rho_at_start = rho_vals[plateau_start]; plateau_end = len(rho_vals)
    for i in range(plateau_start + MIN_FLAT, len(rho_vals)):
        if rho_vals[i] < rho_at_start * 0.95:
            plateau_end = i; break

    mid = plateau_start + (plateau_end - plateau_start) // 2
    plateau_vals = rho_vals[plateau_start:mid]
    rho_infty    = float(np.clip(np.median(plateau_vals), 0.0, 1.0 - 1e-6))
    cv = np.std(plateau_vals) / (np.mean(plateau_vals) + 1e-8)
    if cv > 0.05:
        warnings.warn(f"[BornSchedule] Plateau CV={cv:.3f} high, ρ_∞ unreliable.", UserWarning)

    denom   = max(1.0 - rho_infty, 1e-8)
    rho_res = np.maximum((rho_vals - rho_infty) / denom, 0.0)
    cross   = np.where(rho_res <= 1.0 / math.e)[0]
    ell_corr = float(rho_s[cross[0]]) if len(cross) > 0 else float(rho_s[-1])
    return rho_infty, ell_corr


def _compute_C_rho_res(rho_s, rho_vals, rho_infty):
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    rho_res_vals = np.maximum(np.asarray(rho_vals, dtype=np.float64) - rho_infty, 0.0)
    return float(_trapz(rho_res_vals, np.asarray(rho_s, dtype=np.float64)))


def _alpha_to_lambdas(alpha, lam0, lamN):
    span = lamN - lam0
    a    = alpha - alpha.max()
    h    = np.exp(a) / np.exp(a).sum() * span
    interior = lam0 + np.cumsum(h)[:-1]
    return np.concatenate([[lam0], interior, [lamN]])


MODEL_STATS_REGISTRY: Dict[str, dict] = {}

def register_model_stats(model_name, stats):
    MODEL_STATS_REGISTRY[model_name] = stats

def load_model_stats_from_file(model_name, path):
    data    = np.load(path)
    required = ["lambda_grid", "g_values", "sigma2_values",
                "lambda_min", "lambda_max", "rho_s", "rho_values",
                "rho_s_gpp", "rho_values_gpp"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Stats file '{path}' missing keys: {missing}")
    stats = {k: np.asarray(data[k], dtype=np.float64) for k in required}
    stats["lambda_min"] = float(data["lambda_min"])
    stats["lambda_max"] = float(data["lambda_max"])

    if "sigma2_gpp_values" in data:
        stats["sigma2_gpp_values"] = np.asarray(data["sigma2_gpp_values"], dtype=np.float64)
    elif "sigma2_gp_values" in data:
        stats["sigma2_gpp_values"] = np.asarray(data["sigma2_gp_values"], dtype=np.float64)
    else:
        warnings.warn("[BornSchedule] No g'' stats found; D_j = 0.", UserWarning)
        stats["sigma2_gpp_values"] = np.zeros_like(stats["sigma2_values"])

    for k in ["seg_lambda_centers", "seg_rho_s", "seg_rho_mat"]:
        if k in data:
            stats[k] = np.asarray(data[k], dtype=np.float64)

    register_model_stats(model_name, stats)
    return stats

def make_alpha_sweep_stats(
    base_stats_path: str,   # nonstat stats npz
    kappa_npz_path:  str,   # nonstat_kappa.npz
    alphas = (0.0, 0.25, 0.5, 0.75, 1.0),
    smooth_sigma_lam: float = 0.4,
):
    """
    return dict: alpha → stats_override dict
    """
    base_data = np.load(base_stats_path)
    base_stats = {k: np.asarray(base_data[k]) for k in base_data.files}
    base_stats["lambda_min"] = float(base_data["lambda_min"])
    base_stats["lambda_max"] = float(base_data["lambda_max"])

    sweep = {}
    for a in alphas:
        g_fn, kappa_smooth = build_g_alpha_fn(
            base_stats_path, kappa_npz_path, alpha=a,
            smooth_sigma_lam=smooth_sigma_lam
        )
        lam_grid  = base_stats["lambda_grid"]
        g_alpha_vals = np.array([g_fn(l) for l in lam_grid])

        stats_a = dict(base_stats)              # shallow copy
        stats_a["g_values"] = g_alpha_vals      # override g only
        sweep[a] = stats_a

        print(f"  α={a:.2f}  g_α ∈ [{g_alpha_vals.min():.4f}, "
              f"{g_alpha_vals.max():.4f}]  mean={g_alpha_vals.mean():.4f}")

    return sweep, kappa_smooth


class OptimalSchedule:
    order = 2

    def __init__(
        self,
        model_name: str,
        r1: float = 0.5,
        max_iter: int = 600,
        lr: float = 1e-2,
        lr_decay: float = 0.9995,
        tol: float = 1e-7,
        fd_eps: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        quad_pts: int = 64,
        barrier_weight_scale: float = 0.005,
        w_rank1: float = 1.0,
        w_vres:  float = 1.0,
        w_disc:  float = 1.0,
        verbose: bool = True,
        stats_override: Optional[dict] = None,
    ):
        self.model_name           = model_name
        self.r1                   = r1
        self.max_iter             = max_iter
        self.lr                   = lr
        self.lr_decay             = lr_decay
        self.tol                  = tol
        self.fd_eps               = fd_eps
        self.beta1                = beta1
        self.beta2                = beta2
        self.quad_pts             = quad_pts
        self.barrier_weight_scale = barrier_weight_scale
        self.verbose              = verbose
        self.w_rank1              = w_rank1
        self.w_vres               = w_vres
        self.w_disc               = w_disc

        stats = stats_override if stats_override is not None else self._load_stats(model_name)
        self._init_from_stats(stats)
        self._lambdas_opt: Optional[np.ndarray] = None

    def _load_stats(self, model_name):
        safe = model_name.replace("/", "--").replace(":", "--")

        base_dirs = [
            Path("stats_sd_v1.5"),
            Path(os.environ.get("OPT_SCHEDULE_STATS_DIR", ".")),
            Path.home() / ".cache" / "opt_schedule",
        ]

        for base in base_dirs:
            fpath = base / f"{safe}_cfg7.5_nonstat.npz"
            if fpath.exists():
                warnings.warn(f"[BornSchedule] Loading stats from {fpath}", UserWarning, stacklevel=3)
                return load_model_stats_from_file(model_name, fpath)

        if model_name in MODEL_STATS_REGISTRY:
            return MODEL_STATS_REGISTRY[model_name]

        raise FileNotFoundError(f"[BornSchedule] No stats found for '{model_name}'.")

    def _init_from_stats(self, stats):
        lam_grid        = np.asarray(stats["lambda_grid"],       dtype=np.float64)
        g_vals          = np.asarray(stats["g_values"],          dtype=np.float64)
        sigma2_vals     = np.asarray(stats["sigma2_values"],     dtype=np.float64)
        sigma2_gpp_vals = np.asarray(stats["sigma2_gpp_values"], dtype=np.float64)
        rho_s           = np.asarray(stats["rho_s"],             dtype=np.float64)
        rho_vals        = np.asarray(stats["rho_values"],        dtype=np.float64)
        rho_s_gpp       = np.asarray(stats["rho_s_gpp"],         dtype=np.float64)
        rho_vals_gpp    = np.asarray(stats["rho_values_gpp"],    dtype=np.float64)

        self.lambda_min = float(stats["lambda_min"])
        self.lambda_max = float(stats["lambda_max"])

        kw = dict(kind="linear", bounds_error=False)
        self._g_fn         = interp1d(lam_grid, g_vals,
                                      fill_value=(g_vals[0], g_vals[-1]), **kw)
        self._sigma2_fn    = interp1d(lam_grid, sigma2_vals,
                                      fill_value=(sigma2_vals[0], sigma2_vals[-1]), **kw)
        self._sigma2gpp_fn = interp1d(lam_grid, sigma2_gpp_vals,
                                      fill_value=(sigma2_gpp_vals[0], sigma2_gpp_vals[-1]), **kw)

        self.rho_infty, self.ell_corr = _extract_rho_infty_and_ell(rho_s, rho_vals)
        self.C_rho_res = _compute_C_rho_res(rho_s, rho_vals, self.rho_infty)

        from scipy.integrate import trapezoid as scipy_trapz
        rho_norm     = rho_vals_gpp / (rho_vals_gpp[0] + 1e-30)
        self.ell_gpp = float(scipy_trapz(rho_norm, rho_s_gpp))
      
        if not ("seg_lambda_centers" in stats and "seg_rho_mat" in stats):
            raise ValueError(
                "[EP-flow] Stats file must contain segmented kernel data "
                "('seg_lambda_centers', 'seg_rho_s', 'seg_rho_mat'). "
                "Run stats collection with non-stationary mode enabled."
            )

        seg_centers = np.asarray(stats["seg_lambda_centers"], dtype=np.float64)
        seg_rho_s   = np.asarray(stats["seg_rho_s"],          dtype=np.float64)
        seg_rho_mat = np.asarray(stats["seg_rho_mat"],        dtype=np.float64)
        tail = max(5, seg_rho_mat.shape[1] // 10)
        rho_infty_per_seg = np.median(seg_rho_mat[:, -tail:], axis=1)

        def _get_phi_res(lam_bar):
            K = len(seg_centers)
            if K == 1 or lam_bar <= seg_centers[0]:
                w = np.zeros(K); w[0] = 1.0
            elif lam_bar >= seg_centers[-1]:
                w = np.zeros(K); w[-1] = 1.0
            else:
                idx = int(np.searchsorted(seg_centers, lam_bar)) - 1
                idx = np.clip(idx, 0, K - 2)
                t   = (lam_bar - seg_centers[idx]) / (seg_centers[idx+1] - seg_centers[idx] + 1e-12)
                w   = np.zeros(K); w[idx] = 1.0 - t; w[idx+1] = t
            rho_local     = w @ seg_rho_mat
            rho_inf_local = float(w @ rho_infty_per_seg)
            phi = _build_phi_fn(seg_rho_s, rho_local, self.quad_pts)
            return (lambda a, b: phi(a, b) - rho_inf_local * a * b), rho_inf_local

        self._get_phi_res = _get_phi_res

        _trapz = getattr(np, "trapezoid", None) or np.trapz
        C_rho_per_seg = np.array([
            float(_trapz(
                np.maximum(seg_rho_mat[k] - rho_infty_per_seg[k], 0.),
                seg_rho_s
            ))
            for k in range(len(seg_centers))
        ])
        banded_per_seg = 1.0 + 2.0 * C_rho_per_seg

        if self.verbose:
            for k in range(len(seg_centers)):
                print(f"  [BornSchedule] seg{k}  λ_c={seg_centers[k]:.3f}"
                      f"  ρ_∞={rho_infty_per_seg[k]:.4f}"
                      f"  C_ρ={C_rho_per_seg[k]:.4f}"
                      f"  banded={banded_per_seg[k]:.4f}")

        _bs_interp = interp1d(
            seg_centers, banded_per_seg,
            kind="linear", bounds_error=False,
            fill_value=(banded_per_seg[0], banded_per_seg[-1])
        )
        self._banded_scale_fn = lambda lam: float(np.clip(_bs_interp(lam), 1.0, None))

        _ri_interp = interp1d(
            seg_centers, rho_infty_per_seg,
            kind="linear", bounds_error=False,
            fill_value=(rho_infty_per_seg[0], rho_infty_per_seg[-1])
        )
        self._rho_inf_fn = lambda lam: float(np.clip(_ri_interp(lam), 0.0, 1.0))

        if self.verbose:
            print(f"  [EP-flow] ρ_∞(global)={self.rho_infty:.4f}  "
                  f"ℓ_corr={self.ell_corr:.4f}  C_ρ={self.C_rho_res:.4f}  "
                  f"λ∈[{self.lambda_min:.3f},{self.lambda_max:.3f}]")

    def g_fn(self,          lam): return float(self._g_fn(lam))
    def sigma2_fn(self,     lam): return max(float(self._sigma2_fn(lam)), 0.0)
    def sigma2_gpp_fn(self, lam): return max(float(self._sigma2gpp_fn(lam)), 0.0)

    @staticmethod
    def lambda_to_sigma(lam):
        return 1.0 / np.sqrt(1.0 + np.exp(2.0 * np.asarray(lam, dtype=np.float64)))

    def _kw(self, barrier_weight=0.0):
        return dict(
            g_fn=self.g_fn, sigma2_fn=self.sigma2_fn,
            sigma2_gpp_fn=self.sigma2_gpp_fn, ell_gpp=self.ell_gpp,
            get_phi_res=self._get_phi_res,
            r1=self.r1, n_quad=self.quad_pts,
            barrier_weight=barrier_weight,
            w_rank1=self.w_rank1, w_vres=self.w_vres, w_disc=self.w_disc,
            banded_scale_fn=self._banded_scale_fn,
            rho_inf_fn=self._rho_inf_fn,
        )

    def _cost(self, lambdas, barrier_weight=0.0):
        return _cost_functional(lambdas, **self._kw(barrier_weight))

    def _optimise(self, N: int, init_lambdas: Optional[np.ndarray] = None) -> np.ndarray:
        lam0 = self.lambda_min
        lamN = self.lambda_max

        cost_uniform   = self._cost(np.linspace(lam0, lamN, N+1), barrier_weight=0.0)
        barrier_weight = self.barrier_weight_scale * cost_uniform / math.sqrt(N)
        if self.verbose:
            print(f"  [BornSchedule] N={N}  C(uniform)={cost_uniform:.4e}  "
                  f"barrier_weight={barrier_weight:.3e}")

        if init_lambdas is not None:
            init_lambdas = np.asarray(init_lambdas, dtype=np.float64)
            assert len(init_lambdas) == N + 1, \
                f"init_lambdas length= {len(init_lambdas)} != N+1={N+1}"
            h_init = np.diff(init_lambdas)
            assert np.all(h_init > 0), "init_lambdas strictly increasing"
          
            alpha = np.log(h_init)
            alpha -= alpha.max()
            init_tag = "custom-init"
        else:
            alpha = np.zeros(N)
            init_tag = "uniform-init"

        m = np.zeros_like(alpha); v = np.zeros_like(alpha)
        lr = self.lr; best_cost = np.inf; best_alpha = alpha.copy()
        kw = self._kw(barrier_weight)

        cost0 = _cost_functional(_alpha_to_lambdas(alpha, lam0, lamN), **kw)
        if self.verbose:
            print(f"  [EP-flow] init={init_tag}  C(init)={cost0:.4e}")

        for it in range(1, self.max_iter + 1):
            grad = np.zeros_like(alpha)
            fd_h = self.fd_eps
            for i in range(N):
                ap = alpha.copy(); ap[i] += fd_h
                am = alpha.copy(); am[i] -= fd_h
                Cp = _cost_functional(_alpha_to_lambdas(ap, lam0, lamN), **kw)
                Cm = _cost_functional(_alpha_to_lambdas(am, lam0, lamN), **kw)
                grad[i] = (Cp - Cm) / (2.0 * fd_h)

            m  = self.beta1 * m + (1.0 - self.beta1) * grad
            v  = self.beta2 * v + (1.0 - self.beta2) * grad ** 2
            mh = m / (1.0 - self.beta1 ** it)
            vh = v / (1.0 - self.beta2 ** it)
            alpha -= lr * mh / (np.sqrt(vh) + 1e-8)
            lr   *= self.lr_decay

            lambdas = _alpha_to_lambdas(alpha, lam0, lamN)
            cost    = _cost_functional(lambdas, **kw)
            if cost < best_cost:
                best_cost = cost; best_alpha = alpha.copy()

            gnorm = float(np.linalg.norm(grad))
            if self.verbose and (it % max(1, self.max_iter // 10) == 0 or it == 1):
                h = np.diff(lambdas)
                print(f"  iter {it:4d}  cost={cost:.4e}  |g|={gnorm:.3e}  "
                      f"h_cv={h.std()/h.mean():.3f}  lr={lr:.4e}")
            if gnorm < self.tol:
                if self.verbose: print(f"  converged  iter={it}  |g|={gnorm:.3e}")
                break

        lams_best = _alpha_to_lambdas(best_alpha, lam0, lamN)
        self._lambdas_opt = lams_best
        return lams_best


# ─────────────────────────────────────────────────────────────────────────────
# λ → t
# ─────────────────────────────────────────────────────────────────────────────

def _sd15_alphas_cumprod(
    beta_start: float = 0.00085,
    beta_end:   float = 0.012,
    num_train_timesteps: int = 1000,
) -> np.ndarray:
    """SD1.5 scaled_linear beta schedule，consistent with diffusers
    DPMSolverSinglestepScheduler default settings"""
    betas = np.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
    return np.cumprod(1.0 - betas)


def _native_lambda_table(alphas_cumprod: np.ndarray) -> np.ndarray:
    ab = np.clip(alphas_cumprod, 1e-12, 1.0 - 1e-12)
    return 0.5 * np.log(ab / (1.0 - ab))


def lambdas_to_timesteps(lambdas, alphas_cumprod: Optional[np.ndarray] = None):
    if alphas_cumprod is None:
        alphas_cumprod = _sd15_alphas_cumprod()
    lam_t = _native_lambda_table(alphas_cumprod)
    t_idx = np.arange(len(lam_t), dtype=np.float64)
    t_cont = np.interp(np.asarray(lambdas, dtype=np.float64), lam_t[::-1], t_idx[::-1])
    ts = np.round(t_cont).astype(int)
    return np.clip(ts, 0, len(lam_t) - 1)


def logsnr_timesteps(lambda_min, lambda_max, N, alphas_cumprod: Optional[np.ndarray] = None):
    lambdas = np.linspace(lambda_min, lambda_max, N + 1)   # high to low noise
    return lambdas_to_timesteps(lambdas, alphas_cumprod)


# ─────────────────────────────────────────────────────────────────────────────
# AYS (Align Your Steps) SD1.5 schedule
# ─────────────────────────────────────────────────────────────────────────────

AYS_SIGMAS_SD15 = np.array(
    [14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.029],
    dtype=np.float64,
)


def _ays_loglinear_interp(sigmas: np.ndarray, num_nodes: int) -> np.ndarray:
    """official log-linear interpolation"""
    xs = np.linspace(0, 1, len(sigmas))
    ys = np.log(sigmas[::-1])
    new_xs = np.linspace(0, 1, num_nodes)
    new_ys = np.interp(new_xs, xs, ys)
    return np.exp(new_ys)[::-1].copy()

def loglinear_interp(t_steps, num_steps):
    """
    Performs log-linear interpolation of a given array of decreasing numbers.
    """
    xs = np.linspace(0, 1, len(t_steps))
    ys = np.log(t_steps[::-1])
    
    new_xs = np.linspace(0, 1, num_steps)
    new_ys = np.interp(new_xs, xs, ys)
    
    interped_ys = np.exp(new_ys)[::-1].copy()
    return interped_ys

def ays_timesteps(N: int, alphas_cumprod: Optional[np.ndarray] = None) -> np.ndarray:
    if alphas_cumprod is None:
        alphas_cumprod = _sd15_alphas_cumprod()

    if N + 1 == len(AYS_SIGMAS_SD15):
        sigmas = AYS_SIGMAS_SD15
    else:
        sigmas = _ays_loglinear_interp(AYS_SIGMAS_SD15, N + 1)

    lambdas = -np.log(sigmas)   # Karras sigma = exp(-lambda)
    return lambdas_to_timesteps(lambdas, alphas_cumprod)

#############################################################
# Xue et al. (CVPR'24) — verbatim from official repo, unmodified
#############################################################
def interpolate_fn(x, xp, yp):
    """
    A piecewise linear function y = f(x), using xp and yp as keypoints.
    ...
    """
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand

def xue_timesteps(N, alphas_cumprod=None, init_type='unif_t',
                   quant='native', return_both=False):
    if alphas_cumprod is None:
        alphas_cumprod = _sd15_alphas_cumprod()

    ac_t = torch.tensor(alphas_cumprod, dtype=torch.float64)
    ns  = NoiseScheduleVP(schedule='discrete', alphas_cumprod=ac_t,
                          dtype=torch.float64)
    opt = StepOptim(ns)

    total_N = len(alphas_cumprod)
    eps = 1.0 / total_N         
    t_res, lambda_res = opt.get_ts_lambdas(N=N, eps=eps, initType=init_type)

    lam_np = lambda_res.numpy().astype(np.float64)
    t_np   = t_res.numpy().astype(np.float64)   
    def _ours():
        return lambdas_to_timesteps(lam_np, alphas_cumprod)

    def _native():
        idx_cont = t_np * total_N - 1.0
        idx = np.round(idx_cont).astype(int)
        return np.clip(idx, 0, total_N - 1)

    if return_both:
        return {"ours": _ours(), "native": _native()}, lam_np, t_np

    ts = _ours() if quant == 'ours' else _native()
    return ts, lam_np

PROMPTS = {
    "basic": [
        "a photo of a cat",
        "a photo of a dog",
    ],

    "composition": [
        "a red cube on top of a blue sphere",
        "three apples on a wooden table with one sliced open",
    ],

    "counting": [
        "five yellow ducks in a row",
        "a group of 7 candles on a cake",
    ],

    "texture": [
        "a close-up of a knitted sweater with detailed texture",
        "a macro photo of tree bark with intricate patterns",
    ],

    "lighting": [
        "a portrait of a woman under dramatic cinematic lighting",
        "a city street at night with neon lights and reflections",
    ],

    "spatial": [
        "a chair behind a glass table",
        "a cat sitting under a chair next to a window",
    ],

    "long_prompt": [
        "a highly detailed cinematic photo of an astronaut riding a horse on mars, ultra realistic, 8k, dramatic lighting, volumetric fog",
    ],
}

_PIPE = {}
def get_pipe():
    if "p" not in _PIPE:
        p = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, variant="fp16"
        ).to("cuda")
        p.scheduler = _DefaultSingle.from_config(p.scheduler.config)
        p.set_progress_bar_config(disable=True)
        # probing
        p.scheduler.set_timesteps(timesteps=list(schedules["logsnr"]), device="cuda")
        print(f"  [probe] len(timesteps)={len(p.scheduler.timesteps)}  "
                f"order_list={getattr(p.scheduler, 'order_list', None)}")
        _PIPE["p"] = p
    return _PIPE["p"]
# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import shutil, argparse
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("--nfe",         type=int,   default=10)
    parser.add_argument("--n_prompts",   type=int,   default=800)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--hps_version", default="v2.1")
    parser.add_argument("--score_only",  action="store_true")
    parser.add_argument("--smooth_sigma_lam", type=float, default=0.4)
    parser.add_argument("--qual",       action="store_true", default=True)
    parser.add_argument("--qual_seeds", type=int, default=3)
    args = parser.parse_args()

    import hpsv2

    N        = args.nfe
    model_id = "runwayml/stable-diffusion-v1-5"
    BASE     = "/media/ssd_horse/keying/Pareto-Optimal-Scheduler/stats_sd_v1.5/runwayml--stable-diffusion-v1-5_cfg7.5_nonstat.npz"
    KAPPA    = "/media/ssd_horse/keying/Pareto-Optimal-Scheduler/stats_sd_v1.5/runwayml--stable-diffusion-v1-5_cfg7.5_nonstat_kappa.npz"
    OUT_ROOT = Path(f"hpsv2_sd15_N{N}_{datetime.now().strftime('%m%d_%H%M')}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[Output dir] {OUT_ROOT}")
  
    alphas_cumprod = _sd15_alphas_cumprod()

    # ── Build α-sweep + logSNR + AYS ─────────────────────────────────────
    sweep, _ = make_alpha_sweep_stats(
        BASE, KAPPA,
        alphas=(0.0, 0.5),
        smooth_sigma_lam=args.smooth_sigma_lam,
    )

    print("\n[Optimizing schedules]")
    schedules = {}
    ref_sched = None

    # AYS init
    ays_sigmas_N = AYS_SIGMAS_SD15 if N + 1 == len(AYS_SIGMAS_SD15) \
                   else _ays_loglinear_interp(AYS_SIGMAS_SD15, N + 1)
    ays_lam_init = -np.log(ays_sigmas_N)   

    for a, stats_override in sweep.items():
        N_a = 1000   
        for init_name, init_lam in [("uniform", None), ("ays", ays_lam_init)]:
            sched = OptimalSchedule(model_id, max_iter=N_a,
                                     stats_override=stats_override, verbose=False)
            lams  = sched._optimise(N=N, init_lambdas=init_lam)
            ts    = lambdas_to_timesteps(lams, alphas_cumprod)
            cost_final = sched._cost(lams, barrier_weight=0.0)
            name  = f"kairos_a{int(a*100):03d}_{init_name}"
            schedules[name] = ts
            if ref_sched is None:
                ref_sched = sched
            print(f"  {name}: cost={cost_final:.4e}  {ts}  "
                  f"h_cv={np.diff(lams).std()/np.diff(lams).mean():.3f}")
            
    logsnr_ts = logsnr_timesteps(ref_sched.lambda_min, ref_sched.lambda_max, N, alphas_cumprod)
    schedules["logsnr"] = logsnr_ts
    print(f"  logsnr:      {logsnr_ts}")

    ays_ts = ays_timesteps(N, alphas_cumprod)
    schedules["ays"] = ays_ts
    print(f"  ays:         {ays_ts}")

    xue_variants, xue_lams, xue_t = xue_timesteps(
        N, alphas_cumprod, init_type='unif_t', return_both=True
    )
    schedules["xue_ours_quant"]   = xue_variants["ours"]
    schedules["xue_native_quant"] = xue_variants["native"]

    print(f"  xue(ours_quant):   {xue_variants['ours']}")
    print(f"  xue(native_quant): {xue_variants['native']}")

    # ── Timestep table ────────────────────────────────────────────────────
    print("\n[Timesteps]  N+1 nodes")
    w = max(len(n) for n in schedules)
    for name, ts in schedules.items():
        ab  = np.clip(alphas_cumprod[ts], 1e-12, 1 - 1e-12)
        lam = 0.5 * np.log(ab / (1 - ab))
        sig = np.sqrt((1 - ab) / ab)          # Karras-style sigma = exp(-λ)
        print(f"  {name:<{w}}  t  = " + " ".join(f"{t:5d}"   for t in ts))
        print(f"  {'':<{w}}  λ  = " + " ".join(f"{l:5.2f}"  for l in lam))
        print(f"  {'':<{w}}  σ  = " + " ".join(f"{s:5.2f}"  for s in sig))
        print()

    import json
    with open(OUT_ROOT / "schedules.json", "w") as f:
        json.dump({k: v.tolist() for k, v in schedules.items()}, f, indent=2)

    # ── Qualitative grid ──────────────────────────────────────────────────
    if args.qual:
        pipe = get_pipe()
        qroot = OUT_ROOT / "qualitative"
        for cat, plist in PROMPTS.items():
            for pi, prompt in enumerate(plist):
                for s in range(args.qual_seeds):
                    seed = args.seed + 1000 * s
                    d = qroot / cat / f"{pi:02d}_seed{seed}"
                    d.mkdir(parents=True, exist_ok=True)
                    (d / "prompt.txt").write_text(prompt)
                    for sched_name, ts in schedules.items():
                        fp = d / f"{sched_name}.jpg"
                        if fp.exists():
                            continue
                        img = pipe(
                            prompt, timesteps=list(ts),
                            generator=torch.Generator(device="cuda").manual_seed(seed),
                        ).images[0]
                        img.save(str(fp), quality=95)
            print(f"  [qual] {cat} done")

    # ── Generation ────────────────────────────────────────────────────────
    if not args.score_only:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, variant="fp16"
        ).to("cuda")
        pipe.scheduler = _DefaultSingle.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        gen = torch.Generator(device="cuda").manual_seed(args.seed)

        all_prompts = hpsv2.benchmark_prompts("all")   # dict: style → list[str]

        for sched_name, ts in schedules.items():
            print(f"\n[Generating] {sched_name}  timesteps={ts}")
            for style, style_prompts in all_prompts.items():
                prompts_sub = style_prompts[: args.n_prompts]
                img_dir = OUT_ROOT / sched_name / style
                img_dir.mkdir(parents=True, exist_ok=True)

                n_existing = sum(1 for _ in img_dir.glob("*.jpg"))
                if n_existing >= len(prompts_sub):
                    print(f"  {style}: {n_existing} already done, skip")
                    continue

                for idx, prompt in enumerate(prompts_sub):
                    save_path = img_dir / f"{idx:05d}.jpg"
                    if save_path.exists():
                        continue
                    img = pipe(
                        prompt,
                        timesteps=ts,
                        generator=torch.Generator(device="cuda").manual_seed(args.seed + idx),
                    ).images[0]
                    img.save(str(save_path), quality=95)

                print(f"  {style}: {len(prompts_sub)} done")

    # ── Scoring ───────────────────────────────────────────────────────────
    print(f"\n[Scoring] HPS {args.hps_version}\n")
    for sched_name in schedules:
        img_path = OUT_ROOT / sched_name
        if not img_path.exists():
            print(f"  {sched_name}: not found, skip\n")
            continue
        print(f"=== {sched_name} ===")
        hpsv2.evaluate(str(img_path), hps_version=args.hps_version)
        print()

    print(f"[Done] {OUT_ROOT}")

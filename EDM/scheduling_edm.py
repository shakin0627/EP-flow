"""
EP-flow optimal timestep scheduler for EDM models.

Cost functional (EDM instance):

  C = w_rank1 · ρ_∞     · (Σ_j Γ_j · c_{j-1} · a_j)²      [rank-1 bias, raw D_θ error]
    + w_vres  · (1+2·C_ρ^η) · Σ_j Γ_j² · c_{j-1}² · V_j^res  [residual variance, raw D_θ error]
    + w_disc  · (1+2·C_ρ^d) · Σ_j Γ_j² · D_j                [discretisation variance, field-level]

  sigma_max / sigma_min are read from the stats .npz (grid endpoints).
"""

import warnings
from typing import Callable

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator, interp1d


# ══════════════════════════════════════════════════════════════════════════════
# φ / φ^res builders  — parametrization-agnostic
# ══════════════════════════════════════════════════════════════════════════════

def _build_phi_fn(
    rho_s: np.ndarray,
    rho_vals: np.ndarray,
    quad_pts: int = 64,
    s_extend_factor: float = 3.0,
) -> Callable[[float, float], float]:
    """φ(a,b) = ∫₀ᵃ ∫₀ᵇ ρ(|u−u'|) du' du  built from empirical (rho_s, rho_vals)."""
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
        slope = ((log_rho_fine[last] - log_rho_fine[max(last - 1, 0)])
                 / (s_fine[1] - s_fine[0] + 1e-30))
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
        a_int = min(a, b)
        b_int = max(a, b)
        u = np.linspace(0.0, a_int, quad_pts)
        return float(_trapz(R_interp(u) + R_interp(b_int - u), u))

    return phi_fn


def _build_phi_res_fn(
    rho_s: np.ndarray,
    rho_vals: np.ndarray,
    rho_infty: float,
    quad_pts: int = 64,
) -> Callable[[float, float], float]:
    """
    φ^res(a,b) = φ(a,b) − ρ_∞ · a · b

    Subtracts the rank-1 plateau analytically, preserving h² curvature:
        φ^res(h,h) ≈ (1−ρ_∞) h²  as h→0
    """
    phi_full = _build_phi_fn(rho_s, rho_vals, quad_pts)

    def phi_res_fn(a: float, b: float) -> float:
        return phi_full(a, b) - rho_infty * a * b

    return phi_res_fn


# ══════════════════════════════════════════════════════════════════════════════
# EDM coupling  c(σ) = -1/σ
# ══════════════════════════════════════════════════════════════════════════════

def _c_edm(sigma: float) -> float:
    """
    EDM coupling coefficient:
        f_θ − f* = c(σ)·η,   c(σ) = -1/σ
    Always negative for σ>0
    """
    return -1.0 / sigma


# ══════════════════════════════════════════════════════════════════════════════
# Euler propagator (forward Euler ONLY — see module-level warning)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_epsilon_edm(sigma_prev: float, sigma_curr: float,
                         g_fn: Callable) -> float:
    """
    ε_k = (σ_k − σ_{k-1}) · g(σ_{k-1})

    g(σ) = (1/σ)·(1 − (1/d)·Tr ∇_x D_θ)  
    """
    return (sigma_curr - sigma_prev) * g_fn(sigma_prev)


def _compute_gamma(eps: np.ndarray) -> np.ndarray:
    """Γ[j] = ∏_{k=j+1}^{N} (1 + ε[k]),   Γ[N-1] = 1.  Parametrization-agnostic."""
    N     = len(eps)
    Gamma = np.ones(N)
    for k in range(N - 2, -1, -1):
        Gamma[k] = Gamma[k + 1] * (1.0 + eps[k + 1])
    return Gamma


# ══════════════════════════════════════════════════════════════════════════════
# Per-step cost components
# ══════════════════════════════════════════════════════════════════════════════

def _compute_a_j(sigma_prev: float, sigma_curr: float,
                 sigma2_eta_fn: Callable, n_quad: int = 32) -> float:
    """
    a_j = ∫_{σ_j}^{σ_{j-1}} σ_η(τ) dτ

    σ_η(σ) = RMS‖D_θ(x,σ) − D*(x,σ)‖ 
    """
    tau     = np.linspace(sigma_curr, sigma_prev, n_quad)
    sig_eta = np.sqrt(np.maximum([sigma2_eta_fn(float(t)) for t in tau], 0.0))
    return float((getattr(np, "trapezoid", None) or np.trapz)(sig_eta, tau))


def _compute_V_res_edm(sigma_prev: float, sigma_curr: float,
                       sigma2_eta_fn: Callable,
                       phi_res_fn: Callable) -> float:
    """
    V_j^res = σ²_η(σ̄_j) · φ^res(h_j, h_j)     [raw D_θ-error kernel]
    """
    h_abs   = sigma_prev - sigma_curr
    sig_bar = 0.5 * (sigma_prev + sigma_curr)
    s2      = max(sigma2_eta_fn(sig_bar), 0.0)
    Q       = phi_res_fn(h_abs, h_abs)
    return s2 * max(Q, 0.0)


def _compute_D_j(sigma_prev: float, sigma_curr: float,
                 sigma2_fdot_fn: Callable) -> float:
    """
    D_j = (h_j⁴ / 4) · σ²_{ḟ}(σ_{j-1})
    """
    h_abs = sigma_prev - sigma_curr
    return (h_abs ** 4 / 4.0) * max(sigma2_fdot_fn(sigma_prev), 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# Full cost functional
# ══════════════════════════════════════════════════════════════════════════════

def _cost_functional_edm(
    sigmas: np.ndarray,
    g_fn: Callable,
    sigma2_eta_fn: Callable,
    sigma2_fdot_fn: Callable,
    phi_res_fn: Callable,
    rho_infty: float,
    C_rho_eta: float,
    C_rho_d: float,
    n_quad: int = 32,
    w_rank1: float = 1.0,
    w_vres:  float = 1.0,
    w_disc:  float = 1.0,
    c_clip: float = 1e6,
) -> float:
    """
    C = w_rank1 · ρ_∞          · (Σ_j Γ_j · c_{j-1} · a_j)²
      + w_vres  · (1+2·C_ρ^η) · Σ_j Γ_j² · c_{j-1}² · V_j^res
      + w_disc  · (1+2·C_ρ^d) · Σ_j Γ_j² · D_j
    """
    N = len(sigmas) - 1

    eps   = np.array([_compute_epsilon_edm(sigmas[k - 1], sigmas[k], g_fn)
                      for k in range(1, N + 1)])
    Gamma = _compute_gamma(eps)

    c_arr = np.clip(np.array([_c_edm(sigmas[k - 1]) for k in range(1, N + 1)]),
                    -c_clip, c_clip)

    a_arr = np.array([_compute_a_j(sigmas[k - 1], sigmas[k], sigma2_eta_fn, n_quad)
                      for k in range(1, N + 1)])
    V_arr = np.array([_compute_V_res_edm(sigmas[k - 1], sigmas[k], sigma2_eta_fn, phi_res_fn)
                      for k in range(1, N + 1)])
    D_arr = np.array([_compute_D_j(sigmas[k - 1], sigmas[k], sigma2_fdot_fn)
                      for k in range(1, N + 1)])

    rank1_term = w_rank1 * rho_infty * float(np.dot(Gamma * c_arr, a_arr)) ** 2
    vres_term  = w_vres  * (1.0 + 2.0*C_rho_eta) * float(np.dot((Gamma * c_arr) ** 2, V_arr))
    disc_term  = w_disc  * (1.0 + 2.0*C_rho_d)   * float(np.dot(Gamma ** 2, D_arr))

    return rank1_term + vres_term + disc_term


# ══════════════════════════════════════════════════════════════════════════════
# Softmax reparameterisation — parametrization-agnostic
# ══════════════════════════════════════════════════════════════════════════════

def _alpha_to_sigmas(alpha: np.ndarray,
                     sigma_max: float,
                     sigma_min: float) -> np.ndarray:
    """
    h_i = softmax(alpha)_i · (sigma_max − sigma_min)   [all h_i > 0, Σh_i = span]
    sigmas = [sigma_max, sigma_max − h_1, …, sigma_min]
    """
    span = sigma_max - sigma_min
    a    = alpha - alpha.max()
    h    = np.exp(a) / np.exp(a).sum() * span
    interior = sigma_max - np.cumsum(h)[:-1]
    return np.concatenate([[sigma_max], interior, [sigma_min]])


def _karras_alpha_init(nfe: int, sigma_max: float, sigma_min: float,
                       rho: float = 7.0) -> np.ndarray:
    """
    Warm-start alpha at the Karras ρ-schedule
    """
    i = np.arange(nfe + 1, dtype=np.float64)
    sig = (sigma_max ** (1/rho) + i/nfe * (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
    h = -np.diff(sig)               # positive step sizes, length N
    h = np.maximum(h, 1e-8)
    alpha = np.log(h)
    alpha -= alpha.mean()
    return alpha

def _grad_alpha(
    alpha: np.ndarray,
    sigma_max: float,
    sigma_min: float,
    fd_h: float,
    **cost_kw,
) -> np.ndarray:
    """Central-difference FD gradient w.r.t. alpha."""
    grad = np.zeros_like(alpha)
    for i in range(len(alpha)):
        ap = alpha.copy(); ap[i] += fd_h
        am = alpha.copy(); am[i] -= fd_h
        cp = _cost_functional_edm(_alpha_to_sigmas(ap, sigma_max, sigma_min), **cost_kw)
        cm = _cost_functional_edm(_alpha_to_sigmas(am, sigma_max, sigma_min), **cost_kw)
        grad[i] = (cp - cm) / (2.0 * fd_h)
    return grad


# ══════════════════════════════════════════════════════════════════════════════
# Stats loader
# ══════════════════════════════════════════════════════════════════════════════

def _extract_rho_infty(rho_s: np.ndarray, rho_vals: np.ndarray) -> float:
    drho     = np.abs(np.diff(rho_vals))
    rel_drho = drho / (np.abs(rho_vals[:-1]) + 1e-8)
    MIN_FLAT = max(5, len(rho_vals) // 10)

    plateau_start = None
    for i in range(len(rel_drho) - MIN_FLAT + 1):
        if np.all(rel_drho[i: i + MIN_FLAT] < 0.02):
            plateau_start = i
            break
    if plateau_start is None:
        warnings.warn("[edm_schedule] No plateau found — extend rho_s range.", UserWarning)
        plateau_start = len(rho_vals) // 2

    rho_at_start = rho_vals[plateau_start]
    plateau_end  = len(rho_vals)
    for i in range(plateau_start + MIN_FLAT, len(rho_vals)):
        if rho_vals[i] < rho_at_start * 0.95:
            plateau_end = i
            break

    plateau_vals = rho_vals[plateau_start:plateau_end]
    ri = float(np.clip(np.median(plateau_vals), 0.0, 1.0 - 1e-6))

    cv = np.std(plateau_vals) / (np.mean(plateau_vals) + 1e-8)
    if cv > 0.05:
        warnings.warn(
            f"[edm_schedule] Plateau CV={cv:.3f} — ρ_∞ estimate unreliable. "
            f"Detected plateau: indices [{plateau_start}, {plateau_end}), "
            f"s=[{rho_s[plateau_start]:.3f}, {rho_s[min(plateau_end, len(rho_s)-1)]:.3f}].",
            UserWarning,
        )
    return ri


def _load_stats_edm(npz_path: str):
    """
    Load offline EDM stats .npz (g(λ)=(1/d)Tr[df/dx],
    σ_η(σ)=RMS‖D_θ−x0‖, D_j=E‖ḟ_θ‖², η-correlation, field-accel correlation).

    Returns:
        sigma2_eta_fn, sigma2_fdot_fn, g_fn   — interpolating callables
        phi_res_fn                             — residual kernel φ^res (on η)
        rho_infty                              — η rank-1 plateau height
        sigma_max, sigma_min                   — grid endpoints (from stats)
        C_rho_eta                              — ∫ ρ_η^res ds
        C_rho_d                                — ∫ ρ_ḟ    ds  (field-accel corr.)
    """
    data = np.load(npz_path)

    sigma_grid = data.get("sigma_grid", data.get("t_grid")).astype(np.float64)
    s2_eta = data.get("sigma2_eta", data.get("sigma_eta_sq")).astype(np.float64)
    # D_j panel in your plot is E||f_dot||^2 vs sigma — this is sigma2_fdot,
    # the FIELD-level curvature, not a raw-output quantity. Do NOT confuse
    # with sigma2_eta above.
    s2_fd  = data.get("sigma2_fdot",
                  data.get("D_j_values",
                  data.get("D_j"))).astype(np.float64)

    g_arr  = data.get("g_values").astype(np.float64)

    # scipy PchipInterpolator requires increasing x
    if sigma_grid[0] > sigma_grid[-1]:
        sigma_grid = sigma_grid[::-1]
        s2_eta = s2_eta[::-1]
        s2_fd = s2_fd[::-1]
        g_arr = g_arr[::-1]
    rho_s  = data.get("rho_s").astype(np.float64)
    rho_v  = data.get("rho_values").astype(np.float64)

    sigma_max = float(data.get("sigma_max"))
    sigma_min = float(data.get("sigma_min"))

    _s2eta = PchipInterpolator(sigma_grid, s2_eta, extrapolate=True)
    _s2fd  = PchipInterpolator(sigma_grid, s2_fd,  extrapolate=True)
    _g     = PchipInterpolator(sigma_grid, g_arr,  extrapolate=True)

    def sigma2_eta_fn(s): return max(float(_s2eta(s)), 1e-10)
    def sigma2_fdot_fn(s): return max(float(_s2fd(s)), 1e-10)
    def g_fn(s):          return float(_g(s))

    rho_infty = _extract_rho_infty(rho_s, rho_v)
    phi_res   = _build_phi_res_fn(rho_s, rho_v, rho_infty, quad_pts=64)

    rho_res_vals  = np.clip(rho_v - rho_infty, 0.0, None)
    s_anchored    = np.concatenate([[0.0], rho_s])
    rres_anchored = np.concatenate([[max(1.0 - rho_infty, rho_res_vals[0])],
                                    rho_res_vals])
    C_rho_eta = float((getattr(np, "trapezoid", None) or np.trapz)(
                      rres_anchored, s_anchored))
  
    if "rho_s_curv" in data.files and "rho_values_curv" in data.files:
        rho_s_fd = data["rho_s_curv"].astype(np.float64)
        rho_fd   = data["rho_values_curv"].astype(np.float64)
        C_rho_d  = float((getattr(np, "trapezoid", None) or np.trapz)(
                         np.clip(rho_fd, 0.0, None), rho_s_fd))
    else:
        warnings.warn("[edm_schedule] rho_s_fdot not in npz — setting C_ρ^d = 0.",
                      UserWarning)
        C_rho_d = 0.0

    return (sigma2_eta_fn, sigma2_fdot_fn, g_fn,
            phi_res, rho_infty,
            sigma_max, sigma_min,
            C_rho_eta, C_rho_d)


# ══════════════════════════════════════════════════════════════════════════════
# Main optimizer
# ══════════════════════════════════════════════════════════════════════════════

def optimize_schedule_edm(
    npz_path: str,
    nfe: int,
    # sigma_max / sigma_min are read from the npz (grid endpoints).
    sigma_max_override: float = None,
    sigma_min_override: float = None,
    init: str              = "karras",   # "karras"
    n_steps: int           = 2000,
    lr: float              = 1e-3,
    lr_decay: float        = 0.995,
    fd_h: float            = 1e-3,
    n_quad: int            = 32,
    w_rank1: float         = 1.0,
    w_vres:  float         = 1.0,
    w_disc:  float         = 1.0,
    c_clip: float          = 1e6,
    n_restarts: int        = 3,
    verbose: bool          = True,
) -> np.ndarray:
    """
    Returns optimal sigmas array (length nfe+1, strictly decreasing).
    Restart 0: Karras-schedule warm-start .
    Restart 1+: warm-start + Gaussian perturbation.
    """
    (sigma2_eta_fn, sigma2_fdot_fn, g_fn,
     phi_res, rho_infty,
     sigma_max, sigma_min,
     C_rho_eta, C_rho_d) = _load_stats_edm(npz_path)

    if sigma_max_override is not None:
        sigma_max = sigma_max_override
    if sigma_min_override is not None:
        sigma_min = sigma_min_override

    if verbose:
        print("═" * 64)
        print(" EDM schedule optimizer — FORWARD EULER ONLY")
        print("═" * 64)
        print(f"  σ ∈ [{sigma_min:.4f}, {sigma_max:.4f}]  NFE={nfe}")
        print(f"  ρ_∞(η)  = {rho_infty:.4f}")
        print(f"  C_ρ^η   = {C_rho_eta:.4f}  →  (1+2C_ρ^η) = {1+2*C_rho_eta:.4f}")
        print(f"  C_ρ^d   = {C_rho_d:.4f}   →  (1+2C_ρ^d) = {1+2*C_rho_d:.4f}")
        print(f"  [disc rank-1 dropped: ρ_∞^d ≈ 0]")
        c_at_min = abs(_c_edm(sigma_min))
        print(f"  |c(σ_min)| = 1/σ_min = {c_at_min:.2f}  "
              f"(rank-1/vres terms up-weighted at low noise — c_clip={c_clip:g})")

    cost_kw = dict(
        g_fn=g_fn,
        sigma2_eta_fn=sigma2_eta_fn,
        sigma2_fdot_fn=sigma2_fdot_fn,
        phi_res_fn=phi_res,
        rho_infty=rho_infty,
        C_rho_eta=C_rho_eta,
        C_rho_d=C_rho_d,
        n_quad=n_quad,
        w_rank1=w_rank1,
        w_vres=w_vres,
        w_disc=w_disc,
        c_clip=c_clip,
    )

    best_cost   = float("inf")
    best_sigmas = None

    for restart in range(n_restarts):
        if init == "karras":
            alpha = _karras_alpha_init(nfe, sigma_max, sigma_min)
        if restart > 0:
            alpha = alpha + np.random.randn(nfe) * 0.5

        m  = np.zeros_like(alpha)
        v  = np.zeros_like(alpha)
        β1, β2, ε_a = 0.9, 0.999, 1e-8
        lr_t = lr
        plateau_count = 0
        cost_prev = float("inf")

        for step in range(1, n_steps + 1):
            g_vec = _grad_alpha(alpha, sigma_max, sigma_min, fd_h, **cost_kw)
            m     = β1 * m + (1 - β1) * g_vec
            v     = β2 * v + (1 - β2) * g_vec ** 2
            alpha -= lr_t * (m / (1 - β1 ** step)) / (np.sqrt(v / (1 - β2 ** step)) + ε_a)
            lr_t  *= lr_decay

            sigmas = _alpha_to_sigmas(alpha, sigma_max, sigma_min)
            cost   = _cost_functional_edm(sigmas, **cost_kw)

            if verbose and step % max(1, n_steps // 10) == 0:
                h = sigmas[:-1] - sigmas[1:]
                print(f"  [r{restart}] step {step:4d}  cost={cost:.4e}  "
                      f"h∈[{h.min():.4f},{h.max():.4f}]  "
                      f"h_cv={h.std()/h.mean():.3f}")

            rel = abs(cost_prev - cost) / (abs(cost_prev) + 1e-12)
            plateau_count = (plateau_count + 1) if rel < 1e-7 else 0
            if plateau_count > 50:
                if verbose:
                    print(f"  [r{restart}] early stop @ step {step}")
                break
            cost_prev = cost

        sigmas = _alpha_to_sigmas(alpha, sigma_max, sigma_min)
        c      = _cost_functional_edm(sigmas, **cost_kw)
        if c < best_cost:
            best_cost   = c
            best_sigmas = sigmas.copy()
            if verbose:
                print(f"  [r{restart}] ★ best cost={best_cost:.4e}")

    if verbose:
        h = best_sigmas[:-1] - best_sigmas[1:]
        print(f"\n  Final schedule (NFE={nfe}):")
        print(f"  sigmas = {np.round(best_sigmas, 4).tolist()}")
        print(f"  h      = {np.round(h, 4).tolist()}")
        print(f"  h_cv   = {h.std()/h.mean():.3f}")
        print("\n  sample this with ablation_sampler(solver='euler', "
              "discretization=<custom>, sigma_steps=<this array>).")

    return best_sigmas


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, help="EDM statistics npz")
    parser.add_argument("--nfe", type=int, required=True)
    parser.add_argument("--out", default=None, help="path to save sigmas .npy")
    parser.add_argument("--init", choices=["karras", "cosine"], default="karras")
    args = parser.parse_args()
  
    sigmas = optimize_schedule_edm(args.npz, args.nfe, init=args.init)

    if args.out:
        np.save(args.out, sigmas)
        print(f"\nSaved σ-schedule to {args.out}")
       

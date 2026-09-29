"""
EP-Flow statistics collection — EDM instance 
forward-Euler sampler on the probability-flow ODE dx/dsigma = (x - D_theta)/sigma).

=============================================================================
RELATION TO THE UNIFIED FRAMEWORK (edm_unified.md)
=============================================================================
The generic first-order sampler integrates  dx/ds = f_theta(x, s).
For EDM:
    s              = sigma         (noise level, sigma_max -> sigma_min)
    f_theta(x,s)   = (x - D_theta(x, sigma)) / sigma
    native output  = D_theta       (the network's x0 prediction)
    eta            = D_theta - D*  (native output error)
    coupling c(s)  = -1/sigma      (applied in the cost)

Statistics in sigma-space:
    g(sigma)          scalar Jacobian proxy  (1/d) tr[ d f_theta / dx ]
                      = (1/sigma) * ( 1 - (1/d) tr[ dD_theta/dx ] )
    sigma_eta(sigma)  RMS native output error  sqrt(E|| D_theta - D* ||^2)
    D_j / dbar_j      field-acceleration magnitude  E|| f_theta_dot ||^2,
                      where f_theta_dot = d/dsigma [ (x*_sigma - D_theta)/sigma ]
    rho(|delta|)      correlation kernel of the native output error eta
    rho_curv(|delta|) correlation kernel of the field acceleration f_theta_dot

=============================================================================
NETWORK CALL CONVENTION (NVlabs edm)
=============================================================================
    net = pickle.load(open(pkl,'rb'))['ema']             # EDMPrecond-wrapped model
    D = net(x, sigma, class_labels)                      # returns denoised x0 pred
    net.sigma_min, net.sigma_max, net.round_sigma(sigma) # available
    net.img_channels, net.img_resolution, net.label_dim  # shape / cond info
=============================================================================
"""

import argparse
import warnings
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# sigma / lambda grid  (EDM: schedule in sigma-space, densify low-noise end)
# ─────────────────────────────────────────────────────────────────────────────

def build_sigma_grid(sigma_min, sigma_max, n_sigma, rho=7.0,
                     dense_frac_low_noise=1.0 / 3, dense_split_frac=0.80):
    """
    Return sigmas decreasing sigma_max -> sigma_min. 
    """
    lam_min = -np.log(sigma_max)   # low lambda  <-> high noise
    lam_max = -np.log(sigma_min)   # high lambda <-> low noise
    n_dense = max(1, int(n_sigma * dense_frac_low_noise))
    n_coarse = n_sigma - n_dense
    lam_split = lam_min + dense_split_frac * (lam_max - lam_min)
    lam_coarse = np.linspace(lam_min, lam_split, n_coarse, endpoint=False)
    lam_dense = np.linspace(lam_split, lam_max, n_dense)
    lam_grid = np.concatenate([lam_coarse, lam_dense])
    sigma_grid = np.exp(-lam_grid)
    return sigma_grid.astype(np.float64), lam_grid.astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Network loading
# ─────────────────────────────────────────────────────────────────────────────

def load_edm_net(pkl_path_or_url, device):
    """
    Load an NVlabs EDM/EDM2 pickle. Requires the `dnnlib` + `torch_utils`
    packages from the edm repo to be importable (clone NVlabs/edm and run from
    its root, or add it to PYTHONPATH). We open via dnnlib.util.open_url so both
    local paths and https URLs work, exactly like generate.py.
    """
    try:
        import dnnlib
    except ImportError as e:
        raise ImportError(
            "Could not import `dnnlib`. Clone https://github.com/NVlabs/edm and run this "
            "script from its root (or add it to PYTHONPATH) so `dnnlib` and `torch_utils` "
            "are importable — the pickled network depends on those modules at unpickle time."
        ) from e

    with dnnlib.util.open_url(pkl_path_or_url) as f:
        net = pickle.load(f)["ema"].to(device)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def call_D(net, x, sigma, class_labels, force_fp32=True):
    """
    Wrapper around net(x, sigma, class_labels) -> D_theta (x0 prediction).
    """
    if not torch.is_tensor(sigma):
        sigma = torch.as_tensor(sigma, device=x.device, dtype=x.dtype)
    if sigma.ndim == 0:
        sigma = sigma.repeat(x.shape[0])
    sigma = net.round_sigma(sigma)
    D = net(x, sigma, class_labels)
    return D.to(torch.float32) if force_fp32 else D


def f_field(net, x, sigma, class_labels):
    """EDM probability-flow field f_theta = (x - D_theta)/sigma."""
    D = call_D(net, x, sigma, class_labels)
    return (x - D) / sigma


# ─────────────────────────────────────────────────────────────────────────────
# g(sigma) — scalar Jacobian proxy of the FIELD f_theta (Hutchinson trace)
#   g(sigma) = (1/d) tr[ d f_theta / dx ]
#            = (1/sigma) ( 1 - (1/d) tr[ dD_theta/dx ] )
# ─────────────────────────────────────────────────────────────────────────────

def estimate_g_at_sigma(net, sigma, d, x0_batch, class_labels, n_probes, device):
    g_acc, count = 0.0, 0
    for bi in range(x0_batch.shape[0]):
        xi = x0_batch[bi:bi + 1]
        cl = None if class_labels is None else class_labels[bi:bi + 1]
        for _ in range(n_probes):
            eps = torch.randn_like(xi)
            x_sig = xi + sigma * eps
            v = torch.randint(0, 2, x_sig.shape, device=device, dtype=x_sig.dtype) * 2 - 1
            x_req = x_sig.detach().requires_grad_(True)
            field = f_field(net, x_req, sigma, cl)          # (x - D)/sigma
            Jv = torch.autograd.grad((field * v).sum(), x_req,
                                     create_graph=False, retain_graph=False)[0]
            g_acc += (Jv * v).sum().item()
            count += 1
            del x_req, field, Jv
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return g_acc / (count * d)

# ─────────────────────────────────────────────────────────────────────────────
# sigma_eta(sigma) — RMS native output error  sqrt(E|| D_theta - x0 ||^2)
#   (surrogate D* = x0. Raw D-space, NO 1/sigma here.)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_sigma2_eta_at_sigma(net, sigma, d, x0_batch, class_labels, device):
    total_sq, total_n = 0.0, 0
    B = x0_batch.shape[0]
    eps = torch.randn_like(x0_batch)
    x_sig = x0_batch + sigma * eps
    D = call_D(net, x_sig, sigma, class_labels)
    total_sq += F.mse_loss(D, x0_batch, reduction="sum").item()  # || D_theta - x0 ||^2 summed
    total_n += B * d
    return total_sq / total_n                                    # = (1/d) E|| eta ||^2  (per-dim mean-square)


# ─────────────────────────────────────────────────────────────────────────────
# D_j — field acceleration magnitude  E|| f_theta_dot ||^2
#   f_theta_dot = d/dsigma [ (x*_sigma - D_theta(x*_sigma, sigma)) / sigma ]
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_Dj_at_index(net, k, sigma_grid, d, x0_batch, class_labels, device, delta_idx=2):
    k_lo = max(0, k - delta_idx)
    k_hi = min(len(sigma_grid) - 1, k + delta_idx)
    if k_lo == k_hi:
        return 0.0
    s_lo, s_mid, s_hi = float(sigma_grid[k_lo]), float(sigma_grid[k]), float(sigma_grid[k_hi])
    h = (s_hi - s_lo) / 2.0
    if abs(h) < 1e-12:
        return 0.0

    eps = torch.randn_like(x0_batch)               # path-coherent
    x_lo  = x0_batch + s_lo  * eps
    x_mid = x0_batch + s_mid * eps
    x_hi  = x0_batch + s_hi  * eps

    f_lo  = (x_lo  - call_D(net, x_lo,  s_lo,  class_labels)) / s_lo
    f_mid = (x_mid - call_D(net, x_mid, s_mid, class_labels)) / s_mid
    f_hi  = (x_hi  - call_D(net, x_hi,  s_hi,  class_labels)) / s_hi

    # central first derivative of the field w.r.t. sigma (dbar_j uses ||f_dot||)
    f_dot = (f_hi - f_lo) / (2.0 * h)
    return (f_dot ** 2).sum().item() / (x0_batch.shape[0] * d)


# ─────────────────────────────────────────────────────────────────────────────
# eta correlation kernel  rho(|delta_lambda|)  (independent eps per sigma)
#   eta = D_theta - x0   (native output error)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_eta_correlation(net, sigma_grid, lam_grid, d, x0_batch, class_labels, device):
    M = len(sigma_grid)
    B = x0_batch.shape[0]
    C = np.zeros((M, M), dtype=np.float64)
    for bi in range(B):
        xi = x0_batch[bi:bi + 1]
        cl = None if class_labels is None else class_labels[bi:bi + 1]
        errs = []
        for s in sigma_grid:
            s = float(s)
            eps = torch.randn_like(xi)             # fresh independent noise per sigma
            x_sig = xi + s * eps
            D = call_D(net, x_sig, s, cl)
            errs.append((D - xi).flatten().cpu().double())
        for i in range(M):
            C[i, i] += (errs[i] * errs[i]).sum().item() / d
            for j in range(i + 1, M):
                dot = (errs[i] * errs[j]).sum().item() / d
                C[i, j] += dot
                C[j, i] += dot
    C /= B
    diag = np.diag(C).clip(min=1e-30)
    rho = np.clip(C / np.sqrt(np.outer(diag, diag)), 1e-8, 1.0)
    return C, rho


# ─────────────────────────────────────────────────────────────────────────────
# field-acceleration correlation kernel  rho_curv(|delta_lambda|)
#   (path-coherent: same eps across the whole sigma grid)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_curv_correlation(net, sigma_grid, d, x0_batch, class_labels, device, delta_idx=2):
    M = len(sigma_grid)
    B = x0_batch.shape[0]
    C = np.zeros((M, M), dtype=np.float64)
    for bi in range(B):
        xi = x0_batch[bi:bi + 1]
        cl = None if class_labels is None else class_labels[bi:bi + 1]
        eps = torch.randn_like(xi)                 # shared across sigma -> coherent
        fdot_vecs = [None] * M
        for k in range(M):
            k_lo = max(0, k - delta_idx)
            k_hi = min(M - 1, k + delta_idx)
            if k_lo == k_hi:
                continue
            s_lo, s_hi = float(sigma_grid[k_lo]), float(sigma_grid[k_hi])
            h = (s_hi - s_lo) / 2.0
            if abs(h) < 1e-12:
                continue
            x_lo = xi + s_lo * eps
            x_hi = xi + s_hi * eps
            f_lo = (x_lo - call_D(net, x_lo, s_lo, cl)) / s_lo
            f_hi = (x_hi - call_D(net, x_hi, s_hi, cl)) / s_hi
            fdot = (f_hi - f_lo) / (2.0 * h)
            fdot_vecs[k] = fdot.flatten().cpu().double()
        for i in range(M):
            if fdot_vecs[i] is None:
                continue
            C[i, i] += (fdot_vecs[i] * fdot_vecs[i]).sum().item() / d
            for j in range(i + 1, M):
                if fdot_vecs[j] is None:
                    continue
                dot = (fdot_vecs[i] * fdot_vecs[j]).sum().item() / d
                C[i, j] += dot
                C[j, i] += dot
    C /= B
    diag = np.diag(C).clip(min=1e-30)
    rho = np.clip(C / np.sqrt(np.outer(diag, diag)), -1.0, 1.0)
    return C, rho


# ─────────────────────────────────────────────────────────────────────────────
# rho-curve extraction + plateau analysis 
# ─────────────────────────────────────────────────────────────────────────────

def _extract_rho_curve(dl, rho, n_bins=40, anchor_at_one=True):
    valid = dl > 0
    if valid.sum() < 5:
        s = np.linspace(0, 1, 5)
        return s, np.ones(5)
    edges = np.unique(np.percentile(dl[valid], np.linspace(0, 100, n_bins + 1)))
    s_list, v_list = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = valid & (dl >= lo) & (dl < hi)
        if mask.sum() < 3:
            continue
        s_list.append(float(np.median(dl[mask])))
        v_list.append(float(np.median(rho[mask])))
    if not s_list:
        return np.array([0.0, dl.max()]), np.array([1.0, 0.5])
    s_arr, v_arr = np.array(s_list), np.array(v_list)
    if anchor_at_one:
        s_arr = np.concatenate([[0.0], s_arr]); v_arr = np.concatenate([[1.0], v_arr])
    else:
        s_arr = np.concatenate([[0.0], s_arr]); v_arr = np.concatenate([[v_arr[0]], v_arr])
    log_v = np.log(np.clip(v_arr, 1e-10, 1.0))
    for k in range(1, len(log_v)):
        if log_v[k] > log_v[k - 1]:
            log_v[k] = log_v[k - 1]
    v_arr = np.exp(log_v)
    pchip = PchipInterpolator(s_arr, v_arr, extrapolate=False)
    s_out = np.linspace(s_arr[0], s_arr[-1], 200)
    v_out = np.clip(pchip(s_out), v_arr[-1], v_arr[0])
    return s_out, v_out


def analyse_plateau(rho_s, rho_vals, label=""):
    drho = np.abs(np.diff(rho_vals))
    rel = drho / np.maximum(np.abs(rho_vals[:-1]), 0.05)
    MIN_FLAT = max(5, len(rho_vals) // 10)
    start = None
    for i in range(len(rel) - MIN_FLAT + 1):
        if np.all(rel[i:i + MIN_FLAT] < 0.02):
            start = i; break
    if start is None:
        warnings.warn(f"[{label}] no plateau; extend sigma range")
        start = len(rho_vals) // 2
    end = len(rho_vals)
    for i in range(start + MIN_FLAT, len(rho_vals)):
        if rho_vals[i] < rho_vals[start] * 0.95:
            end = i; break
    plateau = rho_vals[start:end]
    rho_infty = float(np.clip(np.median(plateau), 0.0, 1.0 - 1e-6))
    frac_r1 = rho_infty / max(float(rho_vals[0]), 1e-8)
    res = np.maximum(rho_vals - rho_infty, 0.0)
    norm = res / max(res[0], 1e-8)
    cross = np.where(norm <= 1.0 / np.e)[0]
    ell_res = float(rho_s[cross[0]]) if len(cross) else float(rho_s[-1])
    print(f"  [{label}] rho_inf={rho_infty:.4f}  rank1_frac={frac_r1:.3f}  ell_res={ell_res:.4f}"
          f"  -> {'STRONG' if frac_r1 > 0.3 else 'WEAK'} rank-1")
    return dict(rho_infty=rho_infty, frac_rank1=frac_r1, ell_res=ell_res)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[edm-stats] device={device}  network={args.network}")

    net = load_edm_net(args.network, device)
    sigma_min = max(float(args.sigma_min), float(net.sigma_min))
    sigma_max = min(float(args.sigma_max), float(net.sigma_max))
    C_img, H = int(net.img_channels), int(net.img_resolution)
    d = C_img * H * H
    label_dim = int(net.label_dim)
    print(f"  img: {C_img}x{H}x{H}  d={d:,}  label_dim={label_dim}  "
          f"sigma in [{sigma_min:.4g}, {sigma_max:.4g}]")

    sigma_grid, lam_grid = build_sigma_grid(sigma_min, sigma_max, args.n_sigma, rho=args.rho)
    n = len(sigma_grid)

    safe = args.network.split("/")[-1].replace(".pkl", "")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    partial_path = out_dir / f"{safe}.partial.npz"
    _store = dict(
        sigma_grid=sigma_grid.astype(np.float32),
        lambda_grid=lam_grid.astype(np.float32),
        sigma_min=np.float32(sigma_min), sigma_max=np.float32(sigma_max),
    )

    def _ckpt(**kv):
        _store.update({k: (v.astype(np.float32) if isinstance(v, np.ndarray) else v)
                       for k, v in kv.items()})
        np.savez(str(partial_path), **_store)
        print(f"  [ckpt] -> {partial_path}  ({', '.join(kv.keys())})")


    # ── load real data samples x0 (NOT gaussian) ──
    print(f"[edm-stats] loading data samples from {args.data_npz}")
    arr = np.load(args.data_npz)
    if hasattr(arr, "files"):            # npz
        key = arr.files[0]
        arr = arr[key]
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[-1] in (1, 3) and arr.shape[1] not in (1, 3):
        arr = np.transpose(arr, (0, 3, 1, 2))   # NHWC -> NCHW
    x0_all = torch.from_numpy(arr[: args.n_samples]).float().to(device)
    # normalize to [-1, 1] if it looks like [0, 255] or [0, 1]
    if x0_all.max() > 1.5:
        x0_all = x0_all / 127.5 - 1.0
    elif x0_all.min() >= 0.0:
        x0_all = x0_all * 2.0 - 1.0
    print(f"  x0 samples: {tuple(x0_all.shape)}  range [{x0_all.min():.3f},{x0_all.max():.3f}]")

    # class labels: for conditional nets, sample random one-hot; for uncond, None
    if label_dim > 0:
        idx = torch.randint(0, label_dim, (x0_all.shape[0],), device=device)
        class_labels = torch.eye(label_dim, device=device)[idx]
        print(f"  using random class conditioning (label_dim={label_dim})")
    else:
        class_labels = None

    # ── g(sigma) ──
    print(f"[edm-stats] g(sigma) at {n} points ...")
    g_vals = np.zeros(n)
    for k, s in enumerate(sigma_grid):
        g_acc, nb = 0.0, 0
        for st in range(0, x0_all.shape[0], args.batch):
            xb = x0_all[st:st + args.batch]
            cb = None if class_labels is None else class_labels[st:st + args.batch]
            g_acc += estimate_g_at_sigma(net, float(s), d, xb, cb, args.n_hutchinson, device)
            nb += 1
        g_vals[k] = g_acc / nb
        if (k + 1) % max(1, n // 10) == 0 or k == 0:
            print(f"  g {k+1}/{n}  sigma={s:.4g}  g={g_vals[k]:.4f}")

    # ── sigma_eta(sigma) ──
    print(f"[edm-stats] sigma_eta(sigma) at {n} points ...")
    sig2_eta = np.zeros(n)
    for k, s in enumerate(sigma_grid):
        acc, nb = 0.0, 0
        for st in range(0, x0_all.shape[0], args.batch):
            xb = x0_all[st:st + args.batch]
            cb = None if class_labels is None else class_labels[st:st + args.batch]
            acc += estimate_sigma2_eta_at_sigma(net, float(s), d, xb, cb, device)
            nb += 1
        sig2_eta[k] = acc / nb
        if (k + 1) % max(1, n // 10) == 0 or k == 0:
            print(f"  sig2_eta {k+1}/{n}  sigma={s:.4g}  val={sig2_eta[k]:.4e}")
    sigma_eta = np.sqrt(np.clip(sig2_eta, 0, None))
    _ckpt(g_values=g_vals, sigma_eta=sigma_eta, sigma2_eta=sig2_eta)

    # ── D_j (field acceleration magnitude) ──
    print(f"[edm-stats] D_j (field accel) at {n} points ...")
    D_j = np.zeros(n)
    for k in range(n):
        acc, nb = 0.0, 0
        for st in range(0, x0_all.shape[0], args.batch):
            xb = x0_all[st:st + args.batch]
            cb = None if class_labels is None else class_labels[st:st + args.batch]
            acc += estimate_Dj_at_index(net, k, sigma_grid, d, xb, cb, device)
            nb += 1
        D_j[k] = acc / nb
        if (k + 1) % max(1, n // 10) == 0:
            print(f"  D_j {k+1}/{n}  val={D_j[k]:.4e}")
    log_D = np.log(np.clip(D_j, 1e-30, None))
    win = max(3, min(7, (n // 4) * 2 + 1))
    D_j_smooth = np.exp(savgol_filter(log_D, window_length=win, polyorder=2))
    dbar_j = np.sqrt(np.clip(D_j_smooth, 0, None))
    _ckpt(D_j=D_j_smooth, dbar_j=dbar_j)

    # ── eta correlation kernel ──
    n_ell = min(args.n_ell_samples, x0_all.shape[0])
    print(f"[edm-stats] eta correlation ({n_ell} samples) ...")
    cl_ell = None if class_labels is None else class_labels[:n_ell]
    _, rho_eta = estimate_eta_correlation(net, sigma_grid, lam_grid, d, x0_all[:n_ell], cl_ell, device)
    dl, rp = [], []
    for i in range(n):
        for j in range(i + 1, n):
            dl.append(abs(lam_grid[i] - lam_grid[j]))
            rp.append(float(rho_eta[i, j]))
    dl, rp = np.array(dl), np.array(rp)
    rho_s, rho_vals = _extract_rho_curve(dl, rp, anchor_at_one=True)
    plateau_eta = analyse_plateau(rho_s, rho_vals, label="eta")
    _ckpt(rho_s=rho_s, rho_values=rho_vals)

    # ── field-accel (curvature) correlation kernel ──
    rho_s_c = rho_vals_c = None
    plateau_curv = None
    if not args.no_curv_corr:
        print(f"[edm-stats] field-accel correlation ({n_ell} samples) ...")
        _, rho_c = estimate_curv_correlation(net, sigma_grid, d, x0_all[:n_ell], cl_ell, device)
        dlc, rpc = [], []
        for i in range(n):
            for j in range(i + 1, n):
                dlc.append(abs(lam_grid[i] - lam_grid[j]))
                rpc.append(float(rho_c[i, j]))
        rho_s_c, rho_vals_c = _extract_rho_curve(np.array(dlc), np.array(rpc), anchor_at_one=False)
        plateau_curv = analyse_plateau(rho_s_c, rho_vals_c, label="curv")

    # ── save (final) ──
    out = out_dir / f"{safe}.npz"
    if rho_s_c is not None:
        _store["rho_s_curv"] = rho_s_c.astype(np.float32)
        _store["rho_values_curv"] = rho_vals_c.astype(np.float32)
    np.savez(str(out), **_store)
    print(f"[edm-stats] saved -> {out}")

    # ── figure ──
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    ax[0, 0].plot(lam_grid, g_vals, color="#2d6a9f"); ax[0, 0].set_title("g(lambda) = (1/d)tr[df/dx]")
    ax[0, 0].set_xlabel("lambda=-log sigma"); ax[0, 0].axhline(0, color="gray", lw=.6, ls="--")
    ax[0, 1].semilogy(sigma_grid, sigma_eta, color="#b5451b"); ax[0, 1].set_title("sigma_eta(sigma) = RMS ||D_theta - x0||")
    ax[0, 1].set_xlabel("sigma"); ax[0, 1].invert_xaxis()
    ax[0, 2].loglog(sigma_grid, D_j_smooth, color="#9b3fa0"); ax[0, 2].set_title("D_j = E||f_dot||^2")
    ax[0, 2].set_xlabel("sigma"); ax[0, 2].invert_xaxis()
    ax[1, 0].scatter(dl, rp, s=3, alpha=.2, color="#aaa")
    ax[1, 0].plot(rho_s, rho_vals, color="#2d6a9f", lw=2)
    ax[1, 0].axhline(plateau_eta["rho_infty"], color="#e05c00", ls="--",
                     label=f"rho_inf={plateau_eta['rho_infty']:.3f}")
    ax[1, 0].set_title("eta correlation rho(|dlambda|)"); ax[1, 0].legend(fontsize=8); ax[1, 0].set_ylim(-.05, 1.05)
    if rho_s_c is not None:
        ax[1, 1].plot(rho_s_c, rho_vals_c, color="#9b3fa0", lw=2)
        if plateau_curv:
            ax[1, 1].axhline(plateau_curv["rho_infty"], color="#e05c00", ls="--",
                             label=f"rho_inf={plateau_curv['rho_infty']:.3f}")
            ax[1, 1].legend(fontsize=8)
        ax[1, 1].set_title("field-accel correlation"); ax[1, 1].set_ylim(-.15, 1.05)
    else:
        ax[1, 1].axis("off")
    # effective eta cost weight including coupling c^2 = 1/sigma^2
    ax[1, 2].loglog(sigma_grid, sigma_eta / sigma_grid, color="#2a7d4f")
    ax[1, 2].set_title("effective |c|*sigma_eta = sigma_eta / sigma\n(low-noise up-weighting)")
    ax[1, 2].set_xlabel("sigma"); ax[1, 2].invert_xaxis()
    fig.suptitle(f"EP-Flow EDM statistics — {safe}")
    fig.tight_layout()
    fig.savefig(str(out.with_suffix(".png")), dpi=150)
    print(f"[edm-stats] figure -> {out.with_suffix('.png')}")


def parse_args():
    p = argparse.ArgumentParser(
        description="EP-Flow statistics for EDM .pkl networks (forward-Euler, sigma-space).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--network", required=True,
                   help="EDM .pkl path or URL, e.g. "
                        "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl")
    p.add_argument("--data_npz", required=True,
                   help="real dataset samples x0, shape [N,C,H,W] or [N,H,W,C]; "
                        "e.g. cifar10 training images. (NOT gaussian noise.)")
    p.add_argument("--output_dir", default=str(Path.home() / ".cache" / "opt_schedule_edm"))
    p.add_argument("--sigma_min", type=float, default=0.002)
    p.add_argument("--sigma_max", type=float, default=80.0)
    p.add_argument("--rho", type=float, default=7.0)
    p.add_argument("--n_sigma", type=int, default=60)
    p.add_argument("--n_samples", type=int, default=256)
    p.add_argument("--n_ell_samples", type=int, default=256)
    p.add_argument("--n_hutchinson", type=int, default=8)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument("--no_curv_corr", action="store_true", help="skip field-accel correlation (faster)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())

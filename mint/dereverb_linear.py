"""
Regularized MINT dereverberation with a linear-convolution matrix.

This module builds the true linear convolution system

    sum_m conv(h_m, g_m) ~= delta[delay]

where h_m is the RIR of microphone m and g_m is the corresponding inverse
filter. It avoids circular shifts and solves a regularized least-squares
problem for numerical stability.
"""

import argparse
import time

import numpy as np
from scipy.signal import fftconvolve
from scipy.sparse import csc_matrix, eye, hstack, vstack
from scipy.sparse.linalg import lsqr


DEFAULT_LCURVE_REG_GRID = np.logspace(-10, 0, 17)


def _as_channel_first_rirs(rir_list):
    """Validate RIRs and return them as a list of 1-D float64 arrays."""
    if not rir_list:
        raise ValueError("rir_list must contain at least one channel")

    rirs = []
    for idx, rir in enumerate(rir_list):
        h = np.asarray(rir, dtype=np.float64).reshape(-1)
        if h.size == 0:
            raise ValueError(f"RIR channel {idx} is empty")
        if not np.all(np.isfinite(h)):
            raise ValueError(f"RIR channel {idx} contains non-finite values")
        rirs.append(h)
    return rirs


def build_linear_convolution_matrix(h, inverse_len):
    """
    Build a sparse linear-convolution matrix H for conv(h, g).

    Parameters
    ----------
    h : ndarray, shape (rir_len,)
        Room impulse response for one channel.
    inverse_len : int
        Length of the inverse filter g.

    Returns
    -------
    H : scipy.sparse.csc_matrix, shape (rir_len + inverse_len - 1, inverse_len)
        Matrix satisfying H @ g == np.convolve(h, g).
    """
    h = np.asarray(h, dtype=np.float64).reshape(-1)
    if inverse_len <= 0:
        raise ValueError("inverse_len must be positive")

    rir_len = h.size
    out_len = rir_len + inverse_len - 1

    data = []
    rows = []
    cols = []
    for col in range(inverse_len):
        rows.extend(range(col, col + rir_len))
        cols.extend([col] * rir_len)
        data.extend(h)

    return csc_matrix((data, (rows, cols)), shape=(out_len, inverse_len))


def build_mint_target(
    rir_list,
    out_len,
    delay,
    target_mode="delta",
    reference_channel=0,
    fs=16000,
    early_ms=64.0,
    target_normalize="none",
    target_alignment="start_at_delay",
):
    """Build the desired equivalent impulse response for MINT/P-MINT."""
    if delay < 0 or delay >= out_len:
        raise ValueError(f"delay must be in [0, {out_len - 1}], got {delay}")
    if target_mode not in ("delta", "pmint_early"):
        raise ValueError(f"unsupported target_mode: {target_mode}")
    if target_normalize not in ("none", "unit_peak"):
        raise ValueError(f"unsupported target_normalize: {target_normalize}")
    if target_alignment not in ("start_at_delay", "direct_at_delay"):
        raise ValueError(f"unsupported target_alignment: {target_alignment}")

    d = np.zeros(out_len, dtype=np.float64)
    info = {
        "target_mode": target_mode,
        "target_reference_channel": None,
        "target_early_ms": None,
        "target_direct_index": None,
        "target_nonzero_start": delay,
        "target_nonzero_stop": delay + 1,
        "target_l2": None,
        "target_linf": None,
        "target_normalize": target_normalize,
        "target_alignment": target_alignment,
        "target_cropped": False,
    }

    if target_mode == "delta":
        d[delay] = 1.0
    else:
        rirs = _as_channel_first_rirs(rir_list)
        reference_channel = int(reference_channel)
        if reference_channel < 0 or reference_channel >= len(rirs):
            raise ValueError(
                f"reference_channel must be in [0, {len(rirs) - 1}], got {reference_channel}")
        if fs <= 0:
            raise ValueError("fs must be positive")
        if early_ms <= 0:
            raise ValueError("early_ms must be positive")

        h_ref = rirs[reference_channel]
        direct_idx = int(np.argmax(np.abs(h_ref)))
        early_len = int(round(float(early_ms) * float(fs) / 1000.0))
        keep_len = min(len(h_ref), direct_idx + early_len)
        h_early = h_ref[:keep_len].copy()
        if target_normalize == "unit_peak":
            peak = h_ref[direct_idx]
            if abs(peak) > 1e-12:
                h_early = h_early / peak

        target_start = delay if target_alignment == "start_at_delay" else delay - direct_idx
        src_start = max(0, -target_start)
        dst_start = max(0, target_start)
        copy_len = min(len(h_early) - src_start, out_len - dst_start)
        if copy_len > 0:
            d[dst_start:dst_start + copy_len] = h_early[src_start:src_start + copy_len]
        info.update({
            "target_reference_channel": reference_channel,
            "target_early_ms": float(early_ms),
            "target_direct_index": direct_idx,
            "target_nonzero_start": dst_start,
            "target_nonzero_stop": dst_start + max(0, copy_len),
            "target_cropped": src_start > 0 or copy_len < len(h_early),
        })

    info["target_l2"] = float(np.linalg.norm(d))
    info["target_linf"] = float(np.max(np.abs(d))) if d.size else 0.0
    return d, info


def _solve_regularized_mint(
    H,
    d,
    inverse_len,
    n_channels,
    reg,
    lsqr_atol=1e-8,
    lsqr_btol=1e-8,
    lsqr_iter_lim=None,
):
    """Solve one regularized MINT LSQR system and return filters and diagnostics."""
    if reg < 0:
        raise ValueError("reg must be non-negative")
    if reg > 0:
        sqrt_reg = np.sqrt(reg)
        A = vstack([H, sqrt_reg * eye(H.shape[1], format="csc")], format="csc")
        b = np.concatenate([d, np.zeros(H.shape[1], dtype=np.float64)])
    else:
        A = H
        b = d

    start = time.time()
    result = lsqr(A, b, atol=lsqr_atol, btol=lsqr_btol, iter_lim=lsqr_iter_lim)
    elapsed = time.time() - start
    coeffs = result[0]
    filters = [
        coeffs[ch * inverse_len : (ch + 1) * inverse_len]
        for ch in range(n_channels)
    ]
    info = {
        "reg": float(reg),
        "augmented_shape": A.shape,
        "lsqr_istop": result[1],
        "lsqr_iterations": result[2],
        "lsqr_r1norm": result[3],
        "lsqr_r2norm": result[4],
        "lsqr_anorm": result[5],
        "lsqr_acond": result[6],
        "elapsed_sec": elapsed,
        "filter_l2": float(np.linalg.norm(coeffs)),
        "filter_linf": float(np.max(np.abs(coeffs))) if coeffs.size else 0.0,
    }
    return filters, coeffs, info


def select_lcurve_lambda_triangle(regs, filter_l2, residual_l2, min_points=5):
    """Select the L-curve corner using the log-log triangle distance method."""
    regs = np.asarray(regs, dtype=np.float64)
    filter_l2 = np.asarray(filter_l2, dtype=np.float64)
    residual_l2 = np.asarray(residual_l2, dtype=np.float64)
    valid = (
        np.isfinite(regs) & np.isfinite(filter_l2) & np.isfinite(residual_l2)
        & (regs > 0) & (filter_l2 > 0) & (residual_l2 > 0)
    )
    idxs = np.flatnonzero(valid)
    diagnostics = {
        "lcurve_status": "ok",
        "lcurve_selected_index": None,
        "lcurve_corner_distance": None,
        "lcurve_num_valid": int(len(idxs)),
    }
    if len(idxs) < int(min_points):
        diagnostics["lcurve_status"] = "fallback_too_few_points"
        selected = int(idxs[len(idxs) // 2]) if len(idxs) else 0
        diagnostics["lcurve_selected_index"] = selected
        return selected, diagnostics

    idxs = idxs[np.argsort(regs[idxs])]
    x = np.log10(filter_l2[idxs])
    y = np.log10(residual_l2[idxs])
    p0 = np.array([x[0], y[0]], dtype=np.float64)
    p1 = np.array([x[-1], y[-1]], dtype=np.float64)
    line = p1 - p0
    denom = np.linalg.norm(line)
    if denom < 1e-12:
        diagnostics["lcurve_status"] = "fallback_degenerate_line"
        selected = int(idxs[len(idxs) // 2])
        diagnostics["lcurve_selected_index"] = selected
        return selected, diagnostics

    distances = []
    for xi, yi in zip(x, y):
        point = np.array([xi, yi], dtype=np.float64)
        distances.append(abs(np.cross(line, point - p0)) / denom)
    distances = np.asarray(distances)
    local_idx = int(np.argmax(distances))
    selected = int(idxs[local_idx])
    diagnostics["lcurve_selected_index"] = selected
    diagnostics["lcurve_corner_distance"] = float(distances[local_idx])
    return selected, diagnostics


def design_mint_inverse_filters(
    rir_list,
    inverse_len=None,
    delay=None,
    reg=1e-6,
    target_mode="delta",
    reference_channel=0,
    fs=16000,
    early_ms=64.0,
    target_normalize="none",
    target_alignment="start_at_delay",
    reg_mode="fixed",
    reg_grid=None,
    lcurve_min_points=5,
    lsqr_atol=1e-8,
    lsqr_btol=1e-8,
    lsqr_iter_lim=None,
    verbose=False,
):
    """
    Design multichannel MINT inverse filters using regularized LSQR.

    The solved objective is

        min_g ||H g - d||_2^2 + reg * ||g||_2^2

    where H horizontally stacks one linear-convolution matrix per channel.
    """
    rirs = _as_channel_first_rirs(rir_list)
    n_channels = len(rirs)
    max_rir_len = max(len(h) for h in rirs)

    if inverse_len is None:
        inverse_len = max_rir_len
    if inverse_len <= 0:
        raise ValueError("inverse_len must be positive")

    out_len = max(len(h) + inverse_len - 1 for h in rirs)
    if delay is None:
        delay = max_rir_len - 1
    if delay < 0 or delay >= out_len:
        raise ValueError(f"delay must be in [0, {out_len - 1}], got {delay}")
    if reg < 0:
        raise ValueError("reg must be non-negative")
    if reg_mode not in ("fixed", "lcurve"):
        raise ValueError(f"unsupported reg_mode: {reg_mode}")

    blocks = []
    for h in rirs:
        block = build_linear_convolution_matrix(h, inverse_len)
        if block.shape[0] < out_len:
            pad = csc_matrix((out_len - block.shape[0], inverse_len))
            block = vstack([block, pad], format="csc")
        blocks.append(block)

    H = hstack(blocks, format="csc")
    d, target_info = build_mint_target(
        rirs,
        out_len,
        delay,
        target_mode=target_mode,
        reference_channel=reference_channel,
        fs=fs,
        early_ms=early_ms,
        target_normalize=target_normalize,
        target_alignment=target_alignment,
    )

    start_all = time.time()
    lcurve_info = {}
    if reg_mode == "lcurve":
        grid = DEFAULT_LCURVE_REG_GRID if reg_grid is None else np.asarray(reg_grid, dtype=np.float64)
        grid = np.asarray(grid, dtype=np.float64).reshape(-1)
        if grid.size == 0:
            raise ValueError("reg_grid must not be empty when reg_mode='lcurve'")
        if np.any(~np.isfinite(grid)) or np.any(grid < 0):
            raise ValueError("reg_grid must contain finite non-negative values")
        if verbose:
            print(f"MINT L-curve: solving {len(grid)} regularization points...", flush=True)
        candidates = []
        for candidate_reg in grid:
            cand_filters, cand_coeffs, cand_info = _solve_regularized_mint(
                H, d, inverse_len, n_channels, float(candidate_reg),
                lsqr_atol=lsqr_atol, lsqr_btol=lsqr_btol, lsqr_iter_lim=lsqr_iter_lim)
            cand_combined = combined_impulse_response(rirs, cand_filters)
            cand_residual = cand_combined[:out_len] - d
            cand_info["residual_l2"] = float(np.linalg.norm(cand_residual))
            cand_info["residual_linf"] = float(np.max(np.abs(cand_residual)))
            candidates.append((cand_filters, cand_coeffs, cand_info))
        regs = [item[2]["reg"] for item in candidates]
        filter_norms = [item[2]["filter_l2"] for item in candidates]
        residual_norms = [item[2]["residual_l2"] for item in candidates]
        selected_idx, triangle_info = select_lcurve_lambda_triangle(
            regs, filter_norms, residual_norms, min_points=lcurve_min_points)
        filters, coeffs, solver_info = candidates[selected_idx]
        lcurve_info = {
            "input_reg": float(reg),
            "selected_reg": float(solver_info["reg"]),
            "lcurve_regs": [float(v) for v in regs],
            "lcurve_filter_l2": [float(v) for v in filter_norms],
            "lcurve_residual_l2": [float(v) for v in residual_norms],
            "lcurve_num_points": int(len(candidates)),
            "lcurve_min_reg": float(np.min(grid)),
            "lcurve_max_reg": float(np.max(grid)),
        }
        lcurve_info.update(triangle_info)
    else:
        filters, coeffs, solver_info = _solve_regularized_mint(
            H, d, inverse_len, n_channels, float(reg),
            lsqr_atol=lsqr_atol, lsqr_btol=lsqr_btol, lsqr_iter_lim=lsqr_iter_lim)
        lcurve_info = {
            "input_reg": float(reg),
            "selected_reg": float(reg),
            "lcurve_status": "not_run",
            "lcurve_selected_index": None,
            "lcurve_num_points": 0,
        }

    combined = combined_impulse_response(rirs, filters)
    residual = combined[:out_len] - d
    elapsed = time.time() - start_all
    info = {
        "n_channels": n_channels,
        "max_rir_len": max_rir_len,
        "inverse_len": inverse_len,
        "output_len": out_len,
        "delay": delay,
        "reg": solver_info["reg"],
        "reg_mode": reg_mode,
        "matrix_shape": H.shape,
        "augmented_shape": solver_info["augmented_shape"],
        "lsqr_istop": solver_info["lsqr_istop"],
        "lsqr_iterations": solver_info["lsqr_iterations"],
        "lsqr_r1norm": solver_info["lsqr_r1norm"],
        "lsqr_r2norm": solver_info["lsqr_r2norm"],
        "lsqr_anorm": solver_info["lsqr_anorm"],
        "lsqr_acond": solver_info["lsqr_acond"],
        "elapsed_sec": elapsed,
        "filter_l2": solver_info["filter_l2"],
        "filter_linf": solver_info["filter_linf"],
        "combined_peak_index": int(np.argmax(np.abs(combined))),
        "combined_peak_value": float(combined[np.argmax(np.abs(combined))]),
        "residual_l2": float(np.linalg.norm(residual)),
        "residual_linf": float(np.max(np.abs(residual))),
    }
    info.update(target_info)
    info.update(lcurve_info)

    if verbose:
        print(
            "MINT LSQR: "
            f"H={H.shape}, A={info['augmented_shape']}, delay={delay}, "
            f"target={target_mode}, alignment={target_alignment}, reg_mode={reg_mode}, reg={info['reg']}, "
            f"iters={info['lsqr_iterations']}, acond={info['lsqr_acond']:.3e}, "
            f"residual_l2={info['residual_l2']:.3e}, "
            f"filter_l2={info['filter_l2']:.3e}, elapsed={elapsed:.2f}s",
            flush=True,
        )

    return filters, info


def combined_impulse_response(rir_list, inverse_filters):
    """Return sum_m conv(rir_m, inverse_filter_m)."""
    rirs = _as_channel_first_rirs(rir_list)
    if len(rirs) != len(inverse_filters):
        raise ValueError("rir_list and inverse_filters must have the same length")

    convs = [fftconvolve(h, np.asarray(g, dtype=np.float64), mode="full") for h, g in zip(rirs, inverse_filters)]
    out_len = max(len(c) for c in convs)
    combined = np.zeros(out_len, dtype=np.float64)
    for c in convs:
        combined[:len(c)] += c
    return combined


def mint_multichannel_inverse_linear(
    y,
    rir_list,
    inverse_len=None,
    delay=None,
    reg=1e-6,
    target_mode="delta",
    reference_channel=0,
    fs=16000,
    early_ms=64.0,
    target_normalize="none",
    target_alignment="start_at_delay",
    reg_mode="fixed",
    reg_grid=None,
    trim_to_input=True,
    return_filters=False,
    verbose=False,
):
    """
    Apply regularized linear-convolution MINT to multichannel observations.

    Parameters
    ----------
    y : ndarray, shape (samples, channels)
        Observed multichannel signal with one microphone per column.
    rir_list : list of ndarray
        One 1-D RIR per microphone channel.
    inverse_len, delay, reg : see design_mint_inverse_filters.
    trim_to_input : bool
        If True, return the first samples entries after inverse filtering.
    return_filters : bool
        If True, also return inverse filters and diagnostics.
    verbose : bool
        Print solver diagnostics.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("y must have shape (samples, channels)")

    rirs = _as_channel_first_rirs(rir_list)
    if y.shape[1] != len(rirs):
        raise ValueError(f"y has {y.shape[1]} channels but rir_list has {len(rirs)} RIRs")

    filters, info = design_mint_inverse_filters(
        rirs,
        inverse_len=inverse_len,
        delay=delay,
        reg=reg,
        target_mode=target_mode,
        reference_channel=reference_channel,
        fs=fs,
        early_ms=early_ms,
        target_normalize=target_normalize,
        target_alignment=target_alignment,
        reg_mode=reg_mode,
        reg_grid=reg_grid,
        verbose=verbose,
    )

    out_len = y.shape[0] + max(len(g) for g in filters) - 1
    s_hat = np.zeros(out_len, dtype=np.float64)
    for ch, g in enumerate(filters):
        s_hat[: y.shape[0] + len(g) - 1] += fftconvolve(y[:, ch], g, mode="full")

    if trim_to_input:
        s_hat = s_hat[: y.shape[0]]

    if return_filters:
        return s_hat, filters, info
    return s_hat


def _run_smoke_test():
    """Run a small synthetic test to verify the linear MINT construction."""
    rng = np.random.default_rng(2026)
    h1 = np.array([0.0, 1.0, 0.35, -0.12, 0.04])
    h2 = np.array([0.0, 0.7, -0.28, 0.10, 0.03])
    clean = rng.standard_normal(4096)
    y = np.column_stack([
        fftconvolve(clean, h1, mode="full")[:len(clean)],
        fftconvolve(clean, h2, mode="full")[:len(clean)],
    ])

    s_hat, filters, info = mint_multichannel_inverse_linear(
        y,
        [h1, h2],
        inverse_len=16,
        delay=4,
        reg=1e-8,
        return_filters=True,
        verbose=True,
    )
    combined = combined_impulse_response([h1, h2], filters)
    aligned = s_hat[4:]
    ref = clean[:len(aligned)]
    mse = np.mean((aligned - ref) ** 2)

    print("Smoke test:")
    print(f"  combined peak index : {info['combined_peak_index']}")
    print(f"  combined peak value : {info['combined_peak_value']:.6f}")
    print(f"  residual_l2         : {info['residual_l2']:.3e}")
    print(f"  filter_l2           : {info['filter_l2']:.3e}")
    print(f"  aligned MSE         : {mse:.3e}")
    print(f"  combined IR first 12: {np.array2string(combined[:12], precision=4)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Regularized linear-convolution MINT smoke test.")
    parser.add_argument("--smoke-test", action="store_true", help="Run the built-in synthetic smoke test.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.smoke_test:
        _run_smoke_test()
    else:
        print("Run with --smoke-test to verify the implementation.")

"""
sdxtg_wave.py  --  Wave-optics (Fresnel) check of the SD-XTG beam path, split axis, 1-D

The ray model in sdxtg_sim.py is geometric.  This module propagates the actual complex field
along the split axis with the Fresnel diffraction integral, so it includes what the rays
leave out: edge diffraction from the beamstop and slit, the finite coherence of the beam,
the real interference of A(-1) and B(+1) (the transient-grating fringes) and their contrast.

Method (paraxial, scalar, thin elements):
  * The field is written as U(x, z) = A(x, z) · exp(-i k x² / 2R(z)), with R(z) the distance to
    the KB focus of the split axis.  Free-space propagation over d then follows the Fresnel
    scaling theorem:   A(x₂, z+d) = M^(-1/2) · F_{d/M}[A](x₂ / M),   M = (R - d) / R,
    so the converging beam can be sampled on a grid that shrinks with the beam, and every
    Fresnel step is an exact FFT transfer-function propagation of the smooth envelope A.
  * KB mirror = ideal thin lens (the reference sphere).  Partial transverse coherence =
    Gaussian Schell model with coherence length l_c = coh_ratio × rms beam size on the KB,
    written as an incoherent sum of beams tilted by α (σ_α = 1/(k l_c)), i.e. converging to
    focal points ξ = α f.  The 40 µm focal spot of the ray model is NOT reproduced: it comes from
    mirror errors / source properties that are not modelled here, so the wave focus is the ideal one.
  * Bandwidth = incoherent sum over wavelengths.
  * Beamstop and slit = hard binary masks.  Grating = thin transmission
    t(x) = Σ E_n exp(i 2π n x / p) with the same E_n as the ray model, |n| ≤ n_max.
    Each order is propagated separately as envelope × carrier exp(i 2π n x / p): the Fresnel step
    moves and rescales the envelope and the carrier is carried analytically, so the grid never has
    to resolve the grating period (works down to periods of tens of nm, i.e. TG periods of ~10 nm).
    The orders are added coherently at the sample on a grid fine enough for the TG fringes.
    Validity: paraxial about the beam axis; at 1st-order angles θ the relative error of the
    defocus term is ~1.5 θ² (≈ 0.1 % at Λ = 50 nm, ≈ 4 % at Λ = 15 nm); the transverse
    wavevectors (hence Λ) are exact.
  * The CCD plane (the focus) is reached with a single Fraunhofer-type transform.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import sdxtg_sim as sim

FWHM = sim.FWHM_PER_SIGMA


def _next_pow2(n):
    return 1 << int(math.ceil(math.log2(max(n, 2))))


def _fresnel(A, dx, lam, L):
    """Fresnel transfer-function propagation of envelope A (same grid in and out)."""
    if L == 0:
        return A
    nu = np.fft.fftfreq(len(A), dx)
    return np.fft.ifft(np.fft.fft(A) * np.exp(-1j * math.pi * lam * L * nu ** 2))


def _step(A, x, lam, R, d):
    """Propagate the envelope A(x) relative to a sphere of radius R by d.
    Returns (A', x', R') on the scaled grid."""
    M = (R - d) / R
    A2 = _fresnel(A, x[1] - x[0], lam, d / M) / math.sqrt(M)
    return A2, x * M, R - d


@dataclass
class WaveResult:
    planes: dict                      # name -> (x [m], intensity)
    ccd: tuple                        # (x [m], intensity) on the CCD (direct beams only)
    carpet: tuple | None              # (z [m], x [m], I[z, x]) coherent, binned
    visibility: float                 # TG fringe visibility at the sample (partially coherent)
    visibility_coh: float             # same for a single coherent component
    tg_period_nm: float
    info: dict = field(default_factory=dict)


def _step_pad(A, x, lam, R, d, nmax=1 << 20):
    """_step, zero-padding the grid first if it is too narrow for the transfer function at this distance."""
    M = (R - d) / R
    dx = x[1] - x[0]
    need = lam * abs(d / M) / dx ** 2 * 1.2
    if need > len(A):
        N = min(_next_pow2(need), nmax)
        if N > len(A):
            pad = N - len(A)
            lft = pad // 2
            A = np.pad(A, (lft, pad - lft))
            x = x[0] + (np.arange(N) - lft) * dx
    return _step(A, x, lam, R, d)


def _ostep(o, lam, R, d):
    """Propagate one diffraction order, stored as envelope a(x) times the carrier exp(i(q x + phi)).
    Fresnel scaling theorem with a tilt: the carrier is carried analytically, so the grid only has to
    resolve the envelope, never the grating period (needed for periods of tens of nm)."""
    k = 2 * math.pi / lam
    M = (R - d) / R
    L = d / M
    a2, x2, R2 = _step_pad(o["a"], o["x"], lam, R, d)
    return dict(n=o["n"], a=a2, x=x2 + M * o["q"] * L / k, q=o["q"] / M,
                phi=o["phi"] - o["q"] ** 2 * L / (2 * k)), R2


def _cinterp(xn, x, a):
    return np.interp(xn, x, a.real, 0, 0) + 1j * np.interp(xn, x, a.imag, 0, 0)


def _union_intensity(orders, max_n=1 << 18):
    """Sum of the order intensities on one lab grid (orders are mutually incoherent once averaged
    over the TG fringes, which is how these planes are compared with the ray model)."""
    lo = min(o["x"][0] for o in orders)
    hi = max(o["x"][-1] for o in orders)
    dx = min(o["x"][1] - o["x"][0] for o in orders)
    n = int(min((hi - lo) / dx + 1, max_n))
    xg = np.linspace(lo, hi, n)
    I = np.zeros(n)
    for o in orders:
        I += np.interp(xg, o["x"], np.abs(o["a"]) ** 2, 0, 0)
    return xg, I


def _coherent(orders, xf):
    U = np.zeros(len(xf), complex)
    for o in orders:
        if o["x"][-1] < xf[0] or o["x"][0] > xf[-1]:
            continue
        U += _cinterp(xf, o["x"], o["a"]) * np.exp(1j * (o["q"] * xf + o["phi"]))
    return np.abs(U) ** 2


def _grids(p: sim.Params, res: sim.Result, n_max):
    """Coarse grid on the mirror (resolves the envelope and the beamstop-edge fringes) and the fine
    grid at the sample (resolves the TG fringes)."""
    lam = p.wavelength_nm * 1e-9
    zm, f = sim._axis_geo(p, p.split_axis)
    D = (p.beam_fwhm_x_mm if p.split_axis == "x" else p.beam_fwhm_y_mm) * 1e-3
    zf = zm + f
    R_g = zf - p.z_grating
    W1 = 3.2 * D
    L1 = (p.z_bs - zm) / ((f - (p.z_bs - zm)) / f)
    M1 = (f - (p.z_bs - zm)) / f
    L2 = (p.z_grating - p.z_bs) / (R_g / (f - (p.z_bs - zm)))
    du = min(D / 1500, W1 / 4096)
    N1 = _next_pow2(max(W1 / du, lam * max(L1, L2 / M1 ** 2) / du ** 2 * 1.2))
    N1 = min(N1, 1 << 17)
    tg = res.metrics.get("tg_period_nm", p.period_nm / 2) * 1e-9
    fw = res.metrics.get("fwhm_A_s_um", 100.0) * 1e-6
    lo_s, hi_s = res.slit
    Wf = 1.15 * max(abs(lo_s), abs(hi_s)) + 2.6 * fw
    dxf = tg / 10
    Nf = int(2 * Wf / dxf)
    if Nf > (1 << 21):                       # very small TG period: keep only the beam centre
        Wf = 2.6 * fw
        Nf = int(min(2 * Wf / dxf, 1 << 21))
    return dict(lam0=lam, zm=zm, f=f, zf=zf, D=D, W1=W1, N1=N1, R_g=R_g, Wf=Wf, Nf=Nf)


def wave_check(p: sim.Params, res: sim.Result | None = None, n_xi=9, n_lam=5, n_max=5,
               carpet=True, nz_carpet=220, progress=None) -> WaveResult:
    res = res or sim.simulate(p)
    p = res.p                                 # grating period already solved in target mode
    z = res.z
    g = _grids(p, res, n_max)
    lam0, zm, f, zf = g["lam0"], g["zm"], g["f"], g["zf"]
    if not (zm < p.z_bs < p.z_grating < z["slit"] < z["sample"] < zf):
        raise ValueError("wave check needs mirror < beamstop < grating < slit < sample < focus")
    sigI = g["D"] / FWHM                       # intensity sigma of the input beam
    l_c = p.coh_ratio * sigI
    sig_f = f / (2 * math.pi / lam0 * l_c)             # rms spread of focal points (GSM)
    # source points (focal-plane positions) and wavelengths, Gauss-Hermite weights
    xi_n, xi_w = np.polynomial.hermite_e.hermegauss(n_xi) if n_xi > 1 else (np.array([0.]), np.array([1.]))
    bw = p.bandwidth_pct / 100 / FWHM
    if n_lam > 1 and bw > 0:
        l_n, l_w = np.polynomial.hermite_e.hermegauss(n_lam)
    else:
        l_n, l_w = np.array([0.]), np.array([1.])
    xi_w, l_w = xi_w / xi_w.sum(), l_w / l_w.sum()
    per = p.period_nm * 1e-9
    orders = [n for n in range(-n_max, n_max + 1) if abs(n) * p.wavelength_nm / p.period_nm < 0.95]
    En = sim.grating_coefficients(p, np.array(orders))
    u = (np.arange(g["N1"]) - g["N1"] / 2) * (g["W1"] / g["N1"])
    lo_s, hi_s = res.slit
    xf = np.linspace(-g["Wf"], g["Wf"], g["Nf"])          # fine grid at the sample

    acc = {}
    comp = 0
    tot = len(l_n) * len(xi_n)
    for li, lw in zip(l_n, l_w):
        lam = lam0 * (1 + li * bw)
        k = 2 * math.pi / lam
        for xn, xw in zip(xi_n, xi_w):
            xi = xn * sig_f
            if p.profile == "flattop":
                a0 = 1.0 / (1 + np.exp((np.abs(u) - g["D"] / 2) / (g["D"] / 400)))
            else:
                a0 = np.exp(-u ** 2 / (4 * sigI ** 2))
            A = a0 * np.exp(1j * k * u * xi / f)               # beam converging to ξ
            A, x, R = _step_pad(A.astype(complex), u, lam, f, p.z_bs - zm)
            A = A * (np.abs(x - p.bs_offset_mm * 1e-3) >= p.bs_width_mm * 1e-3 / 2)
            A, x, R = _step_pad(A, x, lam, R, p.z_grating - p.z_bs)
            R_g = R
            # the grating: one envelope per order, carrier exp(i 2 pi n x / p)
            og = [dict(n=n, a=e * A, x=x, q=2 * math.pi * n / per, phi=0.0) for n, e in zip(orders, En)
                  if abs(e) > 1e-6]
            out = {"grating": (x, np.abs(A) ** 2 * sum(abs(e) ** 2 for e in En))}
            osl = []
            for o in og:
                o2, Rs = _ostep(o, lam, R_g, z["slit"] - p.z_grating)
                osl.append(o2)
            out["slit"] = _union_intensity(osl)
            # slit blades, then on to the sample
            opass = []
            for o in osl:
                a = o["a"] * ((o["x"] >= lo_s) & (o["x"] <= hi_s))
                if np.any(a != 0):
                    opass.append(dict(o, a=a))
            osa = []
            for o in opass:
                o2, Rb = _ostep(o, lam, Rs, z["sample"] - z["slit"])
                osa.append(o2)
            out["sample"] = (xf, _coherent(osa, xf))
            if comp == 0:
                ref = dict(lam=lam, og=og, R_g=R_g, opass=opass, Rs=Rs, sample=out["sample"])
            # CCD: per order, one Fraunhofer-type transform from the sample plane
            d = p.z_det - z["sample"]
            ccd = []
            for o in osa:
                xb = o["x"]
                dx = xb[1] - xb[0]
                chirp = np.exp(1j * k * xb ** 2 / 2 * (1 / d - 1 / Rb)) if abs(d - Rb) > 1e-9 else 1.0
                F = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(o["a"] * chirp)))
                nu = np.fft.fftshift(np.fft.fftfreq(len(xb), dx)) + o["q"] / (2 * math.pi)
                ccd.append(dict(x=nu * lam * d, a=F / math.sqrt(len(F))))
            out["ccd"] = _union_intensity(ccd) if ccd else (np.array([0.0, 1.0]), np.zeros(2))
            w = lw * xw
            for key, (xx, I) in out.items():
                if key not in acc:
                    acc[key] = [xx, w * I]
                else:
                    acc[key][1] = acc[key][1] + w * (I if len(I) == len(acc[key][1]) and xx[0] == acc[key][0][0]
                                                     else np.interp(acc[key][0], xx, I, 0, 0))
            comp += 1
            if progress:
                progress(0.8 * comp / tot)

    planes = {k: tuple(v) for k, v in acc.items() if k != "ccd"}
    tg = res.metrics["tg_period_nm"] * 1e-9

    def vis(xx, I):
        """TG modulation depth: 2|Ĩ(1/Λ)| / Ĩ(0) over the sample (beam-weighted, like the ray metric)."""
        c = np.abs(xx) < 2.5 * res.metrics["fwhm_A_s_um"] * 1e-6
        if c.sum() < 50:
            return float("nan")
        x, y = xx[c], I[c]
        dx = x[1] - x[0]
        F = np.abs(np.fft.fft(y)) ** 2
        nu = np.fft.fftfreq(len(y), dx)
        band = (nu > 0.7 / tg) & (nu < 1.4 / tg)          # the +1/Λ fringe peak
        low = np.abs(nu) < 0.3 / tg                        # the envelope (both sides of DC)
        if not band.any():
            return float("nan")
        # I = I0 (1 + V cos): the +1/Λ peak carries (V/2)² of the envelope power
        return float(2 * math.sqrt(F[band].sum() / F[low].sum()))

    V = vis(*planes["sample"])
    Vc = vis(*ref["sample"])
    # TG period actually present in the wave field: peak of the fringe spectrum
    xx, I = ref["sample"]
    c = np.abs(xx) < 2.5 * res.metrics["fwhm_A_s_um"] * 1e-6
    Fs = np.abs(np.fft.rfft(I[c] - I[c].mean()))
    nu = np.fft.rfftfreq(c.sum(), xx[1] - xx[0])
    sel = nu > 0.5 / tg
    tg_wave = 1e9 / nu[sel][np.argmax(Fs[sel])] if sel.any() else float("nan")

    car = None
    if carpet:
        car = _carpet(p, res, ref, nz_carpet, progress)
    if progress:
        progress(1.0)
    return WaveResult(planes, tuple(acc["ccd"]), car, V, Vc, tg_wave,
                      info=dict(N_grating=g["N1"], dx_grating_nm=float((x[1] - x[0]) * 1e9), n_xi=len(xi_n),
                                n_lam=len(l_n), n_max=n_max, N_sample=g["Nf"],
                                dx_sample_nm=float((xf[1] - xf[0]) * 1e9), tg_ray_nm=res.metrics["tg_period_nm"]))


def _carpet(p, res, ref, nz, progress=None, nx=700):
    """|U|² (one source point, central λ) in the x-z plane from the grating to just past the sample,
    binned to nx columns (bins average over the TG fringes, so the orders add in intensity)."""
    z = res.z
    lam = ref["lam"]
    zs, zl = z["sample"], z["slit"]
    z_end = min(zs + 0.35 * (zs - p.z_grating), z["dbs"], res.z["det"] - 1e-3,
                sim._axis_geo(p, p.split_axis)[0] + sim._axis_geo(p, p.split_axis)[1] - 1e-3)
    zz = np.linspace(p.z_grating, z_end, nz)
    xmax = 1.25 * float(np.max(np.abs(res.rays.s(p.z_grating, np.concatenate(
        [b.idx for b in res.bundles if b.wanted])))))
    edges = np.linspace(-xmax, xmax, nx + 1)
    img = np.zeros((nz, nx))
    for i, zi in enumerate(zz):
        if zi <= zl:
            ol, R0, d = ref["og"], ref["R_g"], zi - p.z_grating
        else:
            ol, R0, d = ref["opass"], ref["Rs"], zi - zl
        h = np.zeros(nx)
        cnt = np.zeros(nx)
        for o in ol:
            o2, _ = _ostep(o, lam, R0, d) if d > 0 else (o, R0)
            hh, _ = np.histogram(o2["x"], edges, weights=np.abs(o2["a"]) ** 2)
            cc, _ = np.histogram(o2["x"], edges)
            h += np.where(cc > 0, hh / np.maximum(cc, 1), 0)
            cnt += cc
        img[i] = np.where(cnt > 0, h, np.nan)                     # nan: outside the computed grids
        if progress and i % 20 == 0:
            progress(0.8 + 0.2 * i / nz)
    return zz, 0.5 * (edges[1:] + edges[:-1]), img


def ray_profile(res: sim.Result, zpos, xgrid, masked):
    """Ray-model intensity on xgrid (same normalisation idea as the wave: total power)."""
    R = res.rays
    I = np.zeros(len(xgrid) - 1)
    for b in res.bundles:
        k = b.idx[b.slit_mask] if masked else b.idx
        if len(k):
            h, _ = np.histogram(R.s(zpos, k, b.order), xgrid, weights=np.full(len(k), b.eff))
            I += h
    return 0.5 * (xgrid[1:] + xgrid[:-1]), I


if __name__ == "__main__":
    import time
    for case in ("pink", "mono"):
        p = sim.preset(case)
        t = time.time()
        w = wave_check(p, carpet=False)
        print(f"{case}: TG fringe visibility {w.visibility:.2f} (coherent {w.visibility_coh:.2f}), "
              f"Λ = {w.tg_period_nm:.0f} nm, {w.info}, {time.time() - t:.1f} s")

"""
sdxtg_sim.py  --  Beam-path simulator for Self-Diffraction X-ray Transient Grating (SD-XTG)
==========================================================================================

Physics core + correlation scans. Import it, run it from the command line, or drive it
from the GUI (sdxtg_gui.py).

Beamline modelled (all z positions in metres, measured from the HORIZONTAL KB mirror):

  beam ──► [HM, f_H] ──0.5 m──► [VM, f_V] ──► [beamstop / splitter] ──► 2 half-beams
       ──► [grating, period p] ──► orders n = -N..N of each half-beam
       ──► [slit] (keeps only A(-1) and B(+1)) ──► [sample] (the two 1st orders cross: TG)
       ──► [detector = KB focal plane, Princeton CCD] with a beamstop on the direct beams

Default assumptions (all of them are parameters, change them in Params / the GUI):
  * Each focal length is measured from its own mirror: HM at z=0 (f_H=2.0 m),
    VM at z=0.5 m (f_V=1.5 m)  ->  both foci land at z = 2.0 m = detector plane.
  * Focal spot ~40 um FWHM in both directions (set by source demagnification / slope
    errors; the diffraction limit at 2.33 nm is only ~1 um, so 40 um is an input).
  * Input beam right before the KB: 4 mm FWHM (pink) or 1 mm FWHM (mono), Gaussian.
  * The beamstop is a bar across the beam (along the "split" axis) that cuts out the
    central stripe, leaving two half-beams A (x>0) and B (x<0) that still converge
    towards the common KB focus.
  * The grating lines are perpendicular to the split axis. Half-beam A is bent towards
    the axis by order -1, half-beam B by order +1; those two orders cross at the sample.
  * Sample is placed at the crossing point of the two 1st orders ("crossing" mode) or at a
    fixed z ("fixed" mode, then the overlap tells you how well grating z/period match).
  * Grating efficiency from the Fourier coefficients of a binary phase/amplitude grating
    (same expression as EnxPhase in talbot_tutorial.ipynb, without the Hamming taper).

Method: paraxial ray Monte-Carlo.  Every ray carries (u, xi, lambda) per axis:
  u   = transverse position at the mirror (input beam profile),
  xi  = position in the focal plane (focal spot blur),
  so x(z) = u + (xi - u) * (z - z_mirror)/f   for z > z_mirror  (exact for an ideal lens
  fed by a beam whose waist reproduces the 40 um spot).  After the grating a ray in order
  n gets the extra slope n*lambda/p.  Each order of each half-beam is a weighted bundle.
  Fresnel edge diffraction from the beamstop (~sqrt(lambda*L) ~ 50 um over 1 m) is NOT in
  the ray model -- keep a margin of that size on clearances.

Requires: numpy, matplotlib (only for the plotting helpers / CLI).
Run:  python sdxtg_sim.py            -> prints both cases and writes scan maps/CSVs
"""
from __future__ import annotations

import csv
import dataclasses
from dataclasses import dataclass, field, asdict
import json
import math

import numpy as np

C_LIGHT = 299_792_458.0
FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))   # 2.3548


# ─────────────────────────────────────────────────────────────────────────────
#  Parameters
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Params:
    # source / beam right before the KB
    wavelength_nm: float = 2.33          # O K-edge (~532 eV)
    bandwidth_pct: float = 0.5           # dE/E FWHM in %
    beam_fwhm_x_mm: float = 4.0
    beam_fwhm_y_mm: float = 4.0
    profile: str = "gaussian"            # "gaussian" | "flattop"
    pulse_energy_uJ: float = 100.0       # on the KB (for absolute fluence), set your value
    # transverse coherence (Gaussian Schell model) of the beam on the KB, split axis:
    # coherence length = coh_ratio x rms beam size.  2.7 ~ 80 % global degree of coherence
    # (typical SASE soft X-ray); use a large number (e.g. 1e3) for a fully coherent beam.
    coh_ratio: float = 2.7
    kb_transmission: float = 1.0         # product of the two mirror reflectivities

    # KB (z measured from the horizontal mirror)
    z_hm: float = 0.0
    f_h: float = 2.0
    z_vm: float = 0.5
    f_v: float = 1.5
    focus_fwhm_x_um: float = 40.0
    focus_fwhm_y_um: float = 40.0

    # beam splitter (beamstop bar after the KB)
    split_axis: str = "x"                # "x" (horizontal) or "y" (vertical)
    z_bs: float = 0.8
    bs_width_mm: float = 2.0
    bs_offset_mm: float = 0.0

    # grating
    z_grating: float = 1.80
    period_nm: float = 400.0
    duty: float = 0.5                    # bar fraction of the period
    grating_type: str = "amplitude"      # "amplitude" (opaque bars) | "phase"
    bar_phase_rad: float = math.pi       # phase grating: bar phase shift
    bar_T: float = 0.0                   # intensity transmission of bars (0 = opaque)
    membrane_T: float = 1.0              # intensity transmission of the support membrane
    max_order: int = 3
    # target transient-grating period: settings whose TG period falls outside
    # [tg_min_nm, tg_max_nm] (with tg_tol_pct tolerance) are discarded; min == max = one exact period
    tg_min_nm: float = 20.0
    tg_max_nm: float = 300.0
    tg_tol_pct: float = 2.0
    # "target": the grating period is solved so that the TG period equals tg_set_nm;
    # "manual": period_nm is used as given
    period_mode: str = "target"
    tg_set_nm: float = 50.0

    # sample / slit / detector
    sample_mode: str = "crossing"        # "crossing" | "fixed"
    z_sample: float = 1.90               # used in "fixed" mode
    sample_tilt_mrad: float = 0.0        # rotation of the sample normal from the z axis (split plane)

    # experiment: "sd" = X-ray self-diffraction on the CCD, "optical" = X-ray pump, optical probe
    experiment: str = "sd"
    probe_wavelength_nm: float = 400.0   # optical probe
    probe_auto_bragg: bool = True        # put the probe at the Bragg (phase-matched) angle
    probe_angle_deg: float = 30.0        # external incidence angle from the sample normal (split plane)
    # sample: SrTiO3, 10 µm. n at 400 nm ~2.65 (approximate literature value; STO is transparent
    # below its 3.2 eV gap). X-ray 1/e intensity attenuation length 159 nm at 532 eV (Henke/CXRO
    # tables, rho = 5.12 g/cm3): the TG amplitude decays as exp(-z / xray_atten_nm) into the sample.
    sample_formula: str = "SrTiO3"
    sample_density: float = 5.12         # g/cm3, for the Henke-table attenuation length
    sample_n: float = 2.65               # sample refractive index at the probe wavelength
    sample_thickness_nm: float = 10000.0
    xray_atten_nm: float = 159.0         # X-ray attenuation length (1/e of the intensity)
    slit_upstream_mm: float = 1.0        # slit distance upstream of the sample
    slit_auto: bool = True               # open the slit just around the wanted orders
    slit_width_um: float = 300.0         # manual slit width (centered on axis)
    slit_margin_um: float = 10.0         # extra opening on each side in auto mode
    z_det: float = 2.0                   # Princeton CCD
    ccd_width_mm: float = 27.6           # chip width along the split axis (e.g. 2048 x 13.5 µm)
    ccd_offset_mm: float = 0.0           # lateral offset of the chip centre from the beam axis
    ccd_follow_sd: bool = True           # True: chip moved sideways to centre the SD beam (offset ignored)
    # beamstop in front of the CCD that blocks the direct (transmitted) 1st orders
    dbs_after_sample_mm: float = 50.0    # its distance downstream of the sample
    dbs_auto: bool = True                # size it to just cover the direct beams
    dbs_width_um: float = 500.0          # manual width (centred on axis)
    dbs_margin_um: float = 10.0          # extra width on each side in auto mode

    # numerics
    n_rays: int = 8000
    seed: int = 1

    def copy(self, **kw) -> "Params":
        return dataclasses.replace(self, **kw)


PRESETS = {
    # bandwidth and pulse energy are placeholders: put in the PAL-XFEL values for your run
    # beamstop / grating values are starting points found with the correlation scan
    "pink": dict(beam_fwhm_x_mm=4.0, beam_fwhm_y_mm=4.0, bandwidth_pct=0.5, pulse_energy_uJ=100.0,
                 z_bs=1.0, bs_width_mm=3.0, z_grating=1.80, period_nm=108.6, tg_set_nm=50.0,
                 dbs_after_sample_mm=50.0),
    "mono": dict(beam_fwhm_x_mm=1.0, beam_fwhm_y_mm=1.0, bandwidth_pct=0.02, pulse_energy_uJ=10.0,
                 z_bs=1.0, bs_width_mm=0.75, z_grating=1.45, period_nm=102.0, tg_set_nm=50.0,
                 dbs_after_sample_mm=100.0),
}


def preset(name: str, **kw) -> Params:
    p = Params(**PRESETS[name])
    return p.copy(**kw) if kw else p


# ─────────────────────────────────────────────────────────────────────────────
#  Transient-grating period target
# ─────────────────────────────────────────────────────────────────────────────
def tg_in_range(p: Params, tg_nm: float) -> bool:
    lo, hi = sorted((p.tg_min_nm, p.tg_max_nm))
    t = p.tg_tol_pct / 100
    return bool(np.isfinite(tg_nm) and lo * (1 - t) <= tg_nm <= hi * (1 + t))


def tg_range_text(p: Params) -> str:
    lo, hi = sorted((p.tg_min_nm, p.tg_max_nm))
    return f"{lo:g} nm (±{p.tg_tol_pct:g} %)" if math.isclose(lo, hi) else f"{lo:g}–{hi:g} nm"


def xray_atten_length_nm(formula: str, density: float, wavelength_nm: float) -> float:
    """1/e intensity attenuation length from the Henke tables (the data behind CXRO), via the
    `periodictable` package (pip install periodictable). Raises ImportError if it is missing."""
    import periodictable.xsf as xsf
    E_keV = 1.239842 / wavelength_nm
    n = xsf.index_of_refraction(formula, density=density, energy=E_keV)
    beta = -float(np.imag(n))
    return wavelength_nm / (4 * math.pi * beta)


def depth_factor(dk, La, d):
    """Phase-matching factor of a TG whose amplitude decays as exp(-z/La) over the thickness d:
        η_rel = |∫₀ᵈ e^{-z/La} e^{iΔk z} dz|² / |∫₀ᵈ e^{-z/La} dz|²
    d ≪ La: uniform slab, sinc²(Δk d/2).   d ≫ La: Lorentzian 1/(1 + (Δk La)²).
    Returns (η_rel, effective depth L_eff = ∫ e^{-z/La} dz)."""
    dk = np.asarray(dk, float)
    a = 1.0 / La if La > 0 else 1e30
    if a * d < 1e-6:                                   # no absorption: uniform slab
        return np.sinc(dk * d / 2 / math.pi) ** 2, d
    F0 = (1 - math.exp(-a * d)) / a
    F = (1 - np.exp((-a + 1j * dk) * d)) / (a - 1j * dk)
    return np.abs(F) ** 2 / F0 ** 2, F0


def probe_curve(p: Params, Kx, Kz, phi_deg):
    """Optical probe diffraction from the X-ray TG, transmission geometry, probe in the split plane.
    Grating vector (Kx along the surface, Kz along the normal); the in-plane wavevector is conserved
    exactly, the mismatch along the normal Δk_z = k_pz + m K_z − k_dz sets the efficiency through
    depth_factor() (TG amplitude ∝ exp(-z / X-ray attenuation length) over the sample thickness).
    Returns dict order -> (eta_rel, external diffraction angle in deg, nan if evanescent/trapped)."""
    phi = np.radians(np.asarray(phi_deg, float))
    k0 = 2 * math.pi / (p.probe_wavelength_nm * 1e-9)
    k = p.sample_n * k0
    kpx = k0 * np.sin(phi)
    kpz = np.sqrt(k ** 2 - kpx ** 2)
    out = {}
    for mo in (-1, +1):
        kdx = kpx + mo * Kx
        prop = np.abs(kdx) < k
        kdz = np.sqrt(np.clip(k ** 2 - kdx ** 2, 0, None))
        dk = kpz + mo * Kz - kdz
        f, _ = depth_factor(dk, p.xray_atten_nm * 1e-9, p.sample_thickness_nm * 1e-9)
        eta = np.where(prop, f, 0.0)
        ang = np.where(np.abs(kdx) < k0, np.degrees(np.arcsin(np.clip(kdx / k0, -1, 1))), np.nan)
        out[mo] = (eta, ang)
    return out


def probe_metrics(p: Params, Kx, Kz):
    lam_p = p.probe_wavelength_nm * 1e-9
    Ls = 2 * math.pi / abs(Kx)
    ok = lam_p / (2 * Ls) < 1
    bragg = math.degrees(math.asin(min(lam_p / (2 * Ls), 1.0)))
    # the order that is phase-matched for a positive incidence angle is m = -sign(Kx)
    mo = -1 if Kx > 0 else +1
    phi = bragg if p.probe_auto_bragg else p.probe_angle_deg
    cur = probe_curve(p, Kx, Kz, [phi])
    eta = max(float(cur[-1][0][0]), float(cur[1][0][0]))
    best = -1 if cur[-1][0][0] >= cur[1][0][0] else 1
    # angular acceptance: FWHM of the phase-matching curve around the Bragg angle
    ph = np.linspace(-89.9, 89.9, 17981)
    e = probe_curve(p, Kx, Kz, ph)[mo][0]
    i0 = int(np.argmin(np.abs(ph - bragg)))
    acc = float("nan")
    if e[i0] > 0:
        lo, hi = i0, i0
        while lo > 0 and e[lo - 1] >= 0.5 * e[i0]:
            lo -= 1
        while hi < len(e) - 1 and e[hi + 1] >= 0.5 * e[i0]:
            hi += 1
        acc = float(ph[hi] - ph[lo])         # ~180: thin (Raman-Nath) grating, no angular selectivity
    _, L_eff = depth_factor(0.0, p.xray_atten_nm * 1e-9, p.sample_thickness_nm * 1e-9)
    # Klein-Cook parameter: Q = 2π λ L / (n Λ²);  Q < 1 thin (Raman-Nath), Q > 10 thick (Bragg)
    Q = 2 * math.pi * lam_p * L_eff / (p.sample_n * Ls ** 2)
    return dict(probe_ok=float(ok), probe_bragg_deg=bragg if ok else float("nan"), probe_angle_used_deg=phi,
                probe_eta_rel=eta, probe_order=best, probe_diff_angle_deg=float(cur[best][1][0]),
                probe_accept_deg=acc, probe_L_nm=L_eff * 1e9, probe_Q=Q,
                probe_regime="thin (Raman–Nath)" if Q < 1 else ("thick (Bragg)" if Q > 10 else "intermediate"))


def _mean_incidence(p: Params, rays=None):
    """Mean incidence angles (rad) of the half-beams A and B on the grating, split axis."""
    R = rays if rays is not None else Rays(p, 4000)
    keep = R.pass_mask()
    xbs = R.x_at_bs()
    sl = (R.xis - R.us) / R.fs
    iA, iB = keep & (xbs > 0), keep & (xbs < 0)
    if iA.sum() < 5 or iB.sum() < 5:
        return float("nan"), float("nan")
    return float(np.arctan(sl[iA]).mean()), float(np.arctan(sl[iB]).mean())


def tg_from_period(p: Params, period_nm: float, rays=None) -> float:
    """TG period (nm) that a grating period gives in crossing geometry (exact grating equation)."""
    aA, aB = _mean_incidence(p, rays)
    s = p.wavelength_nm / period_nm
    th = math.asin(min(math.sin(aB) + s, 1)) - math.asin(max(math.sin(aA) - s, -1))
    return p.wavelength_nm / (2 * math.sin(th / 2)) if th > 0 else float("inf")


def natural_tg_nm(p: Params, rays=None) -> float:
    """TG period of A and B with no grating at all (just the KB convergence): the largest Λ possible."""
    aA, aB = _mean_incidence(p, rays)
    th = aB - aA
    return p.wavelength_nm / (2 * math.sin(th / 2)) if th > 0 else float("inf")


def solve_period(p: Params, tg_nm: float, rays=None) -> float:
    """Grating period (nm) that makes A(-1) and B(+1) cross with TG period tg_nm.
    sin(beta_A) = sin(alpha_A) - lambda/p, sin(beta_B) = sin(alpha_B) + lambda/p,
    beta_B - beta_A = theta = 2 asin(lambda / 2 Lambda).  Independent of the grating z.
    Returns nan if tg_nm is not reachable (larger than the natural KB value, or < lambda/2)."""
    lam = p.wavelength_nm
    if not tg_nm > lam / 2:
        return float("nan")
    aA, aB = _mean_incidence(p, rays)
    if not np.isfinite(aA):
        return float("nan")
    th_t = 2 * math.asin(lam / (2 * tg_nm))
    f = lambda s: math.asin(min(math.sin(aB) + s, 1)) - math.asin(max(math.sin(aA) - s, -1)) - th_t
    lo, hi = 0.0, 1.0 - max(abs(math.sin(aA)), abs(math.sin(aB)))
    if f(lo) >= 0 or f(hi) < 0:
        return float("nan")
    for _ in range(60):                           # bisection on s = lambda / p
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return lam / (0.5 * (lo + hi))


# ─────────────────────────────────────────────────────────────────────────────
#  Grating efficiency (Fourier coefficients of a binary grating)
# ─────────────────────────────────────────────────────────────────────────────
def grating_coefficients(p: Params, orders) -> np.ndarray:
    """Complex amplitude coefficients E_n of the grating transmission
    t(x) = sum_n E_n exp(-i n K x).  Bars (fraction a) transmit t_b, gaps t_g:
        E_n = (t_b - t_g) a sinc(a n) + t_g delta_n0      (cf. EnxPhase in the notebook)
    """
    n = np.asarray(orders, dtype=float)
    a = p.duty
    if p.grating_type == "phase":
        t_b = math.sqrt(max(p.bar_T, 0.0)) * np.exp(1j * p.bar_phase_rad)
    else:
        t_b = math.sqrt(max(p.bar_T, 0.0)) + 0j
    t_g = 1.0 + 0j
    E = (t_b - t_g) * a * np.sinc(a * n) + t_g * (n == 0)
    return E * math.sqrt(max(p.membrane_T, 0.0))


def order_efficiencies(p: Params, orders) -> np.ndarray:
    return np.abs(grating_coefficients(p, orders)) ** 2


# ─────────────────────────────────────────────────────────────────────────────
#  Ray bundle machinery
# ─────────────────────────────────────────────────────────────────────────────
def _axis_geo(p: Params, axis: str):
    return (p.z_hm, p.f_h) if axis == "x" else (p.z_vm, p.f_v)


@dataclass
class Bundle:
    name: str            # e.g. "A(-1)"
    beam: str            # "A" | "B" | "full"
    order: int | None    # grating order (None before the grating)
    eff: float           # grating intensity efficiency of this order
    idx: np.ndarray      # ray indices (into the Rays arrays)
    wanted: bool = False
    slit_mask: np.ndarray | None = None   # bool per ray of idx: passes the slit
    color: str = "0.6"


def _kick(slope_in, order, lam, per):
    """Slope change of a ray diffracted into `order` (exact grating equation, not paraxial):
    sin(beta) = sin(alpha) + n lambda / p,  slope = tan."""
    a = np.arctan(slope_in)
    sb = np.clip(np.sin(a) + order * lam / per, -0.999999, 0.999999)
    return np.tan(np.arcsin(sb)) - slope_in


class Rays:
    """All rays + helpers to get positions in the split (s) and other (o) axis.

    Sampling is mirror-symmetric (every ray has a twin with u, xi -> -u, -xi in the split
    axis and the same other-axis coordinates and wavelength), so with a centred beamstop the
    half-beams A and B are exact mirror images: any A/B difference you see is physics
    (beamstop offset), not Monte-Carlo noise.  n_rays is the number of rays that get PAST the
    beamstop; enough rays are generated to reach it (the blocked ones only count for the
    power normalisation), so a wide beamstop no longer leaves the half-beams starved.
    """

    def __init__(self, p: Params, n_rays=None, seed=None):
        self.p = p
        n_pass = int(n_rays or p.n_rays)
        seed = p.seed if seed is None else seed
        s_ax = p.split_axis
        o_ax = "y" if s_ax == "x" else "x"
        self.s_ax, self.o_ax = s_ax, o_ax
        self.zms, self.fs = _axis_geo(p, s_ax)
        self.zmo, self.fo = _axis_geo(p, o_ax)
        self.lam0 = p.wavelength_nm * 1e-9
        # pilot run: fraction of rays that pass the beamstop
        self._draw(20000, seed + 7919)
        frac = max(self.pass_mask().mean(), 1e-4)
        M = int(np.clip(n_pass / frac * 1.05, n_pass, 3_000_000))
        self._draw(M, seed)
        self.N = len(self.us)
        k = min(self.N, 20000)
        self.full_idx = np.arange(self.N) if k == self.N else \
            np.random.default_rng(seed + 1).choice(self.N, k, replace=False)

    def _draw(self, M, seed):
        p = self.p
        rng = np.random.default_rng(seed)
        h = max(M // 2, 1)
        D = {"x": p.beam_fwhm_x_mm * 1e-3, "y": p.beam_fwhm_y_mm * 1e-3}
        F = {"x": p.focus_fwhm_x_um * 1e-6, "y": p.focus_fwhm_y_um * 1e-6}

        def draw_u(d):
            if p.profile == "flattop":
                return rng.uniform(-d / 2, d / 2, h)
            return rng.normal(0.0, d / FWHM_PER_SIGMA, h)

        us, uo = draw_u(D[self.s_ax]), draw_u(D[self.o_ax])
        xis = rng.normal(0.0, F[self.s_ax] / FWHM_PER_SIGMA, h)
        xio = rng.normal(0.0, F[self.o_ax] / FWHM_PER_SIGMA, h)
        lam = self.lam0 * (1.0 + rng.normal(0.0, p.bandwidth_pct / 100 / FWHM_PER_SIGMA, h))
        self.us, self.xis = np.concatenate([us, -us]), np.concatenate([xis, -xis])
        self.uo, self.xio = np.concatenate([uo, uo]), np.concatenate([xio, xio])
        self.lam = np.concatenate([lam, lam])

    def x_at_bs(self):
        return self.s(self.p.z_bs) - self.p.bs_offset_mm * 1e-3

    def pass_mask(self):
        return np.abs(self.x_at_bs()) >= self.p.bs_width_mm * 1e-3 / 2

    # positions --------------------------------------------------------------
    def s(self, z, idx=slice(None), order=0):
        """Split-axis position (m). z scalar or (Z,1) array; returns (N,) or (Z,N)."""
        z = np.asarray(z, dtype=float)
        u, xi = self.us[idx], self.xis[idx]
        t = np.clip((z - self.zms) / self.fs, 0.0, None)
        x = u + (xi - u) * t
        if order:
            zg = self.p.z_grating
            sl = (xi - u) / self.fs if zg > self.zms else 0.0 * u
            x = x + _kick(sl, order, self.lam[idx], self.p.period_nm * 1e-9) * np.clip(z - zg, 0.0, None)
        return x

    def s_slope(self, z, idx=slice(None), order=0):
        u, xi = self.us[idx], self.xis[idx]
        sl = np.where(z > self.zms, (xi - u) / self.fs, 0.0)
        if order and z > self.p.z_grating:
            sl = sl + _kick(sl, order, self.lam[idx], self.p.period_nm * 1e-9)
        return sl

    def o(self, z, idx=slice(None)):
        z = np.asarray(z, dtype=float)
        u, xi = self.uo[idx], self.xio[idx]
        t = np.clip((z - self.zmo) / self.fo, 0.0, None)
        return u + (xi - u) * t


def _fwhm(x):
    return FWHM_PER_SIGMA * float(np.std(x)) if len(x) > 1 else float("nan")


def _extent(x, q=0.01):
    if len(x) < 2:
        return (float("nan"), float("nan"))
    lo, hi = np.quantile(x, [q, 1 - q])
    return float(lo), float(hi)


def _gap(a, b):
    """Clear gap between intervals a=(lo,hi) and b (negative = overlap)."""
    return max(b[0] - a[1], a[0] - b[1])


# ─────────────────────────────────────────────────────────────────────────────
#  Main simulation
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Result:
    p: Params
    rays: Rays
    bundles: list
    z: dict                      # key plane positions
    metrics: dict
    warnings: list = field(default_factory=list)
    slit: tuple = (0.0, 0.0)     # (lo, hi) in m, split axis
    dbs: tuple = (0.0, 0.0)      # (lo, hi) of the direct-beam beamstop, m
    sd: dict = field(default_factory=dict)   # beams on the CCD: name -> (x, y)
    sd_slopes: tuple = (0.0, 0.0)            # mean slopes of A(-1), B(+1) at the sample


def simulate(p: Params, n_rays=None, full=True) -> Result:
    """Trace the beamline. full=False skips the detector/SD bookkeeping (faster scans)."""
    R = Rays(p, n_rays)
    warn = []
    if p.period_mode == "target":
        per_t = solve_period(p, p.tg_set_nm, R)
        if np.isfinite(per_t):
            p = p.copy(period_nm=float(per_t))
            R.p = p
        else:
            warn.append(f"TG period {p.tg_set_nm:g} nm cannot be reached with this beam split (A and B "
                        f"alone already cross at Λ = {natural_tg_nm(p, R):.0f} nm); using the manual "
                        f"grating period {p.period_nm:g} nm")
    zg = p.z_grating
    lam0 = R.lam0
    per = p.period_nm * 1e-9
    if p.z_bs >= zg:
        warn.append("beamstop is downstream of the grating")
    if p.z_bs < max(p.z_hm, p.z_vm):
        warn.append("beamstop is upstream of the second KB mirror")

    # ── split at the beamstop ────────────────────────────────────────────────
    xbs = R.x_at_bs()
    keep = R.pass_mask()
    idxA = np.nonzero(keep & (xbs > 0))[0]
    idxB = np.nonzero(keep & (xbs < 0))[0]

    orders = np.arange(-p.max_order, p.max_order + 1)
    effs = order_efficiencies(p, orders)
    bundles = []
    for beam, idx, wanted_n, col in (("A", idxA, -1, "tab:red"), ("B", idxB, +1, "tab:blue")):
        for n, e in zip(orders, effs):
            if e < 1e-6:
                continue
            bundles.append(Bundle(f"{beam}({n:+d})" if n else f"{beam}(0)", beam, int(n), float(e),
                                  idx, wanted=(n == wanted_n), color=col if n == wanted_n else "0.55"))
    wA = next((b for b in bundles if b.beam == "A" and b.wanted), None)
    wB = next((b for b in bundles if b.beam == "B" and b.wanted), None)
    m = dict(eff_1st=float(order_efficiencies(p, [1])[0]),
             eff_0=float(order_efficiencies(p, [0])[0]),
             frac_after_bs=len(idxA) / R.N + len(idxB) / R.N)
    if wA is None or wB is None or len(idxA) < 10 or len(idxB) < 10:
        warn.append("no usable +-1 orders (grating efficiency 0 or beamstop blocks everything)")
        return Result(p, R, bundles, dict(z_grating=zg), m, warn)

    # ── crossing point of the two first orders ───────────────────────────────
    mA, mB = R.s(zg, wA.idx).mean(), R.s(zg, wB.idx).mean()
    sA = R.s_slope(zg + 1e-9, wA.idx, -1).mean()
    sB = R.s_slope(zg + 1e-9, wB.idx, +1).mean()
    z_cross = zg + (mB - mA) / (sA - sB) if sA != sB else float("inf")
    m["z_cross"] = float(z_cross)
    z_s = z_cross if p.sample_mode == "crossing" else p.z_sample
    if not (zg < z_s):
        warn.append("sample is not downstream of the grating")
        z_s = zg + 1e-3
    if z_s > p.z_det:
        warn.append("1st orders cross AFTER the detector plane")
    z_sl = z_s - p.slit_upstream_mm * 1e-3
    if z_sl <= zg:
        warn.append("slit would sit upstream of the grating; placed just after it")
        z_sl = zg + 0.25 * (z_s - zg)
    z_dbs = z_s + p.dbs_after_sample_mm * 1e-3
    if z_dbs >= p.z_det:
        warn.append("direct-beam beamstop is behind the CCD; placed just in front of it")
        z_dbs = p.z_det - 1e-3
    zpl = dict(hm=p.z_hm, vm=p.z_vm, bs=p.z_bs, grating=zg, slit=float(z_sl), sample=float(z_s),
               dbs=float(z_dbs), det=p.z_det)

    # ── slit ─────────────────────────────────────────────────────────────────
    ext = {id(b): _extent(R.s(z_sl, b.idx, b.order)) for b in bundles}
    wanted_ext = [ext[id(wA)], ext[id(wB)]]
    if p.slit_auto:
        lo = min(e[0] for e in wanted_ext) - p.slit_margin_um * 1e-6
        hi = max(e[1] for e in wanted_ext) + p.slit_margin_um * 1e-6
    else:
        lo, hi = -p.slit_width_um * 1e-6 / 2, p.slit_width_um * 1e-6 / 2
    P_w, P_u = 0.0, 0.0
    sep_min, sep_name, clear_min = float("inf"), "", float("inf")
    for b in bundles:
        xsl = R.s(z_sl, b.idx, b.order)
        b.slit_mask = (xsl >= lo) & (xsl <= hi)
        power = b.eff * b.slit_mask.sum() / R.N
        if b.wanted:
            P_w += power
            continue
        P_u += power
        if b.eff < 1e-3 * wA.eff:
            continue
        for we in wanted_ext:
            g = _gap(we, ext[id(b)])
            if g < sep_min:
                sep_min, sep_name = g, b.name
        e = ext[id(b)]
        if e[1] < lo or e[0] > hi:            # outside the opening -> clearance
            clear_min = min(clear_min, max(lo - e[1], e[0] - hi))
    m.update(slit_lo_um=lo * 1e6, slit_hi_um=hi * 1e6, slit_width_um_used=(hi - lo) * 1e6,
             sep_slit_um=sep_min * 1e6, sep_nearest=sep_name,
             clear_slit_um=(clear_min * 1e6 if np.isfinite(clear_min) else float("nan")),
             contamination_pct=100 * P_u / P_w if P_w > 0 else float("inf"))

    # ── sample ───────────────────────────────────────────────────────────────
    def at(b, z):
        k = b.idx[b.slit_mask]
        return R.s(z, k, b.order), R.o(z, k), k

    xa, ya, ka = at(wA, z_s)
    xb, yb, kb = at(wB, z_s)
    PA = wA.eff * len(ka) / R.N * p.kb_transmission
    PB = wB.eff * len(kb) / R.N * p.kb_transmission
    fa_s, fa_o, fb_s, fb_o = _fwhm(xa), _fwhm(ya), _fwhm(xb), _fwhm(yb)

    def fluence(P, f1, f2):   # Gaussian peak fluence, mJ/cm^2
        area_cm2 = 1.1331 * f1 * f2 * 1e4
        return p.pulse_energy_uJ * 1e-3 * P / area_cm2 if area_cm2 > 0 else float("nan")

    # overlap integral along the split axis
    if len(xa) > 5 and len(xb) > 5:
        lo_, hi_ = min(xa.min(), xb.min()), max(xa.max(), xb.max())
        bins = np.linspace(lo_, hi_, int(np.clip(np.sqrt(min(len(xa), len(xb))), 15, 100)))
        ha, _ = np.histogram(xa, bins)
        hb, _ = np.histogram(xb, bins)
        ov = float((ha * hb).sum() / math.sqrt((ha ** 2).sum() * (hb ** 2).sum()))
    else:
        ov = 0.0
    sA_s = float(R.s_slope(z_s, ka, -1).mean()) if len(ka) else sA
    sB_s = float(R.s_slope(z_s, kb, +1).mean()) if len(kb) else sB
    theta = abs(math.atan(sA_s) - math.atan(sB_s))
    tg_period = lam0 / (2 * math.sin(theta / 2)) if theta > 0 else float("inf")
    # pulse-front tilt across one first-order beam: N_lines * lambda / c
    w_gr = _fwhm(R.s(zg, wA.idx))
    pft_fs = w_gr / per * lam0 / C_LIGHT * 1e15
    m.update(z_sample=z_s, z_slit=z_sl, d_grating_sample_mm=(z_s - zg) * 1e3,
             d_sample_det_mm=(p.z_det - z_s) * 1e3,
             P_A=PA, P_B=PB, P_1st_total=PA + PB,
             fwhm_A_s_um=fa_s * 1e6, fwhm_A_o_um=fa_o * 1e6,
             fwhm_B_s_um=fb_s * 1e6, fwhm_B_o_um=fb_o * 1e6,
             centroid_offset_um=(xa.mean() - xb.mean()) * 1e6 if len(xa) and len(xb) else float("nan"),
             overlap=ov, fluence_A=fluence(PA, fa_s, fa_o), fluence_B=fluence(PB, fb_s, fb_o),
             crossing_angle_mrad=theta * 1e3, tg_period_nm=tg_period * 1e9,
             n_fringes=min(fa_s, fb_s) / tg_period if tg_period > 0 else 0.0,
             beam_at_grating_um=w_gr * 1e6, pft_fs=pft_fs,
             order_angle_mrad=math.asin(min(lam0 / per, 1.0)) * 1e3, grating_period_nm=p.period_nm,
             tg_in_range=float(tg_in_range(p, tg_period * 1e9)))
    # beam angles with respect to the sample normal (split plane; + = towards +split axis)
    tilt = p.sample_tilt_mrad * 1e-3
    bA, bB = math.atan(sA_s) - tilt, math.atan(sB_s) - tilt
    k_x = 2 * math.pi / lam0
    Kx, Kz = k_x * (math.sin(bB) - math.sin(bA)), k_x * (math.cos(bB) - math.cos(bA))
    m.update(angle_A_mrad=bA * 1e3, angle_B_mrad=bB * 1e3, angle_A_deg=math.degrees(bA),
             angle_B_deg=math.degrees(bB), bisector_mrad=(bA + bB) / 2 * 1e3,
             tg_surface_nm=2 * math.pi / abs(Kx) * 1e9 if Kx else float("inf"),
             tg_tilt_mrad=math.atan2(Kz, Kx) * 1e3)
    if p.experiment == "optical":
        m.update(probe_metrics(p, Kx, Kz))
        if not m["probe_ok"]:
            warn.append(f"TG period {tg_period * 1e9:.0f} nm < λ_probe/2 = {p.probe_wavelength_nm / 2:g} nm: "
                        "the optical probe has no diffracted order")
    if not m["tg_in_range"]:
        warn.append(f"TG period {tg_period * 1e9:.1f} nm is outside the target range "
                    f"{tg_range_text(p)}: setting discarded")
    m["fluence_sum"] = m["fluence_A"] + m["fluence_B"]
    # simple relative figure of merit for a 3rd-order (SD) signal
    m["sd_fom"] = ov * m["fluence_A"] * m["fluence_B"] * m["fluence_sum"] / 2

    # ── overlap region ("rhombus") of the two first orders ───────────────────
    # z range in which the 2-98 % bands of A(-1) and B(+1) intersect; the TG only exists there
    rng_ = np.random.default_rng(3)
    pick = lambda k: k if len(k) <= 1500 else rng_.choice(k, 1500, replace=False)
    zc = min(max(m["z_cross"], zg), p.z_det)
    kA, kB = pick(wA.idx), pick(wB.idx)

    def band(zz):
        aL, aH = np.quantile(R.s(zz, kA, -1), [0.02, 0.98], axis=1)
        bL, bH = np.quantile(R.s(zz, kB, +1), [0.02, 0.98], axis=1)
        return np.minimum(aH, bH) - np.maximum(aL, bL)       # > 0 where they overlap

    # the rhombus is ~ (beam size / crossing angle) long: mm for small TG periods, so search a
    # finely sampled window around the crossing first and fall back to the whole range
    half = 4 * (fa_s + fb_s) / max(theta, 1e-9)
    zz = np.linspace(max(zg, zc - half), min(p.z_det, zc + half), 161)[:, None]
    inter = band(zz)
    if (inter[0] > 0 and zz[0, 0] > zg) or (inter[-1] > 0 and zz[-1, 0] < p.z_det):
        zz = np.linspace(zg, p.z_det, 401)[:, None]
        inter = band(zz)
    ic = int(np.argmin(np.abs(zz[:, 0] - zc)))
    if inter[ic] > 0:
        i0, i1 = ic, ic
        while i0 > 0 and inter[i0 - 1] > 0:
            i0 -= 1
        while i1 < len(inter) - 1 and inter[i1 + 1] > 0:
            i1 += 1
        # refine the two ends by linear interpolation of the intersection width
        def cross(i, j):
            return float(zz[i, 0] + (zz[j, 0] - zz[i, 0]) * inter[i] / (inter[i] - inter[j]))
        z_ov0 = cross(i0, i0 - 1) if i0 > 0 else float(zz[0, 0])
        z_ov1 = cross(i1, i1 + 1) if i1 < len(inter) - 1 else float(zz[-1, 0])
    else:
        z_ov0 = z_ov1 = float("nan")
    margin = min(z_s - z_ov0, z_ov1 - z_s) if np.isfinite(z_ov0) else float("-inf")
    m.update(z_ov0=z_ov0, z_ov1=z_ov1, ov_len_mm=(z_ov1 - z_ov0) * 1e3 if np.isfinite(z_ov0) else 0.0,
             sample_ov_margin_mm=margin * 1e3, slit_in_ov=bool(np.isfinite(z_ov0) and z_sl >= z_ov0))
    if not margin > 0:
        warn.append("sample is OUTSIDE the region where A(-1) and B(+1) overlap: no transient grating")
        m["overlap"] = 0.0 if not np.isfinite(z_ov0) else m["overlap"]
    m["sd_fom"] = m["sd_fom"] if margin > 0 else 0.0

    # ── TG fringe contrast at the sample ─────────────────────────────────────
    # A(-1) and B(+1) rays that meet at the same sample point left the KB at different
    # positions u_A, u_B; their mutual coherence (Gaussian Schell model) is
    # exp(-(u_A-u_B)^2 / 2 l_c^2).  The modulation depth of the TG is
    #     sum 2 sqrt(I_A I_B) V / sum (I_A + I_B)   over the sample.
    # Bandwidth does NOT wash the fringes out: in this geometry the TG is the converging-beam
    # projection of the grating's +-1 interference, Lambda = (p/2) R_sample/R_grating for every
    # wavelength (checked with the wave-optics propagator in sdxtg_wave.py).
    D_s = (p.beam_fwhm_x_mm if p.split_axis == "x" else p.beam_fwhm_y_mm) * 1e-3
    l_c = p.coh_ratio * D_s / FWHM_PER_SIGMA
    if len(xa) > 20 and len(xb) > 20:
        edges = np.linspace(min(xa.min(), xb.min()), max(xa.max(), xb.max()), 41)
        cx = 0.5 * (edges[1:] + edges[:-1])
        IA, _ = np.histogram(xa, edges)
        IB, _ = np.histogram(xb, edges)
        uA, _ = np.histogram(xa, edges, weights=R.us[ka])
        uB, _ = np.histogram(xb, edges, weights=R.us[kb])
        both = (IA > 0) & (IB > 0)
        du = np.where(both, uA / np.maximum(IA, 1) - uB / np.maximum(IB, 1), 0.0)
        V_sp = np.exp(-du ** 2 / (2 * l_c ** 2))
        IA, IB = IA * wA.eff, IB * wB.eff
        num = (2 * np.sqrt(IA * IB) * V_sp).sum()
        den = (IA + IB).sum()
        w = np.sqrt(IA * IB)
        m.update(tg_contrast=float(num / den) if den > 0 else 0.0,
                 coherence_AB=float((w * V_sp).sum() / w.sum()) if w.sum() > 0 else 0.0,
                 du_KB_mm=float((w * np.abs(du)).sum() / w.sum() * 1e3) if w.sum() > 0 else float("nan"),
                 l_coh_KB_mm=l_c * 1e3)
    else:
        m.update(tg_contrast=0.0, coherence_AB=0.0, du_KB_mm=float("nan"),
                 l_coh_KB_mm=l_c * 1e3)
    if margin <= 0:
        m["tg_contrast"] = 0.0
    # relative SD figure of merit: diffraction efficiency ~ (modulation depth)^2, times F_A F_B (F_A+F_B)/2
    m["sd_fom"] = m["tg_contrast"] ** 2 * m["fluence_A"] * m["fluence_B"] * m["fluence_sum"] / 2

    # optical probe: diffracted probe ~ (TG modulation x pump fluence)^2 x phase matching
    m["opt_fom"] = ((m["tg_contrast"] * m["fluence_sum"]) ** 2 * m.get("probe_eta_rel", 0.0)
                    * m.get("probe_ok", 0.0)) if p.experiment == "optical" else float("nan")
    res = Result(p, R, bundles, zpl, m, warn, slit=(lo, hi))
    if full:
        _detector(res, wA, wB, sA_s, sB_s)
    return res


def _detector(res: Result, wA, wB, sA, sB):
    """Direct 1st orders and self-diffracted beams (2kA-kB, 2kB-kA): checked at the
    direct-beam beamstop (z = sample + dbs_after_sample_mm) and recorded on the CCD."""
    p, R, z_s, zb, zd = res.p, res.rays, res.z["sample"], res.z["dbs"], res.z["det"]
    res.sd_slopes = (sA, sB)
    at_bs, out = {}, {}
    for b, extra, nm in ((wA, sA - sB, "SD_A"), (wB, sB - sA, "SD_B")):
        k = b.idx[b.slit_mask]
        xb, xd = R.s(zb, k, b.order), R.s(zd, k, b.order)
        at_bs[b.name], at_bs[nm] = xb, xb + extra * (zb - z_s)
        out[b.name] = (xd, R.o(zd, k))
        out[nm] = (xd + extra * (zd - z_s), R.o(zd, k))
    dA, dB = _extent(at_bs[wA.name]), _extent(at_bs[wB.name])
    sdA, sdB = _extent(at_bs["SD_A"]), _extent(at_bs["SD_B"])
    if p.dbs_auto:
        mg = p.dbs_margin_um * 1e-6
        bs = (min(dA[0], dB[0]) - mg, max(dA[1], dB[1]) + mg)
    else:
        bs = (-p.dbs_width_um * 1e-6 / 2, p.dbs_width_um * 1e-6 / 2)
    res.dbs = bs
    inside = lambda x: ((x >= bs[0]) & (x <= bs[1])).mean() if len(x) else 0.0
    direct = np.concatenate([at_bs[wA.name], at_bs[wB.name]])
    sd = np.concatenate([at_bs["SD_A"], at_bs["SD_B"]])
    # unwanted orders that leaked through the slit: do they get past the beamstop?
    leak_P, leak_hit = 0.0, 0.0
    for b in res.bundles:
        if b.wanted:
            continue
        k = b.idx[b.slit_mask]
        if len(k):
            x = R.s(zb, k, b.order)
            leak_P += b.eff * len(k)
            leak_hit += b.eff * ((x < bs[0]) | (x > bs[1])).sum()
    res.sd = out
    m = res.metrics
    m.update(z_dbs=zb, dbs_lo_um=bs[0] * 1e6, dbs_hi_um=bs[1] * 1e6, dbs_width_um_used=(bs[1] - bs[0]) * 1e6,
             sd_clear_dbs_um=min(_gap(bs, sdA), _gap(bs, sdB)) * 1e6,
             sd_direct_sep_dbs_um=min(_gap(dA, sdA), _gap(dB, sdB)) * 1e6,
             direct_leak_pct=100 * (1 - inside(direct)), sd_blocked_pct=100 * inside(sd),
             leak_to_ccd_pct=100 * leak_hit / R.N / max(m["P_1st_total"], 1e-30),
             det_direct_A_um=float(np.mean(out[wA.name][0]) * 1e6),
             det_direct_B_um=float(np.mean(out[wB.name][0]) * 1e6),
             det_SD_A_um=float(np.mean(out["SD_A"][0]) * 1e6),
             det_SD_B_um=float(np.mean(out["SD_B"][0]) * 1e6),
             det_fwhm_direct_um=_fwhm(out[wA.name][0]) * 1e6,
             det_fwhm_sd_um=_fwhm(out["SD_A"][0]) * 1e6)
    # is (at least) one self-diffracted beam completely on the CCD chip?
    off = abs(m["det_SD_A_um"]) * 1e-3 * np.sign(m["det_SD_A_um"]) if p.ccd_follow_sd else p.ccd_offset_mm
    m["ccd_offset_used_mm"] = float(off)
    c_lo = (off - p.ccd_width_mm / 2) * 1e-3
    c_hi = (off + p.ccd_width_mm / 2) * 1e-3
    mg = []
    for nm in ("SD_A", "SD_B"):
        e = _extent(out[nm][0])
        mg.append(min(e[0] - c_lo, c_hi - e[1]))
    m["sd_ccd_margin_mm"] = max(mg) * 1e3


# ─────────────────────────────────────────────────────────────────────────────
#  Beam-size tracking along z
# ─────────────────────────────────────────────────────────────────────────────
def sd_envelope(res: Result, nz=60, q=0.02):
    """Envelopes of the self-diffracted beams from the sample to the CCD."""
    R, z_s, zd = res.rays, res.z["sample"], res.z["det"]
    wA = next(b for b in res.bundles if b.beam == "A" and b.wanted)
    wB = next(b for b in res.bundles if b.beam == "B" and b.wanted)
    kA, kB = wA.idx[wA.slit_mask], wB.idx[wB.slit_mask]
    sA, sB = res.sd_slopes
    Z = np.linspace(z_s, zd, nz)[:, None]
    out = []
    for b, k, extra in ((wA, kA, sA - sB), (wB, kB, sB - sA)):
        x = R.s(Z, k, b.order) + extra * (Z - z_s)
        lo, hi = np.quantile(x, [q, 1 - q], axis=1)
        out.append(("SD_" + b.beam, Z[:, 0], lo, hi))
    return out


def overlap_region(res: Result, nz=120, q=0.02):
    """(z, lo, hi) of the region where the q..1-q bands of A(-1) and B(+1) intersect: the
    'rhombus' in which a transient grating exists.  Empty arrays if they never overlap."""
    R, p = res.rays, res.p
    wA = next(b for b in res.bundles if b.beam == "A" and b.wanted)
    wB = next(b for b in res.bundles if b.beam == "B" and b.wanted)
    z0, z1 = res.metrics.get("z_ov0", np.nan), res.metrics.get("z_ov1", np.nan)
    if not np.isfinite(z0):
        return np.array([]), np.array([]), np.array([])
    Z = np.linspace(z0, z1, nz)[:, None]
    kA, kB = wA.idx[wA.slit_mask], wB.idx[wB.slit_mask]
    aL, aH = np.quantile(R.s(Z, kA, -1), [q, 1 - q], axis=1)
    bL, bH = np.quantile(R.s(Z, kB, +1), [q, 1 - q], axis=1)
    lo, hi = np.maximum(aL, bL), np.minimum(aH, bH)
    ok = hi >= lo
    return Z[ok, 0], lo[ok], hi[ok]


def size_table(res: Result):
    """FWHM (um, 2.355*rms) of each beam component at every key plane."""
    R, z = res.rays, res.z
    wA = next(b for b in res.bundles if b.beam == "A" and b.wanted)
    wB = next(b for b in res.bundles if b.beam == "B" and b.wanted)
    rows = []
    for name in ("hm", "vm", "bs", "grating", "slit", "sample", "dbs", "det"):
        zz = z[name]
        row = dict(plane=name, z_m=round(zz, 5),
                   full_s=_fwhm(R.s(zz, R.full_idx)) * 1e6, full_o=_fwhm(R.o(zz, R.full_idx)) * 1e6)
        if zz >= res.p.z_bs:
            for b in (wA, wB):
                k = b.idx if zz < z["slit"] else b.idx[b.slit_mask]
                order = b.order if zz > res.p.z_grating else 0
                row[f"{b.beam}_s"] = _fwhm(R.s(zz, k, order)) * 1e6
                row[f"{b.beam}_o"] = _fwhm(R.o(zz, k)) * 1e6
                row[f"{b.beam}_centroid"] = float(np.mean(R.s(zz, k, order))) * 1e6
        rows.append(row)
    return rows


def size_vs_z(res: Result, nz=300):
    """Continuous FWHM(z) curves (um) for the full beam and the two wanted beams."""
    R, p = res.rays, res.p
    zz = np.linspace(min(p.z_hm, p.z_vm), res.z["det"], nz)
    Z = zz[:, None]
    out = dict(z=zz, full_s=FWHM_PER_SIGMA * R.s(Z, R.full_idx).std(1) * 1e6,
               full_o=FWHM_PER_SIGMA * R.o(Z, R.full_idx).std(1) * 1e6)
    for b in res.bundles:
        if not b.wanted:
            continue
        k = b.idx[b.slit_mask]
        xs = np.where(Z > p.z_grating, R.s(Z, k, b.order), R.s(Z, k))
        f = FWHM_PER_SIGMA * xs.std(1) * 1e6
        f[zz < p.z_bs] = np.nan
        out[b.beam + "_s"] = f
        out[b.beam + "_o"] = np.where(zz < p.z_bs, np.nan, FWHM_PER_SIGMA * R.o(Z, k).std(1) * 1e6)
    return out


def envelopes(res: Result, nz=160, q=0.02, max_rays=6000):
    """(name, tag, z, lo, hi, wanted) envelopes in the split axis for the sketch.

    Each half-beam is drawn between its own q / 1-q quantile on the outside and its beamstop
    shadow (0.2 % quantile) on the inside, from the KB all the way to the CCD, using the SAME
    rays in every segment, so its outline is continuous through the beamstop and the grating.
    Upstream of the beamstop the green full beam is drawn so that its edges coincide with the
    outer edges of A and B, and the parts of it that will become A and B are tinted.
    """
    R, p, zp = res.rays, res.p, res.z
    rng = np.random.default_rng(0)
    sub = lambda idx: idx if len(idx) <= max_rays else np.sort(rng.choice(idx, max_rays, replace=False))
    env = []
    z0 = np.linspace(min(p.z_hm, p.z_vm) - 0.05, p.z_bs, nz // 3)[:, None]
    flo, fhi = np.quantile(R.s(z0, sub(R.full_idx)), [q, 1 - q], axis=1)
    qi = 0.002                                       # inner edge = beamstop shadow
    levels, rays = {}, {}
    for beam in ("A", "B"):
        b0 = next((b for b in res.bundles if b.beam == beam), None)
        if b0 is None or len(b0.idx) < 5:
            continue
        levels[beam] = [qi, 1 - q] if beam == "A" else [q, 1 - qi]
        rays[beam] = sub(b0.idx)
    pre = {}
    for beam in levels:
        pre[beam] = np.quantile(R.s(z0, rays[beam]), levels[beam], axis=1)
    if "A" in pre:
        fhi = np.maximum(fhi, pre["A"][1])
    if "B" in pre:
        flo = np.minimum(flo, pre["B"][0])
    env.append(("full", "full", z0[:, 0], flo, fhi, True))
    for beam in pre:
        env.append((beam + " (upstream)", beam, z0[:, 0], pre[beam][0], pre[beam][1], False))
    zsplit = np.linspace(p.z_bs, p.z_grating, nz // 3)[:, None]
    for beam in levels:
        lo, hi = np.quantile(R.s(zsplit, rays[beam]), levels[beam], axis=1)
        env.append((beam, beam, zsplit[:, 0], lo, hi, True))
    if "slit" not in zp:
        return env
    z1 = np.linspace(p.z_grating, zp["slit"], nz // 3)[:, None]
    z2 = np.linspace(zp["slit"], zp["dbs"], nz // 3)[:, None]
    z3 = np.linspace(zp["dbs"], zp["det"], nz // 4)[:, None]
    for b in res.bundles:
        if b.beam not in levels:
            continue
        lv = levels[b.beam]
        tag = b.beam if b.wanted else "other"
        k = rays[b.beam]
        lo, hi = np.quantile(R.s(z1, k, b.order), lv, axis=1)
        env.append((b.name, tag, z1[:, 0], lo, hi, b.wanted))
        kp = sub(b.idx[b.slit_mask])
        if len(kp) > 20:
            lv2 = lv if b.wanted else [q, 1 - q]
            lo, hi = np.quantile(R.s(z2, kp, b.order), lv2, axis=1)
            env.append((b.name, tag, z2[:, 0], lo, hi, b.wanted))
            if b.wanted:                     # stopped by the direct-beam beamstop: draw faint
                lo, hi = np.quantile(R.s(z3, kp, b.order), lv, axis=1)
                env.append((b.name + " (blocked)", tag, z3[:, 0], lo, hi, False))
    return env


# ─────────────────────────────────────────────────────────────────────────────
#  Correlation scans
# ─────────────────────────────────────────────────────────────────────────────
# Every map is one simulation per (grating z, grating period) pixel with ALL other
# parameters held at the values in Params.  In sample_mode="crossing" the sample is moved,
# for every pixel, to the crossing point of the A(-1) and B(+1) centroids; the slit stays
# slit_upstream_mm before it and the direct-beam beamstop dbs_after_sample_mm after it.
# In sample_mode="fixed" the sample stays at z_sample and the overlap map shows where the
# orders actually meet it.
SCAN_INFO = {
    # key: (short title, unit, precise definition)
    "d_grating_sample_mm": ("Grating → sample distance", "mm",
        "Distance from the grating to the sample. In crossing mode this is where the centroids of "
        "A(−1) and B(+1) meet."),
    "tg_period_nm": ("Transient-grating period Λ", "nm",
        "Λ = λ / (2 sin(θ/2)), θ = angle between the mean directions of A(−1) and B(+1) at the sample."),
    "overlap": ("Spatial overlap at sample", "",
        "∫I_A I_B dx / √(∫I_A² ∫I_B²) of the A(−1) and B(+1) profiles along the split axis at the "
        "sample (1 = identical and on top of each other; ≈0.8 is the best possible for two mirror-image half-beams)."),
    "fluence_sum": ("Peak fluence on sample, A(−1)+B(+1)", "mJ/cm²",
        "Sum of the Gaussian-equivalent peak fluences E·P/(1.133·FWHM_x·FWHM_y) of the two 1st orders "
        "after the slit, for the pulse energy and KB transmission set."),
    "P_1st_total": ("Photons in the two 1st orders", "% of KB input",
        "Fraction of the photons on the KB that reach the sample in A(−1)+B(+1): beamstop transmission "
        "× grating 1st-order efficiency × slit transmission."),
    "sep_slit_um": ("Order separation at slit", "µm",
        "Smallest edge-to-edge gap, at the slit plane, between a wanted order and the nearest unwanted order "
        "(edges = 1 % / 99 % quantiles; orders with < 0.1 % of the 1st-order efficiency ignored). "
        "> 0: the slit can cut the unwanted orders away."),
    "contamination_pct": ("Unwanted light through slit", "% of 1st orders",
        "Power of all unwanted orders (0th, ±3rd, the outward ±1st …) that pass the slit opening, relative "
        "to the wanted A(−1)+B(+1) power."),
    "fwhm_A_s_um": ("1st-order beam size at sample", "µm FWHM",
        "FWHM (2.355·rms) of A(−1) along the split axis at the sample (B(+1) is its mirror image)."),
    "sd_clear_dbs_um": ("SD clearance at direct-beam beamstop", "µm",
        "Gap between the edge of the direct-beam beamstop and the nearest self-diffracted beam (2k_A−k_B, "
        "2k_B−k_A) at the beamstop plane. > 0: the SD signal passes the beamstop and reaches the CCD."),
    "sd_fom": ("SD figure of merit", "rel.",
        "(TG contrast)² × F_A × F_B × (F_A+F_B)/2, a relative measure of a 3rd-order self-diffraction signal."),
    "pft_fs": ("Pulse-front tilt", "fs",
        "Delay across one 1st-order beam introduced by the grating: (illuminated lines) × λ / c."),
    "z_sample": ("Sample position", "m", "Sample z measured from the horizontal KB mirror."),
    "tg_contrast": ("TG fringe contrast (modulation depth)", "",
        "Σ 2√(I_A I_B)·V / Σ(I_A+I_B) over the sample, V = mutual coherence of the A and B rays that meet "
        "(Gaussian-Schell coherence length on the KB = coh_ratio × rms beam size). Bandwidth does not reduce it: "
        "the TG period (p/2)·R_sample/R_grating is the same for every wavelength. "
        "1 = full-contrast grating; it drops when A and B come from far-apart parts of the beam."),
    "sample_ov_margin_mm": ("Sample inside the A/B overlap region", "mm",
        "Distance from the sample to the nearest end of the region (the 'rhombus') where the 2–98 % bands of "
        "A(−1) and B(+1) intersect. > 0: the sample sees both beams, so a TG exists; ≤ 0: no TG."),
    "ov_len_mm": ("Length of the A/B overlap region", "mm",
        "Length along z of the rhombus where A(−1) and B(+1) overlap (tolerance on the sample z position)."),
    "grating_period_nm": ("Grating period used", "nm",
        "Grating period of this point. When the scan runs over the TG period (or the beamstop width at a fixed "
        "TG period) it is solved with the exact grating equation so that A(−1) and B(+1) cross at the target Λ."),
    "sd_ccd_margin_mm": ("SD beam on the CCD chip", "mm",
        "Distance from the edge of the better-placed self-diffracted beam (1–99 % extent) to the nearest edge "
        "of the CCD chip (width and lateral offset set in the sidebar). > 0: that SD beam lands completely "
        "on the chip; small Λ means large angles, so the SD beams can miss a CCD that is far away."),
    "angle_A_mrad": ("Angle of A(−1) to the sample normal", "mrad",
        "Angle between the mean direction of A(−1) at the sample and the sample normal (B(+1) is at the "
        "mirror angle for an untilted sample). Crossing angle θ = |angle_B − angle_A|, Λ = λ/(2 sin(θ/2))."),
    "probe_bragg_deg": ("Optical probe Bragg angle", "deg",
        "External incidence angle (from the sample normal) at which the optical probe is phase matched to the "
        "TG: sin φ_B = λ_probe / (2 Λ_surface). Blank where Λ < λ_probe/2 (no diffracted probe order)."),
    "probe_eta_rel": ("Probe phase-matching factor", "",
        "|∫ g(z) e^{iΔk_z z} dz|² / |∫ g dz|² at the probe angle used (1 at the Bragg angle), g(z) = e^{−z/L_att} over the "
        "sample thickness: sinc² for a uniform slab, Lorentzian when the X-ray attenuation length is the shorter."),
    "opt_fom": ("Optical-probe figure of merit", "rel.",
        "(TG contrast × pump fluence A+B)² × phase-matching factor: relative diffracted probe signal."),
}
SCAN_KEYS = list(SCAN_INFO)
SCAN_LABELS = {k: f"{v[0]} [{v[1]}]" if v[1] else v[0] for k, v in SCAN_INFO.items()}
DEFAULT_MAPS = ["d_grating_sample_mm", "grating_period_nm", "tg_period_nm", "sample_ov_margin_mm", "overlap",
                "tg_contrast", "fluence_sum", "sep_slit_um", "contamination_pct", "sd_clear_dbs_um",
                "sd_ccd_margin_mm", "sd_fom"]
DEFAULT_MAPS_OPTICAL = ["d_grating_sample_mm", "grating_period_nm", "tg_period_nm", "sample_ov_margin_mm",
                        "overlap", "tg_contrast", "fluence_sum", "sep_slit_um", "contamination_pct",
                        "angle_A_mrad", "probe_bragg_deg", "opt_fom"]


def default_maps(p: Params):
    return DEFAULT_MAPS_OPTICAL if p.experiment == "optical" else DEFAULT_MAPS


def fom_key(p: Params):
    return "opt_fom" if p.experiment == "optical" else "sd_fom"

# second scan axis (the first one is always the grating position)
Y_KINDS = {
    # kind: (axis label, unit, log axis)
    "tg": ("TG period Λ (target)", "nm", True),
    "period": ("grating period", "nm", True),
    "bs_width": ("beamstop width", "mm", False),
}


def scan_point(p: Params, zg: float, y: float, kind: str) -> Params:
    """Parameters of one scan point."""
    if kind == "tg":
        return p.copy(z_grating=float(zg), period_mode="target", tg_set_nm=float(y))
    if kind == "period":
        return p.copy(z_grating=float(zg), period_mode="manual", period_nm=float(y))
    if kind == "bs_width":
        return p.copy(z_grating=float(zg), bs_width_mm=float(y))
    raise ValueError(kind)


def y_current(p: Params, kind: str) -> float:
    return {"tg": p.tg_set_nm, "period": p.period_nm, "bs_width": p.bs_width_mm}[kind]


def _scan_row(args):
    p, z_grating, y, kind, n_rays = args
    row = {k: np.full(len(z_grating), np.nan) for k in SCAN_KEYS}
    row["discarded"] = np.zeros(len(z_grating), bool)
    for i, zg in enumerate(z_grating):
        if zg <= p.z_bs:
            continue
        r = simulate(scan_point(p, zg, y, kind), n_rays=n_rays)
        if any("AFTER" in w or "not downstream" in w or "cannot be reached" in w for w in r.warnings) \
                or "z_sample" not in r.metrics:
            continue
        if not r.metrics.get("tg_in_range", 0.0):          # TG period outside the target range
            row["discarded"][i] = True
            continue
        for k in SCAN_KEYS:
            row[k][i] = r.metrics.get(k, np.nan)
    row["P_1st_total"] = row["P_1st_total"] * 100
    return row


def scan(p: Params, z_grating, y_values, n_rays=3000, progress=None, workers=None, kind="tg"):
    """2-D scan over grating position x a second parameter (kind: see Y_KINDS).
    Returns dict of (n_y, n_zg) maps. Settings whose TG period is outside the target range
    are discarded (NaN, flagged in maps["discarded"]).
    Runs rows in parallel processes (workers=None: all cores, 1: serial)."""
    import os
    z_grating, y_values = np.asarray(z_grating, float), np.asarray(y_values, float)
    jobs = [(p, z_grating, float(y), kind, n_rays) for y in y_values]
    rows = [None] * len(jobs)
    workers = workers or os.cpu_count() or 1
    done = 0
    if workers > 1:
        try:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=workers) as ex:
                fut = {ex.submit(_scan_row, j): n for n, j in enumerate(jobs)}
                for f in as_completed(fut):
                    rows[fut[f]] = f.result()
                    done += 1
                    if progress:
                        progress(done / len(jobs))
        except Exception:                      # no multiprocessing available: serial
            rows = [None] * len(jobs)
            done = 0
    for n, j in enumerate(jobs):
        if rows[n] is None:
            rows[n] = _scan_row(j)
            done += 1
            if progress:
                progress(done / len(jobs))
    maps = {k: np.vstack([r[k] for r in rows]) for k in SCAN_KEYS + ["discarded"]}
    maps["z_grating"], maps["y"], maps["kind"] = z_grating, y_values, kind
    maps["experiment"] = p.experiment
    return maps


def feasible_mask(maps, min_sep_um=20.0, max_cont_pct=1.0, min_overlap=0.5, min_sd_clear_um=20.0):
    optical = maps.get("experiment", "sd") == "optical"
    with np.errstate(invalid="ignore"):
        ok = (maps["sep_slit_um"] > min_sep_um) & (maps["contamination_pct"] < max_cont_pct)
        ok &= maps["overlap"] > min_overlap
        ok &= maps["sample_ov_margin_mm"] > 0          # sample inside the A/B overlap region
        if optical:
            ok &= np.isfinite(maps["probe_bragg_deg"])     # Λ > λ_probe / 2: a diffracted probe exists
        else:
            ok &= maps["sd_clear_dbs_um"] > min_sd_clear_um
            ok &= maps["sd_ccd_margin_mm"] > 0         # one SD beam lands on the CCD chip
        ok &= ~maps["discarded"] & np.isfinite(maps["tg_period_nm"])   # TG period in the target range
    return ok


def plot_scan(maps, p: Params, keys=None, fig=None, title=""):
    """Matplotlib version of the scan maps (used by the CLI)."""
    import textwrap
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, TwoSlopeNorm
    keys = keys or default_maps(p)
    zg, per, kind = maps["z_grating"], maps["y"], maps.get("kind", "period")
    ylab, yunit, ylog = Y_KINDS[kind]
    ncol = 3
    nrow = int(math.ceil(len(keys) / ncol))
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#888",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white"}):
        if fig is None:
            fig = plt.figure(figsize=(18, 6.4 * nrow))
        ok = feasible_mask(maps)
        axes = []
        for i, k in enumerate(keys):
            ax = fig.add_subplot(nrow, ncol, i + 1)
            A = maps[k]
            kw = dict(cmap="viridis")
            fin = A[np.isfinite(A)]
            if k in ("contamination_pct", "sd_fom", "fluence_sum", "opt_fom"):
                pos = fin[fin > 0]
                if pos.size:
                    vmin = max(pos.min(), pos.max() * 1e-4)
                    kw["norm"] = LogNorm(vmin=vmin, vmax=pos.max())
                    A = np.where(A <= 0, vmin, A)          # 0 % contamination -> bottom of the scale
                kw["cmap"] = "magma" if k == "contamination_pct" else "viridis"
            if k in ("sep_slit_um", "sd_clear_dbs_um", "sample_ov_margin_mm", "sd_ccd_margin_mm") and fin.size and fin.min() < 0 < fin.max():
                kw.update(cmap="RdBu", norm=TwoSlopeNorm(0, fin.min(), fin.max()))
            ax.set_facecolor("#D9DDE3")                    # grey = discarded / no valid setting
            im = ax.pcolormesh(zg, per, A, shading="auto", rasterized=True, **kw)
            cb = fig.colorbar(im, ax=ax, pad=0.02)
            cb.set_label(SCAN_INFO[k][1])
            if ok.any():
                ax.contour(zg, per, ok.astype(float), levels=[0.5], colors="white", linewidths=2.2)
                ax.contour(zg, per, ok.astype(float), levels=[0.5], colors="black", linewidths=1.0)
            if p.sample_mode == "fixed" and np.isfinite(maps["overlap"]).any():
                ax.contour(zg, per, np.nan_to_num(maps["overlap"]), levels=[0.6],
                           colors="w", linewidths=0.8, linestyles="--")
            ax.plot([p.z_grating], [y_current(p, kind)], marker="*", ms=13, color="gold", mec="k")
            if ylog:
                ax.set_yscale("log")
            desc = textwrap.wrap(SCAN_INFO[k][2], 88)
            ax.set_title(f"({chr(97 + i)}) {SCAN_INFO[k][0]}", loc="left", fontweight="bold", fontsize=11,
                         pad=8 + 9.5 * len(desc))
            ax.text(0, 1.012, "\n".join(desc), transform=ax.transAxes, fontsize=7.3, color="#4A5563",
                    va="bottom", linespacing=1.25)
            ax.set_xlabel("grating z from HM [m]")
            ax.set_ylabel(f"{ylab} [{yunit}]")
            axes.append(ax)
        head, _, cap = title.partition("\n")
        fig.suptitle(head, fontsize=14, fontweight="bold", y=0.998)
        if cap:
            fig.text(0.5, 0.982, "\n".join(textwrap.wrap(cap, 190)), ha="center", va="top", fontsize=9,
                     color="#333")
        fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=3.0, w_pad=2.0)
    return fig, axes


def scan_caption(p: Params, maps) -> str:
    kind = maps.get("kind", "period")
    n = len(maps["z_grating"]) * len(maps["y"])
    nd = int(maps["discarded"].sum()) if "discarded" in maps else 0
    ylab = Y_KINDS[kind][0]
    held = {"tg": "grating period solved for each Λ",
            "period": "grating period as on the axis",
            "bs_width": f"grating period solved for Λ = {p.tg_set_nm:g} nm"}[kind]
    return (f"{len(maps['z_grating'])} grating positions × {len(maps['y'])} values of {ylab} = {n} simulations; "
            f"{nd} discarded because their TG period is outside the target {tg_range_text(p)} (grey). "
            f"{held.capitalize()}. Held fixed: beam {p.beam_fwhm_x_mm}×{p.beam_fwhm_y_mm} mm, ΔE/E {p.bandwidth_pct}%, focus "
            f"{p.focus_fwhm_x_um} µm, beamstop {p.bs_width_mm if kind != 'bs_width' else '(axis)'} mm at z={p.z_bs} m, "
            f"{p.grating_type} grating duty {p.duty}, sample {p.sample_mode}, slit {p.slit_upstream_mm} mm before sample, "
            f"direct-beam beamstop {p.dbs_after_sample_mm} mm after sample, CCD {p.ccd_width_mm} mm at z={p.z_det} m. "
            + ("Optical probe {p.probe_wavelength_nm:g} nm. Outline = feasible (Λ in target, Λ > λ_probe/2, sep > 20 µm, "
               "unwanted < 1 %, overlap > 0.5, sample inside the A/B overlap region). Star = current setting."
               .format(p=p) if p.experiment == "optical" else
               "Outline = feasible (Λ in target, sep > 20 µm, unwanted < 1 %, overlap > 0.5, SD clearance > 20 µm, "
               "sample inside the A/B overlap region, SD beam on the CCD). Star = current setting."))


# ─────────────────────────────────────────────────────────────────────────────
#  Reporting
# ─────────────────────────────────────────────────────────────────────────────
def summary(res: Result) -> str:
    m, p = res.metrics, res.p
    if "z_sample" not in m:
        return "No valid configuration:\n  " + "\n  ".join(res.warnings)
    L = [
        f"λ = {p.wavelength_nm} nm, ΔE/E = {p.bandwidth_pct}%, beam {p.beam_fwhm_x_mm}×{p.beam_fwhm_y_mm} mm, split along {p.split_axis}",
        f"TG target {tg_range_text(p)}; grating period "
        + (f"solved for Λ = {p.tg_set_nm:g} nm" if p.period_mode == "target" else "set manually")
        + f" → Λ = {m['tg_period_nm']:.1f} nm " + ("(in range)" if m.get("tg_in_range") else "(OUT OF RANGE: discarded)"),
        f"grating z = {p.z_grating:.4f} m, p = {p.period_nm:.1f} nm, η(±1) = {m['eff_1st']*100:.1f}%, η(0) = {m['eff_0']*100:.1f}%",
        f"1st-order angle λ/p = {m['order_angle_mrad']:.2f} mrad; beam at grating {m['beam_at_grating_um']:.0f} µm",
        f"1st orders cross at z = {m['z_cross']:.4f} m ({(m['z_cross']-p.z_grating)*1e3:.1f} mm after grating)",
        f"sample z = {m['z_sample']:.4f} m; sample→CCD = {m['d_sample_det_mm']:.1f} mm",
        f"slit z = {m['z_slit']:.4f} m, opening [{m['slit_lo_um']:.0f}, {m['slit_hi_um']:.0f}] µm",
        f"order separation at slit = {m['sep_slit_um']:.0f} µm (nearest {m['sep_nearest']}), contamination = {m['contamination_pct']:.2g}%",
        f"at sample: A(-1) {m['fwhm_A_s_um']:.0f}×{m['fwhm_A_o_um']:.0f} µm, B(+1) {m['fwhm_B_s_um']:.0f}×{m['fwhm_B_o_um']:.0f} µm, overlap {m['overlap']:.2f}",
        f"crossing angle {m['crossing_angle_mrad']:.2f} mrad → TG period Λ = {m['tg_period_nm']:.1f} nm (~{m['n_fringes']:.0f} fringes)",
        f"angles to the sample normal (tilt {p.sample_tilt_mrad:g} mrad): A(-1) {m['angle_A_mrad']:+.3f} mrad "
        f"({m['angle_A_deg']:+.4f}°), B(+1) {m['angle_B_mrad']:+.3f} mrad ({m['angle_B_deg']:+.4f}°); "
        f"Λ along the surface {m['tg_surface_nm']:.1f} nm, fringe planes tilted {m['tg_tilt_mrad']:.2f} mrad",
        f"A/B overlap region z = {m['z_ov0']:.4f}–{m['z_ov1']:.4f} m ({m['ov_len_mm']:.1f} mm); sample margin {m['sample_ov_margin_mm']:.1f} mm",
        f"TG contrast {m['tg_contrast']:.2f} (A–B coherence {m['coherence_AB']:.2f}: they leave the KB {m['du_KB_mm']:.2f} mm apart, l_coh = {m['l_coh_KB_mm']:.2f} mm)",
        f"photons in A(-1)+B(+1): {m['P_1st_total']*100:.2f}% of KB input; peak fluence {m['fluence_A']:.3g}+{m['fluence_B']:.3g} mJ/cm²",
        f"pulse-front tilt across each 1st-order beam ≈ {m['pft_fs']:.1f} fs",
    ]
    if "z_dbs" in m:
        L.append(f"direct-beam beamstop at z = {m['z_dbs']:.4f} m ({p.dbs_after_sample_mm:.0f} mm after sample), "
                 f"[{m['dbs_lo_um']:.0f}, {m['dbs_hi_um']:.0f}] µm: SD clearance {m['sd_clear_dbs_um']:.0f} µm, "
                 f"direct leak {m['direct_leak_pct']:.2g}%, SD blocked {m['sd_blocked_pct']:.2g}%")
        L.append(f"CCD: direct beams at {m['det_direct_A_um']:.0f}/{m['det_direct_B_um']:.0f} µm, "
                 f"SD at {m['det_SD_A_um']:.0f}/{m['det_SD_B_um']:.0f} µm (FWHM {m['det_fwhm_sd_um']:.0f} µm); "
                 f"best SD beam {m['sd_ccd_margin_mm']:.1f} mm inside the {p.ccd_width_mm:g} mm chip edge "
                 + ("" if m['sd_ccd_margin_mm'] > 0 else "(MISSES THE CHIP)"))
    if p.experiment == "optical" and "probe_ok" in m:
        L.append(f"optical probe {p.probe_wavelength_nm:g} nm: Bragg angle {m['probe_bragg_deg']:.2f}° from the normal, "
                 f"used {m['probe_angle_used_deg']:.2f}° → η_rel = {m['probe_eta_rel']:.3f}, diffracted order "
                 f"{m['probe_order']:+d} leaves at {m['probe_diff_angle_deg']:.2f}°; acceptance (FWHM) "
                 f"{m['probe_accept_deg']:.2f}° for effective depth {m['probe_L_nm']:.0f} nm (thickness {p.sample_thickness_nm:g} nm, "
                 f"X-ray attenuation {p.xray_atten_nm:g} nm), n = {p.sample_n:g}; Klein Q = {m['probe_Q']:.2f}, {m['probe_regime']}")
    for w in res.warnings:
        L.append("WARNING: " + w)
    return "\n".join(L)


def save_size_table(res: Result, path):
    rows = size_table(res)
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[-1]).index(k) if k in rows[-1] else 99)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in r.items()})


def save_scan_csv(maps, path):
    zg, per = maps["z_grating"], maps["y"]
    kind = maps.get("kind", "period")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["z_grating_m", f"{kind}_{Y_KINDS[kind][1]}", "discarded"] + SCAN_KEYS)
        for j, pp in enumerate(per):
            for i, z in enumerate(zg):
                w.writerow([f"{z:.4f}", f"{pp:.4g}", int(maps["discarded"][j, i])]
                           + [f"{maps[k][j, i]:.5g}" for k in SCAN_KEYS])


# ─────────────────────────────────────────────────────────────────────────────
#  CLI: simulate both cases and write scans
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import argparse, os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default="sdxtg_output")
    ap.add_argument("--nz", type=int, default=90, help="grating positions in scan")
    ap.add_argument("--np", type=int, default=90, help="grating periods in scan (log-spaced)")
    ap.add_argument("--rays", type=int, default=3000, help="rays past the beamstop per simulation")
    ap.add_argument("--kind", choices=list(Y_KINDS), default="tg",
                    help="second scan axis: tg = TG period over the target range (grating period solved), "
                         "period = grating period, bs_width = beamstop width at the TG setpoint")
    ap.add_argument("--tgmin", type=float, help="target TG period range [nm] (default: preset)")
    ap.add_argument("--tgmax", type=float, help="equal to --tgmin for one exact TG period")
    ap.add_argument("--pmin", type=float, default=20.0, help="grating period range for --kind period")
    ap.add_argument("--pmax", type=float, default=5000.0)
    ap.add_argument("--zgmin", type=float, default=1.02)
    ap.add_argument("--zgmax", type=float, default=1.99)
    ap.add_argument("--config", help="JSON file with Params overrides")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    over = json.load(open(args.config)) if args.config else {}
    if args.tgmin is not None:
        over.update(tg_min_nm=args.tgmin, tg_max_nm=args.tgmax if args.tgmax is not None else args.tgmin)
        if "tg_set_nm" not in over:
            over["tg_set_nm"] = math.sqrt(over["tg_min_nm"] * over["tg_max_nm"])
    zg = np.linspace(args.zgmin, args.zgmax, args.nz)
    for case in ("pink", "mono"):
        p = preset(case, **over)
        kind = args.kind
        lo, hi = sorted((p.tg_min_nm, p.tg_max_nm))
        if kind == "tg" and math.isclose(lo, hi):
            kind = "bs_width"                  # one exact TG period: scan the beamstop width instead
        if kind == "tg":
            yv = np.geomspace(lo, hi, args.np)
        elif kind == "period":
            yv = np.geomspace(args.pmin, args.pmax, args.np)
        else:
            D = p.beam_fwhm_x_mm if p.split_axis == "x" else p.beam_fwhm_y_mm
            yv = np.linspace(0.3 * D, 1.6 * D, args.np)
        r = simulate(p, n_rays=20000)
        txt = summary(r)
        try:
            import sdxtg_wave
            wv = sdxtg_wave.wave_check(p, r, carpet=False)
            txt += (f"\nwave-optics check: TG contrast {wv.visibility:.2f} (ray model {r.metrics['tg_contrast']:.2f}, "
                    f"fully coherent {wv.visibility_coh:.2f})")
        except Exception as e:                       # noqa: BLE001
            txt += f"\nwave-optics check skipped: {e}"
        print(f"\n===== {case.upper()} =====\n{txt}")
        open(os.path.join(args.out, f"{case}_summary.txt"), "w").write(txt + "\n")
        save_size_table(r, os.path.join(args.out, f"{case}_beam_sizes.csv"))
        maps = scan(p, zg, yv, kind=kind, n_rays=args.rays,
                    progress=lambda f: print(f"\r  scan {f*100:5.1f}%", end="", flush=True))
        print()
        save_scan_csv(maps, os.path.join(args.out, f"{case}_scan.csv"))
        fig, _ = plot_scan(maps, p, title=f"{case.upper()} BEAM — correlation maps vs grating position and {Y_KINDS[kind][0]}\n"
                                           + scan_caption(p, maps))
        fig.savefig(os.path.join(args.out, f"{case}_scan.png"), dpi=110)
        plt.close(fig)
    print(f"\nwrote results to {args.out}/")


if __name__ == "__main__":
    main()

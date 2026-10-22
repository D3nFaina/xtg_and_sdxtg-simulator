"""
sdxtg_3d.py  --  3-D view of the SD-XTG setup (Plotly, rotate / zoom / pan in the browser)

Every beam is drawn as a tube whose cross-section at each z is the ellipse spanned by the ray
envelopes in the split axis (x) and in the other axis (y), so the 3-D shapes come straight from the
same rays as the x and y views.  Optionally a cloud of rays is drawn as soft points ("splats").
Elements (KB mirrors, beamstop, grating chip, slit blades, sample, direct-beam stop, CCD) are thin
boxes; in optical-probe mode the steering mirror and the probe, transmitted and diffracted beams are
added with the geometry of sdxtg_sim.probe_geometry().

Plot axes: horizontal = z (along the beam, mm), the two transverse axes = x and y (mm) multiplied
by `exag` (transverse exaggeration), because the beams are mm wide over 2 m.  With exag = 1 and the
"sample" region the drawing is to scale, so the probe angles are true.
"""
from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go

import sdxtg_sim as sim

COL = dict(full="#3BB273", A="#E4572E", B="#2E86AB", other="#A0AAB4", SD="#8E44AD", probe="#14B8A6",
           probe_d="#F59E0B", el="#334155", sample="#D6336C", ccd="#111827", grating="#64748B")


def _tube(z, cx, cy, ax, ay, color, name, opacity, exag, nth=28, show=True):
    """Mesh3d tube through ellipses (centre cx, cy; semi-axes ax, ay) at positions z (all in m)."""
    th = np.linspace(0, 2 * np.pi, nth, endpoint=False)
    Z = np.repeat(z, nth) * 1e3
    X = (cx[:, None] + ax[:, None] * np.cos(th)[None, :]).ravel() * 1e3 * exag
    Y = (cy[:, None] + ay[:, None] * np.sin(th)[None, :]).ravel() * 1e3 * exag
    i, j, k = [], [], []
    for a in range(len(z) - 1):
        for b in range(nth):
            p0, p1 = a * nth + b, a * nth + (b + 1) % nth
            q0, q1 = p0 + nth, p1 + nth
            i += [p0, p1]; j += [p1, q1]; k += [q0, q0]
    return go.Mesh3d(x=Z, y=X, z=Y, i=i, j=j, k=k, color=color, opacity=opacity, name=name,
                     flatshading=False, showlegend=show, hoverinfo="name",
                     lighting=dict(ambient=0.55, diffuse=0.6, specular=0.2, roughness=0.6))


def _box(z0, z1, x0, x1, y0, y1, color, name, opacity, exag, show=False):
    """Axis-aligned box (m) as Mesh3d."""
    zz = np.array([z0, z1]) * 1e3
    xx = np.array([x0, x1]) * 1e3 * exag
    yy = np.array([y0, y1]) * 1e3 * exag
    vx, vy, vz = [], [], []
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                vx.append(zz[a]); vy.append(xx[b]); vz.append(yy[c])
    I = [0, 0, 4, 4, 0, 0, 2, 2, 0, 0, 1, 1]
    J = [1, 2, 5, 6, 1, 4, 3, 6, 2, 4, 3, 5]
    K = [3, 3, 7, 7, 5, 5, 7, 7, 6, 6, 7, 7]
    return go.Mesh3d(x=vx, y=vy, z=vz, i=I, j=J, k=K, color=color, opacity=opacity, name=name,
                     showlegend=show, hoverinfo="name", flatshading=True)


def _disc(center, normal, radius, color, name, exag, n=40):
    """Flat disc (mirror) as Mesh3d; centre/normal in lab (x, y, z) metres."""
    nrm = sim._unit(normal)
    a = sim._unit(np.cross(nrm, [0, 0, 1]) if abs(nrm[2]) < 0.99 else np.cross(nrm, [1, 0, 0]))
    b = np.cross(nrm, a)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = center[None, :] + radius * (np.cos(th)[:, None] * a + np.sin(th)[:, None] * b)
    pts = np.vstack([center, pts])
    X, Y, Z = pts[:, 0] * 1e3 * exag, pts[:, 1] * 1e3 * exag, pts[:, 2] * 1e3
    i = [0] * n
    j = list(range(1, n + 1))
    k = [(t % n) + 1 for t in range(1, n + 1)]
    return go.Mesh3d(x=Z, y=X, z=Y, i=i, j=j, k=k, color=color, opacity=0.95, name=name, showlegend=True,
                     hoverinfo="name", flatshading=True)


def _line(P0, P1, color, name, exag, width=6, dash=None, show=True):
    return go.Scatter3d(x=[P0[2] * 1e3, P1[2] * 1e3], y=[P0[0] * 1e3 * exag, P1[0] * 1e3 * exag],
                        z=[P0[1] * 1e3 * exag, P1[1] * 1e3 * exag], mode="lines", name=name,
                        line=dict(color=color, width=width, dash=dash), showlegend=show, hoverinfo="name")


def _env(R, idx, order, zz, q=0.02, max_rays=3000, extra=None, z_ref=None):
    """Ellipse parameters (cx, cy, ax, ay) of a ray subset at the z positions zz."""
    if len(idx) > max_rays:
        idx = np.sort(np.random.default_rng(0).choice(idx, max_rays, replace=False))
    xs = R.s(zz[:, None], idx, order)
    if extra is not None:
        xs = xs + extra * (zz[:, None] - z_ref)
    ys = R.o(zz[:, None], idx)
    xl, xh = np.quantile(xs, [q, 1 - q], axis=1)
    yl, yh = np.quantile(ys, [q, 1 - q], axis=1)
    return (xl + xh) / 2, (yl + yh) / 2, np.maximum((xh - xl) / 2, 1e-7), np.maximum((yh - yl) / 2, 1e-7)


def figure_3d(res: sim.Result, region="full", exag=None, show_orders=True, show_rays=False, nz=48):
    p, R, z, m = res.p, res.rays, res.z, res.metrics
    full = region == "full"
    D = max(p.beam_fwhm_x_mm, p.beam_fwhm_y_mm) * 1e-3
    if exag is None:
        exag = 100.0 if full else 1.0
    if full:
        zlo, zhi = min(p.z_hm, p.z_vm) - 0.1, p.z_det + 0.02
    else:
        zlo = p.z_grating - 0.006
        if p.experiment == "optical" and "probe_mirror_z" in m:
            zlo = min(zlo, m["probe_mirror_z"] - 0.01)
        zhi = min(p.z_det, z["sample"] + max(0.06, 1.2 * (z["dbs"] - z["sample"])))
    tr = []

    def seg(a, b):
        a, b = max(a, zlo), min(b, zhi)
        return np.linspace(a, b, nz) if b > a else None

    # ── beams ───────────────────────────────────────────────────────────────
    zz = seg(min(p.z_hm, p.z_vm) - 0.1, p.z_bs)
    if zz is not None:
        tr.append(_tube(zz, *_env(R, R.full_idx, 0, zz), COL["full"], "full beam", 0.35, exag))
    wA = next(b for b in res.bundles if b.beam == "A" and b.wanted)
    wB = next(b for b in res.bundles if b.beam == "B" and b.wanted)
    for b in (wA, wB):
        zz = seg(p.z_bs, p.z_grating)
        if zz is not None:
            tr.append(_tube(zz, *_env(R, b.idx, 0, zz), COL[b.beam], f"beam {b.beam}", 0.45, exag))
    for b in res.bundles:
        if b.wanted:
            k = b.idx[b.slit_mask]
            zz = seg(p.z_grating, z["slit"])
            if zz is not None:
                tr.append(_tube(zz, *_env(R, b.idx, b.order, zz), COL[b.beam], b.name, 0.6, exag, show=False))
            zz = seg(z["slit"], z["dbs"])
            if zz is not None and len(k) > 5:
                tr.append(_tube(zz, *_env(R, k, b.order, zz), COL[b.beam], b.name, 0.75, exag))
        elif show_orders and b.eff > 0.005 and abs(b.order) <= 2:
            zz = seg(p.z_grating, z["slit"])
            if zz is not None:
                tr.append(_tube(zz, *_env(R, b.idx, b.order, zz), COL["other"], "other orders (to the slit)",
                                0.18, exag, nth=16, show=b is next(bb for bb in res.bundles if not bb.wanted)))
    # self-diffracted beams (SD mode)
    if p.experiment == "sd" and res.sd_slopes != (0.0, 0.0):
        sA, sB = res.sd_slopes
        for b, extra, nm in ((wA, sA - sB, "SD 2kA−kB"), (wB, sB - sA, "SD 2kB−kA")):
            k = b.idx[b.slit_mask]
            zz = seg(z["sample"], p.z_det)
            if zz is not None and len(k) > 5:
                tr.append(_tube(zz, *_env(R, k, b.order, zz, extra=extra, z_ref=z["sample"]), COL["SD"], nm,
                                0.7, exag, show=nm.endswith("kB")))
    # ray cloud ("splats")
    if show_rays:
        rng = np.random.default_rng(1)
        pts = []
        for b in (wA, wB):
            k = b.idx[b.slit_mask]
            k = rng.choice(k, min(250, len(k)), replace=False)
            zz = np.linspace(max(p.z_grating, zlo), min(z["dbs"], zhi), 40)
            xs = R.s(zz[:, None], k, b.order)
            ys = np.broadcast_to(R.o(zz[:, None], k), xs.shape)
            pts.append((np.broadcast_to(zz[:, None], xs.shape).ravel(), xs.ravel(), ys.ravel(), COL[b.beam]))
        for zz_, xs, ys, c in pts:
            tr.append(go.Scatter3d(x=zz_ * 1e3, y=xs * 1e3 * exag, z=ys * 1e3 * exag, mode="markers",
                                   marker=dict(size=1.6, color=c, opacity=0.25), showlegend=False, hoverinfo="skip"))

    # ── elements ────────────────────────────────────────────────────────────
    th = 0.4e-3 / (1 if full else 1)           # element thickness along z
    Hy = 1.3 * D if full else max(4e-3, 2 * m.get("fwhm_A_o_um", 300) * 1e-6 * 3)
    if full:
        Lm = 0.3
        tr.append(_box(p.z_hm - Lm / 2, p.z_hm + Lm / 2, -D / 2 - 1.5e-3, -D / 2 - 1e-3, -D, D, COL["el"],
                       "HM (horizontal focusing)", 0.5, exag, True))
        tr.append(_box(p.z_vm - Lm / 2, p.z_vm + Lm / 2, -D, D, -D / 2 - 1.5e-3, -D / 2 - 1e-3, COL["grating"],
                       "VM (vertical focusing)", 0.5, exag, True))
        bw = p.bs_width_mm * 1e-3 / 2
        o = p.bs_offset_mm * 1e-3
        if p.split_axis == "x":
            tr.append(_box(p.z_bs - th, p.z_bs + th, o - bw, o + bw, -D, D, COL["el"], "beamstop (splitter)", 0.9,
                           exag, True))
        else:
            tr.append(_box(p.z_bs - th, p.z_bs + th, -D, D, o - bw, o + bw, COL["el"], "beamstop (splitter)", 0.9,
                           exag, True))
    gc = p.grating_chip_mm * 1e-3 / 2
    if zlo <= p.z_grating <= zhi:
        tr.append(_box(p.z_grating - th / 4, p.z_grating + th / 4, -gc, gc, -gc, gc, COL["grating"], "grating chip",
                       0.35, exag, True))
    lo, hi = res.slit
    bl, bh = p.slit_blade_len_mm * 1e-3, p.slit_blade_h_mm * 1e-3 / 2
    tr.append(_box(z["slit"] - th / 2, z["slit"] + th / 2, lo - bl, lo, -bh, bh, COL["el"], "slit blades", 0.85,
                   exag, True))
    tr.append(_box(z["slit"] - th / 2, z["slit"] + th / 2, hi, hi + bl, -bh, bh, COL["el"], "slit blades", 0.85,
                   exag, False))
    ss_ = 2.5e-3
    tr.append(_box(z["sample"] - th / 4, z["sample"] + th / 4, -ss_, ss_, -ss_, ss_, COL["sample"], "sample", 0.45,
                   exag, True))
    if res.dbs != (0.0, 0.0) and zlo <= z["dbs"] <= zhi:
        tr.append(_box(z["dbs"] - th, z["dbs"] + th, res.dbs[0], res.dbs[1], -Hy, Hy, COL["el"], "direct-beam stop",
                       0.9, exag, True))
    if zlo <= p.z_det <= zhi + 0.03:
        off = m.get("ccd_offset_used_mm", p.ccd_offset_mm) * 1e-3
        cw = p.ccd_width_mm * 1e-3 / 2
        tr.append(_box(p.z_det, p.z_det + 2 * th, off - cw, off + cw, -cw, cw, COL["ccd"], "CCD chip", 0.6, exag, True))

    # ── optical probe ───────────────────────────────────────────────────────
    if p.experiment == "optical" and np.isfinite(m.get("probe_angle_used_deg", np.nan)):
        g = sim.probe_geometry(res)
        Lout = max(0.03, 1.2 * p.probe_mirror_dist_mm * 1e-3)
        tr.append(_disc(g["mirror"], g["mirror_normal"], p.probe_mirror_diam_mm * 1e-3 / 2, COL["probe"],
                        "steering mirror", exag))
        tr.append(_line(g["mirror"] - 0.04 * g["feed"], g["mirror"], COL["probe"], "probe (feed)", exag, 5, "dot",
                        False))
        tr.append(_line(g["mirror"], g["S"], COL["probe"], f"probe {p.probe_wavelength_nm:g} nm", exag, 7))
        tr.append(_line(g["S"], g["S"] + Lout * g["d_tr"], COL["probe"], "transmitted probe", exag, 4, "dash"))
        if g["d_df"] is not None:
            tr.append(_line(g["S"], g["S"] + Lout * g["d_df"], COL["probe_d"], "diffracted probe (signal)", exag, 7))

    # ── layout ──────────────────────────────────────────────────────────────
    fig = go.Figure(tr)
    if full:
        aspect = dict(x=3.2, y=1, z=1)
        mode = "manual"
    else:
        mode = "data"
        aspect = None
    fig.update_layout(
        height=720, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor="white",
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=11), itemsizing="constant"),
        scene=dict(
            xaxis=dict(title="z [mm]", backgroundcolor="#F6F8FB", gridcolor="#E6EAF0", range=[zlo * 1e3, zhi * 1e3]),
            yaxis=dict(title=f"x [mm] × {exag:g}" if exag != 1 else "x [mm]", backgroundcolor="#F6F8FB",
                       gridcolor="#E6EAF0"),
            zaxis=dict(title=f"y [mm] × {exag:g}" if exag != 1 else "y [mm]", backgroundcolor="#F6F8FB",
                       gridcolor="#E6EAF0"),
            aspectmode=mode, aspectratio=aspect,
            camera=dict(eye=dict(x=-1.1, y=-1.6, z=0.9) if full else dict(x=-1.4, y=-1.3, z=0.8),
                        up=dict(x=0, y=0, z=1))),
        uirevision=f"3d-{region}")
    return fig


def figure_probe_2d(res: sim.Result, plane="xz"):
    """To-scale drawing (mm, equal axes) of the sample region seen from above (x–z) or from the side
    (y–z): X-ray 1st orders, grating chip, slit blades, sample, direct-beam stop, steering mirror,
    probe, transmitted and diffracted probe."""
    p, R, z, m = res.p, res.rays, res.z, res.metrics
    g = sim.probe_geometry(res)
    ax = 0 if plane == "xz" else 1
    lab = p.split_axis if plane == "xz" else ("y" if p.split_axis == "x" else "x")
    f = go.Figure()
    zs = z["sample"]
    Lout = max(0.03, 1.2 * p.probe_mirror_dist_mm * 1e-3)
    zlo = min(g["mirror"][2], p.z_grating) - 0.008
    zhi = zs + Lout * 0.9
    # X-ray beams A(-1), B(+1): 2-98 % bands
    for b in res.bundles:
        if not b.wanted:
            continue
        zz = np.linspace(max(p.z_grating, zlo), min(z["dbs"], zhi), 80)
        k = b.idx[b.slit_mask]
        if plane == "xz":
            v = R.s(zz[:, None], k, b.order)
        else:
            v = np.broadcast_to(R.o(zz[:, None], k), (len(zz), len(k)))
        lo, hi = np.quantile(v, [0.02, 0.98], axis=1)
        f.add_trace(go.Scatter(x=np.r_[zz, zz[::-1]] * 1e3 - zs * 1e3, y=np.r_[lo, hi[::-1]] * 1e3, fill="toself",
                               line=dict(width=0), fillcolor=COL[b.beam], opacity=0.55, name=f"X-ray {b.name}"))

    def rect(z0, z1, a0, a1, color, name, show=True):
        f.add_trace(go.Scatter(x=np.array([z0, z1, z1, z0, z0]) * 1e3 - zs * 1e3, y=np.array([a0, a0, a1, a1, a0]) * 1e3,
                               fill="toself", line=dict(color=color, width=1), fillcolor=color, opacity=0.8,
                               name=name, showlegend=show))
    gc = p.grating_chip_mm * 1e-3 / 2
    rect(p.z_grating - 2e-4, p.z_grating + 2e-4, -gc, gc, COL["grating"], "grating chip")
    lo, hi = res.slit
    if plane == "xz":
        bl = p.slit_blade_len_mm * 1e-3
        rect(z["slit"] - 2e-4, z["slit"] + 2e-4, lo - bl, lo, COL["el"], "slit blades")
        rect(z["slit"] - 2e-4, z["slit"] + 2e-4, hi, hi + bl, COL["el"], "slit blades", False)
        if res.dbs != (0.0, 0.0):
            rect(z["dbs"] - 4e-4, z["dbs"] + 4e-4, res.dbs[0], res.dbs[1], COL["el"], "direct-beam stop")
    else:
        bh = p.slit_blade_h_mm * 1e-3 / 2
        rect(z["slit"] - 2e-4, z["slit"] + 2e-4, -bh, bh, COL["el"], "slit blades (seen edge-on, gap is in x)")
    t = p.sample_tilt_mrad * 1e-3 if plane == "xz" else 0.0
    hs = 2.5e-3
    f.add_trace(go.Scatter(x=[-hs * math.sin(t) * 1e3, hs * math.sin(t) * 1e3], y=[hs * math.cos(t) * 1e3 * -1,
                           hs * math.cos(t) * 1e3], mode="lines", line=dict(color=COL["sample"], width=5), name="sample"))
    # mirror: a segment along the mirror surface, in this projection
    M, n = g["mirror"], g["mirror_normal"]
    tdir = np.cross(n, [0, 1, 0] if plane == "xz" else [1, 0, 0])
    if np.linalg.norm(tdir) < 1e-6:
        tdir = np.array([1.0, 0, 0])
    tdir = sim._unit(tdir) * p.probe_mirror_diam_mm * 1e-3 / 2
    P0, P1 = M - tdir, M + tdir
    f.add_trace(go.Scatter(x=[(P0[2] - zs) * 1e3, (P1[2] - zs) * 1e3], y=[P0[ax] * 1e3, P1[ax] * 1e3], mode="lines",
                           line=dict(color=COL["probe"], width=7), name="steering mirror"))

    def ray(A, B, color, name, width=3, dash=None):
        f.add_trace(go.Scatter(x=[(A[2] - zs) * 1e3, (B[2] - zs) * 1e3], y=[A[ax] * 1e3, B[ax] * 1e3], mode="lines",
                               line=dict(color=color, width=width, dash=dash), name=name))
    ray(M - 0.03 * g["feed"], M, COL["probe"], "probe feed", 2, "dot")
    ray(M, g["S"], COL["probe"], f"probe {p.probe_wavelength_nm:g} nm, {m['probe_angle_used_deg']:.2f}°", 4)
    ray(g["S"], g["S"] + Lout * g["d_tr"], COL["probe"], "transmitted probe", 2, "dash")
    if g["d_df"] is not None:
        ray(g["S"], g["S"] + Lout * g["d_df"], COL["probe_d"], f"diffracted probe, {m['probe_diff_angle_deg']:.2f}°", 4)
    f.update_layout(height=430, template="plotly_white", margin=dict(l=55, r=15, t=40, b=45),
                    legend=dict(orientation="h", y=-0.18, font=dict(size=10)),
                    title=dict(text=("Top view (x–z)" if plane == "xz" else "Side view (y–z)") + ", to scale, "
                               "origin at the sample", font=dict(size=13)))
    f.update_xaxes(title="z − z_sample [mm]", range=[(zlo - zs) * 1e3, (zhi - zs) * 1e3])
    f.update_yaxes(title=f"{lab} [mm]", scaleanchor="x", scaleratio=1)
    return f

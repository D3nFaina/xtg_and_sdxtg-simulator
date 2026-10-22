"""
check_views.py  --  automatic consistency checks of the drawings, run before every release.

    python check_views.py          (exit code 0 = all checks passed)

What it checks, for the pink and mono presets in the X-ray pump / optical probe mode:
  1. Probe angle. The probe uses the phase-matching angle sin φ_B = λ_probe / (2Λ), passes the slit gap,
     and is drawn at that angle in every view: the 3D whole beamline (angle measured in the displayed,
     stretched scene), the 3D to-scale regions (grating → CCD, close-up), the 2D beamline zoom (arrows in
     screen pixels) and the 2D to-scale close-up.
  2. Beam continuity. A and B are the two parts of the full beam on either side of the beamstop, so in the
     3D tubes and in the 2D sketch their outer edges must meet the full beam at the beamstop, and each
     half-beam must meet its diffracted order at the grating and across the slit, without a step.
  3. Parameters survive a switch of the experiment (optical → SD → optical) in the running app.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sdxtg_sim as sim      # noqa: E402
import sdxtg_3d as s3d       # noqa: E402

TOL_DEG = 0.5                # angle tolerance
TOL_EDGE = 0.03              # beam-edge step tolerance, fraction of the full beam width at that z
FAILS = []


def check(ok, what):
    print(("  ok    " if ok else "  FAIL  ") + what)
    if not ok:
        FAILS.append(what)


def rings(tr, nth=48):
    """Cross-section rings of a _shape_tube Mesh3d: list of (z, y-array, z-array) per ring."""
    x, y, z = (np.asarray(v, float) for v in (tr.x, tr.y, tr.z))
    n = len(x) // nth
    return [(x[i * nth], y[i * nth:(i + 1) * nth], z[i * nth:(i + 1) * nth]) for i in range(n)]


def pca_angle(P):
    """Angle (deg) of the main axis of points P (N, 2) = (along the beam, across) to the beam axis."""
    P = P - P.mean(0)
    w, v = np.linalg.eigh(P.T @ P)
    d = v[:, -1]
    return math.degrees(math.atan2(abs(d[1]), abs(d[0])))


def check_case(name):
    print(f"\n── {name} preset, X-ray pump · optical probe ──")
    p = sim.preset(name).copy(experiment="optical")
    res = sim.simulate(p)
    m, z = res.metrics, res.z
    lam_p, Ls = p.probe_wavelength_nm * 1e-9, m["tg_surface_nm"] * 1e-9
    phiB = math.degrees(math.asin(lam_p / (2 * Ls)))
    phi = m["probe_angle_used_deg"]
    # 1. probe physics -----------------------------------------------------
    check(abs(phi - phiB) < 0.01, f"probe at the phase-matching angle: φ = {phi:.3f}°, φ_B = {phiB:.3f}°")
    check(1.0 < phi < 80.0, f"probe angle is physical (1° < φ < 80°): {phi:.2f}°")
    g = sim.probe_geometry(res)
    check(g["slit_T"] > 0.9, f"probe passes the slit gap: {g['slit_T'] * 100:.0f} %")
    check(not g["obstacles"], "no probe obstacles" + ("" if not g["obstacles"] else ": " + "; ".join(g["obstacles"])))

    # 1b. 3D whole beamline: angle in the displayed (stretched) scene
    f = s3d.figure_3d(res, "full")
    sc = f.layout.scene
    xr, yr = sc.xaxis.range, sc.yaxis.range
    ax, ay = sc.aspectratio.x, sc.aspectratio.y
    probe_tr = [t for t in f.data if getattr(t, "meta", None) == "probe"]
    check(len(probe_tr) >= 2, f"3D whole beamline: probe drawn ({len(probe_tr)} lines)")
    for t in probe_tr:
        dz = (t.x[1] - t.x[0]) / (xr[1] - xr[0]) * ax
        dy = (t.y[1] - t.y[0]) / (yr[1] - yr[0]) * ay
        a = math.degrees(math.atan2(abs(dy), abs(dz)))
        check(abs(a - phi) < TOL_DEG, f"3D whole beamline: '{t.name[:30]}' at {a:.2f}° on screen (φ = {phi:.2f}°)")
        check(abs(t.x[0] - z["sample"] * 1e3) < 1e-6 and abs(t.y[0]) < 1e-9,
              f"3D whole beamline: '{t.name[:30]}' goes through the sample point")

    # 1c. 3D to-scale regions: angle of the probe tubes in the lab frame
    for reg in ("layout", "sample"):
        f = s3d.figure_3d(res, reg, exag=1)
        for t in f.data:
            if t.type == "mesh3d" and (t.name or "").startswith(("probe", "diffracted probe")):
                P = np.c_[np.asarray(t.x, float), np.asarray(t.y, float)]
                a = pca_angle(P)
                check(abs(a - phi) < TOL_DEG, f"3D {reg}: '{t.name[:30]}' at {a:.2f}° (φ = {phi:.2f}°)")

    # 1d. 2D to-scale close-up (equal axes): probe band direction
    f = s3d.figure_pump_probe(res)
    for t in f.data:
        if (t.name or "").startswith("optical probe"):
            P = np.c_[np.asarray(t.x, float), np.asarray(t.y, float)]
            P = P[: len(P) // 2]                       # one edge of the band (the polygon is edge + reversed edge)
            P = P[np.isfinite(P).all(1)]
            a = pca_angle(P)
            check(abs(a - phi) < TOL_DEG, f"2D close-up: '{t.name[:30]}' at {a:.2f}° (φ = {phi:.2f}°)")

    # 2. continuity, 3D whole beamline -------------------------------------
    f = s3d.figure_3d(res, "full")
    byname = {}
    for t in f.data:
        if t.type == "mesh3d":
            byname.setdefault(t.name, []).append(t)
    full = rings(byname["full beam"][0])[-1]
    span = full[1].max() - full[1].min()
    for beam, side in (("A", 1), ("B", -1)):
        r0 = rings(byname[f"beam {beam}"][0])[0]
        e_full = full[1].max() if side > 0 else full[1].min()
        e_half = r0[1].max() if side > 0 else r0[1].min()
        check(abs(e_full - e_half) < TOL_EDGE * span,
              f"3D: beam {beam} meets the full beam at the beamstop (outer edge step "
              f"{abs(e_full - e_half) / span * 100:.1f} % of the beam width)")
        wd = r0[1].max() - r0[1].min()
        check(wd > 0.03 * span, f"3D: beam {beam} is visible after the beamstop (width {wd / span * 100:.0f} % of the full beam)")
        v_full, v_half = full[2].max() - full[2].min(), r0[2].max() - r0[2].min()
        check(abs(v_full - v_half) < TOL_EDGE * v_full,
              f"3D: beam {beam} keeps the full beam height at the beamstop ({v_half / v_full * 100:.0f} %)")
        last = rings(byname[f"beam {beam}"][0])[-1]
        wanted = next(b for b in res.bundles if b.beam == beam and b.wanted)
        segs = byname[wanted.name]
        first = rings(segs[0])[0]
        sp = last[1].max() - last[1].min()
        step = max(abs(last[1].max() - first[1].max()), abs(last[1].min() - first[1].min()))
        check(step < TOL_EDGE * sp, f"3D: beam {beam} meets {wanted.name} at the grating (step {step / sp * 100:.1f} %)")
        if len(segs) > 1:
            a_, b_ = rings(segs[0])[-1], rings(segs[1])[0]
            sp = a_[1].max() - a_[1].min()
            step = max(abs(a_[1].max() - b_[1].max()), abs(a_[1].min() - b_[1].min()))
            check(step < TOL_EDGE * sp, f"3D: {wanted.name} continuous across the slit (step {step / sp * 100:.1f} %)")

    # 2b. continuity, 2D sketch (sim.envelopes is what the Beamline tab draws)
    env = sim.envelopes(res)
    E = {(e[0], round(float(e[2][0]), 6)): e for e in env}
    fulle = next(e for e in env if e[0] == "full")
    w = fulle[4][-1] - fulle[3][-1]
    for beam, side in (("A", 1), ("B", -1)):
        seg = next(e for e in env if e[0] == beam)
        e_full = fulle[4][-1] if side > 0 else fulle[3][-1]
        e_half = seg[4][0] if side > 0 else seg[3][0]
        check(abs(e_full - e_half) < TOL_EDGE * w, f"2D: beam {beam} meets the full beam at the beamstop "
                                                    f"(step {abs(e_full - e_half) / w * 100:.1f} %)")
        wanted = next(b for b in res.bundles if b.beam == beam and b.wanted)
        post = [e for e in env if e[0] == wanted.name]
        ww = seg[4][-1] - seg[3][-1]
        step = max(abs(seg[3][-1] - post[0][3][0]), abs(seg[4][-1] - post[0][4][0]))
        check(step < TOL_EDGE * ww, f"2D: beam {beam} meets {wanted.name} at the grating (step {step / ww * 100:.1f} %)")
    return res


def check_app():
    print("\n── running app (Streamlit AppTest) ──")
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as exc:          # very old Streamlit
        print(f"  skip  AppTest not available ({exc})")
        return
    at = AppTest.from_file(os.path.join(HERE, "sdxtg_app.py"), default_timeout=300)
    at.run()
    check(not at.exception, "app runs without exceptions (SD mode)")
    for mode in ("optical", "sd", "optical"):
        at.radio(key="experiment").set_value(mode).run()
    check(not at.exception, "app runs without exceptions after optical → SD → optical")
    ss = at.session_state
    check(abs(ss["probe_wavelength_nm"] - 400.0) < 1e-9 and ss["probe_auto_bragg"] and ss["probe_waist_um"] == 50.0,
          f"probe settings kept after switching: λ = {ss['probe_wavelength_nm']}, Bragg = {ss['probe_auto_bragg']}, "
          f"w₀ = {ss['probe_waist_um']}")
    res = sim.simulate(sim.preset("pink").copy(experiment="optical"))
    phi = res.metrics["probe_angle_used_deg"]
    # the Beamline tab: probe arrows in the stretched 2D zoom, in screen pixels
    found = 0
    for e in at.get("plotly_chart"):
        sp = json.loads(e.proto.spec)
        for an in sp["layout"].get("annotations", []):
            if an.get("xref") == "x2" and an.get("showarrow") and "ax" in an and an.get("arrowcolor"):
                a = math.degrees(math.atan2(abs(an["ay"]), abs(an["ax"])))
                found += 1
                check(abs(a - phi) < TOL_DEG, f"2D beamline zoom: probe arrow '{an.get('text', '')[:25]}' at "
                                              f"{a:.2f}° on screen (φ = {phi:.2f}°)")
    check(found >= 2, f"2D beamline zoom: probe drawn ({found} arrows)")


if __name__ == "__main__":
    for case in ("pink", "mono"):
        check_case(case)
    if "--no-app" not in sys.argv:
        check_app()
    print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} CHECK(S) FAILED"))
    sys.exit(1 if FAILS else 0)

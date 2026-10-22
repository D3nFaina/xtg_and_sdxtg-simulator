"""
sdxtg_app.py  --  Interactive SD-XTG beam-path simulator (Streamlit + Plotly)

    pip install numpy matplotlib streamlit plotly
    cd sdxtg && streamlit run sdxtg_app.py        # opens http://localhost:8501

All physics lives in sdxtg_sim.py; this file only builds the interface.
"""
from __future__ import annotations

import io
import json
import math
import os
import sys


def _under_streamlit() -> bool:
    """True when this file is being executed by `streamlit run` (not by plain python / Spyder F5)."""
    try:
        from streamlit.runtime import exists
        return exists()
    except Exception:
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            return get_script_run_ctx() is not None
        except Exception:
            return False


if __name__ == "__main__" and not _under_streamlit():
    # Started with Spyder's Run button or `python sdxtg_app.py`: hand over to the launcher,
    # which starts the real Streamlit server and opens the browser.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import run_app
    run_app.main()
    raise SystemExit(0)

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

import sdxtg_sim as sim
import sdxtg_wave as wave
import sdxtg_3d as s3d

# Compatibility with older Streamlit releases (width="stretch" exists only from 1.46 on).
ST_VER = tuple(int(x) for x in st.__version__.split(".")[:2] if x.isdigit())
WIDE = dict(width="stretch") if ST_VER >= (1, 46) else dict(use_container_width=True)
CAN_SELECT = ST_VER >= (1, 35)          # click-on-map needs plotly_chart(on_select=...)
if ST_VER < (1, 30):
    raise SystemExit(f"Streamlit {st.__version__} is too old for this app; run: pip install -U streamlit")

st.set_page_config(page_title="SD-XTG beam path simulator", page_icon="🔬", layout="wide",
                   initial_sidebar_state="expanded")

# ─────────────────────────────────────────────────────────────────────────────
#  Look & feel
# ─────────────────────────────────────────────────────────────────────────────
C = dict(A="#E4572E", B="#2E86AB", full="#3BB273", SD="#8E44AD", other="#A0AAB4",
         ink="#1F2933", muted="#6B7785", grid="#E6EAF0", slit="#334155", sample="#D6336C",
         good="#1E9E6A", bad="#D64545", warn="#E0A100")

st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1600px;}
  h1 {font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0 !important;}
  .subtitle {color:#6B7785; font-size: 0.95rem; margin-bottom: 1.1rem;}
  .kpi {background: #FFFFFF; border: 1px solid #E6EAF0; border-radius: 12px; padding: 12px 14px 10px;
        box-shadow: 0 1px 2px rgba(16,24,40,.04); height: 100%;}
  .kpi .lab {font-size: .74rem; text-transform: uppercase; letter-spacing: .06em; color:#6B7785;}
  .kpi .val {font-size: 1.45rem; font-weight: 650; color:#1F2933; line-height: 1.25; margin-top: 2px;}
  .kpi .sub {font-size: .78rem; color:#6B7785;}
  .kpi.good {border-left: 4px solid #1E9E6A;} .kpi.bad {border-left: 4px solid #D64545;}
  .kpi.neutral {border-left: 4px solid #2E86AB;}
  section[data-testid="stSidebar"] {background: #F6F8FB;}
  section[data-testid="stSidebar"] .stMarkdown p {font-size: .85rem;}
  div[data-testid="stExpander"] details {border-radius: 10px; border-color: #E6EAF0; background: white;}
  .stTabs [data-baseweb="tab-list"] {gap: 4px;}
  .stTabs [data-baseweb="tab"] {padding: 8px 16px; border-radius: 8px 8px 0 0;}
  .mapdesc {font-size: .78rem; color:#55606E; margin: -6px 0 18px 0; line-height: 1.35;}
</style>
""", unsafe_allow_html=True)

LAYOUT = dict(template="plotly_white", font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", size=12,
                                                    color=C["ink"]),
              margin=dict(l=60, r=20, t=40, b=45), hovermode="closest",
              legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom", font=dict(size=11)))
AX = dict(gridcolor=C["grid"], zeroline=False, linecolor="#C9D1DB", ticks="outside", tickcolor="#C9D1DB")


def rgba(hex_, a):
    h = hex_.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


# ─────────────────────────────────────────────────────────────────────────────
#  Parameters <-> session state
# ─────────────────────────────────────────────────────────────────────────────
FIELDS = [f.name for f in sim.dataclasses.fields(sim.Params)]
ss = st.session_state


def load_params(p: sim.Params):
    for k in FIELDS:
        ss[k] = getattr(p, k)
    ss["bar_phase_pi"] = p.bar_phase_rad / math.pi
    ss["beam_fwhm"] = p.beam_fwhm_x_mm
    ss["focus_fwhm"] = p.focus_fwhm_x_um


if "z_grating" not in ss:
    ss["case"] = "Pink · 4 mm"
    load_params(sim.preset("pink"))
# Streamlit drops the state of a widget that was not drawn in the previous run (e.g. the probe inputs
# while the experiment is "X-ray SD"); the widget would then come back at its minimum value. Restore
# every parameter from the copy saved at the end of the previous run.
for _k, _v in ss.get("_saved_params", {}).items():
    if _k not in ss:
        ss[_k] = _v
# a click on a scan map / "apply" button stores a pending change, applied before widgets exist
if ss.get("pending"):
    for k, v in ss.pop("pending").items():
        ss[k] = v


def on_case():
    name = "pink" if ss["case"].startswith("Pink") else "mono"
    cur = current_params()
    load_params(cur.copy(**sim.PRESETS[name]))


def current_params() -> sim.Params:
    kw = {k: ss[k] for k in FIELDS if k in ss}
    kw["bar_phase_rad"] = ss["bar_phase_pi"] * math.pi
    kw["beam_fwhm_x_mm"] = kw["beam_fwhm_y_mm"] = ss["beam_fwhm"]
    kw["focus_fwhm_x_um"] = kw["focus_fwhm_y_um"] = ss["focus_fwhm"]
    kw["n_rays"] = int(kw["n_rays"])
    return sim.Params(**kw)


# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar controls
# ─────────────────────────────────────────────────────────────────────────────
sb = st.sidebar
sb.markdown("### ⚙️ Beamline parameters")
sb.radio("Input beam", ["Pink · 4 mm", "Mono · 1 mm"], key="case", horizontal=True, on_change=on_case,
         help="Loads the preset beam size, bandwidth, pulse energy, beamstop and grating of that case.")
sb.radio("Experiment", ["sd", "optical"], key="experiment", horizontal=True,
         format_func=lambda v: "X-ray SD on CCD" if v == "sd" else "X-ray pump · optical probe",
         help="SD: self-diffracted X-rays on the CCD (direct-beam stop and CCD checks apply). "
              "Optical probe: the X-ray TG is read by an optical beam at its phase-matching angle; needs Λ > λ_probe/2.")

with sb.expander("Beam before the KB", expanded=False):
    st.number_input("Beam size at the KB [mm]", 0.1, 10.0, step=0.1, key="beam_fwhm",
                    help="Round beam, same size in x and y. Its meaning is set below.")
    st.radio("Beam size is the", ["waist", "d2w", "fwhm"], key="beam_size_def", horizontal=True,
             format_func=lambda v: {"waist": "waist w₀", "d2w": "1/e² diameter", "fwhm": "FWHM"}[v],
             help="Gaussian beam: FWHM = 1.177 w₀ = 0.589 × (1/e² diameter).")
    _pf = current_params()
    st.caption(f"FWHM used: {_pf.beam_fwhm('x'):.2f} mm")
    st.number_input("Focal spot FWHM [µm]", 1.0, 200.0, step=1.0, key="focus_fwhm")
    st.number_input("Bandwidth ΔE/E FWHM [%]", 0.0, 5.0, step=0.01, format="%.3f", key="bandwidth_pct")
    st.number_input("Pulse energy on the KB [µJ]", 0.1, 5000.0, step=10.0, key="pulse_energy_uJ")
    st.number_input("KB transmission", 0.0, 1.0, step=0.05, key="kb_transmission")
    st.number_input("Transverse coherence length ÷ rms beam size", 0.1, 1000.0, step=0.1, key="coh_ratio",
                    help="Gaussian Schell model on the KB. 2.7 ≈ 80 % global degree of coherence (typical SASE). "
                         "Large = fully coherent. A and B must be mutually coherent to form a grating.")
    st.number_input("Wavelength [nm]", 0.5, 20.0, step=0.01, key="wavelength_nm")
    st.radio("Profile", ["gaussian", "flattop"], key="profile", horizontal=True)

with sb.expander("KB mirrors and CCD", expanded=False):
    st.caption("z is measured from the horizontal mirror (HM). Each focal length is measured from its own mirror.")
    st.number_input("f_H, from HM [m]", 0.3, 5.0, step=0.01, key="f_h")
    st.number_input("VM position z [m]", 0.05, 2.0, step=0.01, key="z_vm")
    st.number_input("f_V, from VM [m]", 0.3, 5.0, step=0.01, key="f_v")
    st.number_input("CCD position z [m]", 0.5, 5.0, step=0.005, format="%.3f", key="z_det")
    st.number_input("CCD chip side [mm]", 1.0, 100.0, step=0.1, key="ccd_width_mm",
                    help="Square chip. 1024 × 13 µm = 13.3 mm (e.g. Andor iKon-M, PI-MTE 1024); "
                         "2048 × 13.5 µm = 27.6 mm (PI-MTE 2048, Andor iKon-L).")
    st.toggle("Move the CCD sideways onto the SD beam", key="ccd_follow_sd",
              help="Small Λ means large angles: the self-diffracted beam leaves at ~1.5× the crossing angle "
                   "and misses a CCD centred on the axis. On: the chip is centred on SD(A).")
    st.number_input("CCD lateral offset [mm]", -200.0, 200.0, step=0.5, key="ccd_offset_mm",
                    disabled=ss["ccd_follow_sd"])

with sb.expander("Beam splitter (beamstop after the KB)", expanded=True):
    st.radio("Split axis", ["x", "y"], key="split_axis", horizontal=True,
             help="x = horizontal (focused by HM), y = vertical (focused by VM).")
    st.slider("Beamstop z [m]", 0.5, 1.95, step=0.005, key="z_bs")
    st.slider("Beamstop width [mm]", 0.0, 6.0, step=0.01, key="bs_width_mm")
    st.slider("Beamstop offset [mm]", -1.0, 1.0, step=0.01, key="bs_offset_mm")

def on_period_mode():
    # switching to manual: start from the period that was solved for the setpoint
    if ss["period_mode"] == "manual" and ss.get("last_solved_period"):
        ss["period_nm"] = round(float(ss["last_solved_period"]), 1)


with sb.expander("Transient-grating target", expanded=True):
    st.caption("Only settings whose TG period Λ falls in this range are kept; the others are discarded "
               "(scans show them grey). Set min = max for one exact Λ.")
    c1, c2 = st.columns(2)
    c1.number_input("Λ min [nm]", 2.0, 5000.0, step=1.0, key="tg_min_nm")
    c2.number_input("Λ max [nm]", 2.0, 5000.0, step=1.0, key="tg_max_nm")
    st.number_input("Tolerance [%]", 0.0, 50.0, step=0.5, key="tg_tol_pct",
                    help="Λ may miss the range by this much (needed for one exact Λ: the ray model has "
                         "Monte-Carlo noise of ~0.1 %).")
    st.radio("Grating period", ["target", "manual"], key="period_mode", horizontal=True, on_change=on_period_mode,
             format_func=lambda v: "solve from Λ" if v == "target" else "set by hand",
             help="solve from Λ: the grating period is computed (exact grating equation) so that A(−1) and B(+1) "
                  "cross with the Λ setpoint below, wherever the grating is.")
    st.number_input("Λ setpoint [nm]", 2.0, 5000.0, step=1.0, key="tg_set_nm",
                    disabled=ss["period_mode"] != "target")
    tg_note = st.empty()

with sb.expander("Grating", expanded=True):
    st.slider("Grating z [m]", 0.6, 1.995, step=0.0005, format="%.4f", key="z_grating")
    st.number_input("Grating period [nm] (set by hand)", 5.0, 20000.0, step=1.0, format="%.1f", key="period_nm",
                    disabled=ss["period_mode"] == "target")
    st.slider("Duty cycle (bar fraction)", 0.05, 0.95, step=0.01, key="duty")
    st.radio("Type", ["amplitude", "phase"], key="grating_type", horizontal=True,
             help="amplitude: bars with transmission T, no phase. phase: bars with transmission T and phase φ.")
    st.slider("Bar intensity transmission T", 0.0, 1.0, step=0.01, key="bar_T")
    st.slider("Bar phase φ [π]", 0.0, 2.0, step=0.01, key="bar_phase_pi",
              disabled=ss["grating_type"] != "phase")
    st.number_input("Membrane transmission", 0.0, 1.0, step=0.05, key="membrane_T")
    st.number_input("Grating chip size [mm]", 0.5, 50.0, step=0.5, key="grating_chip_mm")

def load_material():
    name = ss["sample_preset"]
    if name == "custom":
        return
    ss["sample_formula"], ss["sample_density"] = name, sim.SAMPLE_PRESETS[name]
    ss["sample_auto_att"] = True
    try:
        import periodictable as pt
        ref = sim.SAMPLE_REF.get(str(pt.formula(name).hill))
    except Exception:                                               # noqa: BLE001
        ref = None
    if ref and ref.get("n"):
        ss["sample_n"] = round(float(np.interp(ss["probe_wavelength_nm"], *ref["n"])), 3)


FLAG = dict(ok=("#1E9E6A", "consistent"), bad=("#D64545", "not consistent"), warn=("#E0A100", "check"),
            na=("#8A94A3", "no reference"))
with sb.expander("Sample material", expanded=True):
    if "sample_preset" not in ss:
        ss["sample_preset"] = "SrTiO3" if ss["sample_formula"] == "SrTiO3" else "custom"
    st.selectbox("Material", list(sim.SAMPLE_PRESETS) + ["custom"], key="sample_preset", on_change=load_material,
                 help="Fills formula, density and the optical index at the probe wavelength from the literature table; "
                      "choose 'custom' and type your own values.")
    c1, c2 = st.columns([1.3, 1])
    c1.text_input("Formula", key="sample_formula")
    c2.number_input("ρ [g/cm³]", 0.01, 25.0, step=0.01, key="sample_density")
    st.number_input("Thickness [nm]", 1.0, 1e6, step=10.0, key="sample_thickness_nm")
    st.toggle("X-ray L_att from the Henke tables (CXRO)", key="sample_auto_att",
              help="On: attenuation length at the pump energy computed from formula and density with periodictable. "
                   "Off: type a value (e.g. a measured one near an absorption edge); it is checked against Henke.")
    st.number_input("X-ray attenuation length [nm]", 0.1, 1e7, step=1.0, key="xray_atten_nm",
                    disabled=ss["sample_auto_att"],
                    help="1/e intensity depth of the pump. The TG amplitude decays as exp(−z/L_att) into the sample.")
    if ss["experiment"] == "optical":
        st.number_input("Refractive index at the probe wavelength", 1.0, 5.0, step=0.01, key="sample_n")
    rows = sim.sample_checks(current_params())
    worst = "bad" if any(r[3] == "bad" for r in rows) else ("ok" if all(r[3] in ("ok", "warn") for r in rows) else "na")
    col, txt = FLAG[worst]
    html = (f"<div style='margin:4px 0 6px;padding:6px 10px;border-radius:8px;background:{col};color:white;"
            f"font-weight:600;font-size:.85rem'>Sample: {'consistent' if worst == 'ok' else txt}</div>")
    for q, val, ref, stt, note in rows:
        c, t = FLAG[stt]
        html += (f"<div style='font-size:.76rem;line-height:1.3;margin:3px 0'><span style='color:{c};font-weight:700'>"
                 f"● {t}</span> · <b>{q}</b> {val}" + (f" vs {ref}" if ref not in ("-", val) else "")
                 + f"<br><span style='color:#6B7785'>{note}</span></div>")
    st.markdown(html, unsafe_allow_html=True)


if ss["experiment"] == "optical":
    with sb.expander("Optical probe", expanded=True):
        st.number_input("Probe wavelength [nm]", 100.0, 5000.0, step=10.0, key="probe_wavelength_nm")
        st.toggle("Probe at the Bragg (phase-matched) angle", key="probe_auto_bragg")
        st.number_input("Probe incidence angle from the normal [deg]", -89.0, 89.0, step=0.1, key="probe_angle_deg",
                        disabled=ss["probe_auto_bragg"])
        st.number_input("Steering mirror → sample [mm]", 5.0, 1000.0, step=1.0, key="probe_mirror_dist_mm")
        st.number_input("Mirror diameter [mm]", 1.0, 100.0, step=0.1, key="probe_mirror_diam_mm")
        st.radio("Probe reaches the mirror", ["z", "side"], key="probe_feed", horizontal=True,
                 format_func=lambda v: "along the beamline (+z)" if v == "z" else "from the side",
                 help="The probe always travels in the horizontal plane of the X-ray beams (same height), so it "
                      "diffracts in the same plane as the self-diffracted beams.")
        st.number_input("Probe waist w₀ on the sample (1/e² radius) [µm]", 1.0, 5000.0, step=5.0, key="probe_waist_um",
                        help="Gaussian waist of the probe at the sample. FWHM = 1.1774 w₀; the drawings show the FWHM.")
        st.caption(f"Probe FWHM = {1.17741 * ss['probe_waist_um']:.1f} µm (footprint on the sample is longer along "
                   f"{ss['split_axis']} by 1/cos φ).")

with sb.expander("Sample and order-selecting slit", expanded=True):
    st.radio("Sample position", ["crossing", "fixed"], key="sample_mode", horizontal=True,
             help="crossing: the sample follows the crossing point of A(−1) and B(+1).")
    st.slider("Sample z (fixed mode) [m]", 0.6, 2.0, step=0.0005, format="%.4f", key="z_sample",
              disabled=ss["sample_mode"] != "fixed")
    st.number_input("Sample tilt [mrad]", -500.0, 500.0, step=0.1, key="sample_tilt_mrad",
                    help="Rotation of the sample normal from the beam axis z, in the split plane. "
                         "0 = normal along z: A(−1) and B(+1) come in at ±θ/2.")
    st.slider("Slit distance before sample [mm]", 0.1, 100.0, step=0.1, key="slit_upstream_mm")
    c1, c2 = st.columns(2)
    c1.number_input("Blade length [mm]", 0.5, 100.0, step=0.5, key="slit_blade_len_mm")
    c2.number_input("Blade height [mm]", 0.5, 100.0, step=0.5, key="slit_blade_h_mm")
    st.toggle("Auto slit opening", key="slit_auto")
    if ss["slit_auto"]:
        st.slider("Extra opening per side [µm]", 0.0, 200.0, step=1.0, key="slit_margin_um")
    else:
        st.slider("Slit width [µm]", 5.0, 3000.0, step=5.0, key="slit_width_um")

with sb.expander("Direct-beam beamstop (in front of the CCD)", expanded=True):
    st.slider("Distance after sample [mm]", 1.0, 800.0, step=1.0, key="dbs_after_sample_mm")
    st.toggle("Auto width (just covers the direct beams)", key="dbs_auto")
    if ss["dbs_auto"]:
        st.slider("Extra width per side [µm]", 0.0, 200.0, step=1.0, key="dbs_margin_um")
    else:
        st.slider("Beamstop width [µm]", 5.0, 5000.0, step=5.0, key="dbs_width_um")

with sb.expander("Numerics and files", expanded=False):
    st.select_slider("Rays past the beamstop", [2000, 4000, 8000, 16000, 32000, 64000], key="n_rays")
    p_now = current_params()
    st.download_button("⬇ Save parameters (JSON)", json.dumps(sim.asdict(p_now), indent=1),
                       file_name="sdxtg_params.json", mime="application/json", **WIDE)
    up = st.file_uploader("Load parameters (JSON)", type="json")
    if up is not None and st.button("Apply loaded file", **WIDE):
        d = json.load(up)
        ss["pending"] = {**{k: v for k, v in d.get("params", d).items() if k in FIELDS}}
        pp = sim.Params(**{k: v for k, v in d.get("params", d).items() if k in FIELDS})
        ss["pending"].update(bar_phase_pi=pp.bar_phase_rad / math.pi, beam_fwhm=pp.beam_fwhm_x_mm,
                             focus_fwhm=pp.focus_fwhm_x_um)
        st.rerun()

p = current_params()
ss["_saved_params"] = {k: ss[k] for k in FIELDS + ["bar_phase_pi", "beam_fwhm", "focus_fwhm"] if k in ss}


@st.cache_data(show_spinner=False, max_entries=64)
def run(pdict):
    r = sim.simulate(sim.Params(**pdict))
    return r


res = run(sim.asdict(p))
m = res.metrics
ss["last_solved_period"] = res.p.period_nm
p = res.p                      # = sidebar parameters, with the grating period solved in target mode
_lo, _hi = sorted((p.tg_min_nm, p.tg_max_nm))
if p.period_mode == "target":
    _msg = (f"Grating period solved: **{res.p.period_nm:.1f} nm** (λ/p = {p.wavelength_nm / res.p.period_nm * 1e3:.1f} mrad). "
            f"Largest Λ possible with this beam split: {sim.natural_tg_nm(p, res.rays):.0f} nm.")
    if not _lo * (1 - p.tg_tol_pct / 100) <= p.tg_set_nm <= _hi * (1 + p.tg_tol_pct / 100):
        _msg += " ⚠️ The setpoint is outside the target range."
    tg_note.markdown(_msg)
else:
    tg_note.markdown(f"Λ with this period: **{m.get('tg_period_nm', float('nan')):.1f} nm**.")

# ─────────────────────────────────────────────────────────────────────────────
#  Header + KPIs
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# SD-XTG beam path simulator")
st.markdown(f"<div class='subtitle'>{'X-ray pump · optical probe' if p.experiment == 'optical' else 'Self-diffraction'} X-ray transient grating · λ = {p.wavelength_nm} nm · "
            f"{ss['case']} · split along <b>{p.split_axis}</b> · {p.grating_type} grating</div>",
            unsafe_allow_html=True)

if "z_sample" not in m:
    st.error("No usable configuration: " + "; ".join(res.warnings))
    st.stop()


def kpi(col, label, value, sub="", state="neutral"):
    col.markdown(f"<div class='kpi {state}'><div class='lab'>{label}</div><div class='val'>{value}</div>"
                 f"<div class='sub'>{sub}</div></div>", unsafe_allow_html=True)


cols = st.columns(4)
kpi(cols[0], "Sample", f"{m['z_sample']:.4f} m",
    f"{m['d_grating_sample_mm']:.1f} mm after grating · {m['sample_ov_margin_mm']:.1f} mm inside A/B overlap · "
    f"overlap {m['overlap']:.2f}",
    "good" if m["sample_ov_margin_mm"] > 0 and m["overlap"] > 0.5 else "bad")
kpi(cols[1], "TG period Λ · contrast", f"{m['tg_period_nm']:.1f} nm · {m['tg_contrast']:.2f}",
    f"target {sim.tg_range_text(p)}{'' if m['tg_in_range'] else ' — OUT OF RANGE'} · grating p = "
    f"{res.p.period_nm:.1f} nm · crossing {m['crossing_angle_mrad']:.1f} mrad",
    "good" if m["tg_in_range"] and m["tg_contrast"] > 0 else "bad")
kpi(cols[2], "Order separation at slit", f"{m['sep_slit_um']:.0f} µm",
    f"nearest {m['sep_nearest']} · unwanted through slit {m['contamination_pct']:.2g} %",
    "good" if m["sep_slit_um"] > 20 and m["contamination_pct"] < 1 else "bad")
if p.experiment == "optical":
    kpi(cols[3], "Optical probe", f"φ_B = {m['probe_bragg_deg']:.2f}°",
        f"{p.probe_wavelength_nm:g} nm probe at {m['probe_angle_used_deg']:.2f}° · η_rel {m['probe_eta_rel']:.2f} · "
        f"diffracted at {m['probe_diff_angle_deg']:.2f}° · acceptance {m['probe_accept_deg']:.1f}°",
        "good" if m["probe_ok"] and m["probe_eta_rel"] > 0.5 else "bad")
else:
    kpi(cols[3], "SD past direct-beam stop", f"{m['sd_clear_dbs_um']:.0f} µm",
        f"direct leak {m['direct_leak_pct']:.2g} % · stop {m['dbs_width_um_used']:.0f} µm wide",
        "good" if m["sd_clear_dbs_um"] > 20 else "bad")
st.write("")
cols = st.columns(4)
kpi(cols[0], "Beam angles to sample normal", f"{m['angle_A_mrad']:+.2f} / {m['angle_B_mrad']:+.2f} mrad",
    f"A(−1) / B(+1) = {m['angle_A_deg']:+.3f}° / {m['angle_B_deg']:+.3f}° · tilt {p.sample_tilt_mrad:g} mrad · "
    f"beams {m['fwhm_A_s_um']:.0f} × {m['fwhm_A_o_um']:.0f} µm")
kpi(cols[1], "A/B overlap region", f"{m['ov_len_mm']:.1f} mm",
    f"z = {m['z_ov0']:.4f} … {m['z_ov1']:.4f} m")
kpi(cols[2], "Peak fluence A+B", f"{m['fluence_sum']:.3g} mJ/cm²", f"{m['P_1st_total'] * 100:.2f} % of photons on KB")
if p.experiment == "optical":
    kpi(cols[3], "Probe signal (rel.)", f"{m['opt_fom']:.3g}",
        f"(contrast × fluence)² × η_rel · grating depth L = {m['probe_L_nm']:g} nm")
else:
    kpi(cols[3], "SD beam on the CCD", f"SD at ±{abs(m['det_SD_A_um']) / 1e3:.1f} mm",
        f"chip {p.ccd_width_mm:g} mm centred at {m['ccd_offset_used_mm']:.1f} mm · margin {m['sd_ccd_margin_mm']:.1f} mm · "
        f"direct beams at ±{abs(m['det_direct_A_um']) / 1e3:.1f} mm",
        "good" if m["sd_ccd_margin_mm"] > 0 else "bad")
if p.experiment == "optical":
    st.write("")
    cols = st.columns(4)
    kpi(cols[0], "SD past direct-beam stop", f"{m['sd_clear_dbs_um']:.0f} µm",
        f"direct leak {m['direct_leak_pct']:.2g} % · stop {m['dbs_width_um_used']:.0f} µm wide",
        "good" if m["sd_clear_dbs_um"] > 20 else "bad")
    kpi(cols[1], "SD beam on the CCD", f"SD at ±{abs(m['det_SD_A_um']) / 1e3:.1f} mm",
        f"chip {p.ccd_width_mm:g} mm centred at {m['ccd_offset_used_mm']:.1f} mm · margin {m['sd_ccd_margin_mm']:.1f} mm",
        "good" if m["sd_ccd_margin_mm"] > 0 else "bad")
    if "probe_mirror_x_mm" in m:
        kpi(cols[2], "Steering mirror", f"x = {m['probe_mirror_x_mm']:+.1f} mm",
            f"{(res.z['sample'] - m['probe_mirror_z']) * 1e3:.1f} mm before the sample, same height as the X-rays · "
            f"AOI {m['probe_mirror_aoi_deg']:.1f}° · clears X-rays by {m['probe_mirror_clear_mm']:.1f} mm",
            "good" if m["probe_mirror_clear_mm"] > 0 else "bad")
        sm = m["probe_slit_max_mm"]
        kpi(cols[3], "Probe through the slit gap", f"{m['probe_slit_T'] * 100:.0f} % passes",
            (f"slit must be ≤ {sm:.3f} mm before the sample for this probe; now {p.slit_upstream_mm:g} mm"
             if sm > 0 else "probe wider than the slit gap: use a smaller probe")
            + f" · {int(m['probe_clashes'])} obstacle(s)", "good" if m["probe_clashes"] == 0 else "bad")
for w in res.warnings:
    st.warning(w, icon="⚠️")
st.write("")

tab_bl, tab_3d, tab_sz, tab_pr, tab_an, tab_tg, tab_sc, tab_wv, tab_md = st.tabs(
    ["🛰️ Beamline", "🧊 3D view", "📏 Beam size", "🎯 Slit · sample · CCD", "📐 Angles · optical probe",
     "🧪 TG in the sample", "🗺️ Correlation scan", "🌊 Wave-optics check", "📘 Model & assumptions"])

with tab_3d:
    c1, c2, c3, c4 = st.columns([1.4, 1.4, 1, 1])
    reg = c1.radio("Region", ["full", "layout", "sample"], horizontal=True, key="v3d_region",
                   format_func=lambda v: {"full": "whole beamline", "layout": "grating → CCD (to scale)",
                                          "sample": "pump–probe overlap (close-up, to scale)"}[v])
    ex = c2.select_slider("Transverse exaggeration (x, y)", [1, 2, 5, 10, 20, 50, 100, 200, 500],
                          value=100 if reg == "full" else 1, key=f"v3d_exag_{reg}")
    orders = c3.toggle("Other orders", value=True, key="v3d_orders")
    rays_on = c4.toggle("Ray cloud", value=False, key="v3d_rays")
    f3d = s3d.figure_3d(res, reg, float(ex), orders, rays_on)
    # the saved camera belongs to one region and set of axes: a different scene starts from its default view
    _xr = f3d.layout.scene.xaxis.range or (0, 0)
    cam_key = f"v3_{reg}_{p.experiment}_{_xr[0]:.1f}_{_xr[1]:.1f}"
    if reg == "sample" and p.experiment != "optical":
        st.info("The optical probe is drawn only in the **X-ray pump · optical probe** experiment (sidebar).", icon="🔦")
    if ss.get("v3d_nav", True):
        if hasattr(st, "iframe"):                  # Streamlit >= 1.52
            st.iframe(s3d.html_3d(f3d, cam_key, 740), height=750)
        else:
            components.html(s3d.html_3d(f3d, cam_key, 740), height=750)
    else:
        st.plotly_chart(f3d, **WIDE)
    st.toggle("Zoom towards the cursor with Alt/⌥ + scroll", value=True, key="v3d_nav",
              help="Off: the plain Plotly chart (scroll always zooms towards the rotation centre).")
    st.caption("Drag to rotate, scroll to zoom towards the rotation centre, **Alt/⌥ + scroll to zoom towards the point "
               "under the cursor** (the rotation centre follows), right-drag to pan, double-click to reset. The view "
               "is kept when you change parameters. Each beam is a tube through "
               "its FWHM outline (hard edges where the beamstop or the slit cut it), the same outline as in the 2D "
               "views, from the same rays. With an exaggeration other than 1 the transverse axes are stretched and angles look "
               "steeper than they are; 'around the sample' at ×1 is to scale.")



# ─────────────────────────────────────────────────────────────────────────────
#  Beamline sketch
# ─────────────────────────────────────────────────────────────────────────────
def band(fig, z, lo, hi, color, alpha, name, row=None, showlegend=False, group=None):
    fig.add_trace(go.Scatter(x=np.r_[z, z[::-1]], y=np.r_[hi, lo[::-1]], fill="toself", mode="lines",
                             line=dict(width=0.6, color=rgba(color, min(alpha + 0.25, 1))),
                             fillcolor=rgba(color, alpha), name=name, hoverinfo="name",
                             showlegend=showlegend, legendgroup=group or name),
                  row=row, col=1 if row else None)


TAG_COL = {"full": C["full"], "A": C["A"], "B": C["B"], "other": C["other"]}


@st.cache_data(show_spinner=False, max_entries=32)
def sketch_data(pdict):
    r = sim.simulate(sim.Params(**pdict))
    return sim.envelopes(r), sim.sd_envelope(r)


env, sdenv = sketch_data(sim.asdict(p))
z = res.z
foc = "HM" if p.split_axis == "x" else "VM"


def elements(fig, row, sc, ymax, full):
    def vline(x, label, color=C["slit"], dash="dot", width=1):
        fig.add_shape(type="line", x0=x, x1=x, y0=-ymax, y1=ymax, line=dict(color=color, dash=dash, width=width),
                      row=row, col=1)
        fig.add_annotation(x=x, y=ymax, text=label, showarrow=False, yanchor="bottom", font=dict(size=11, color=color),
                           row=row, col=1)
    if full:
        vline(p.z_hm, "HM", C["ink"], "solid", 3)
        vline(p.z_vm, "VM", C["ink"], "solid", 3)
        h, o = p.bs_width_mm * 1e-3 / 2 * sc, p.bs_offset_mm * 1e-3 * sc
        fig.add_shape(type="rect", x0=p.z_bs - 0.006, x1=p.z_bs + 0.006, y0=o - h, y1=o + h,
                      fillcolor=C["ink"], line=dict(width=0), row=row, col=1)
        fig.add_annotation(x=p.z_bs, y=ymax, text="beamstop", showarrow=False, yanchor="bottom",
                           font=dict(size=11), row=row, col=1)
    vline(z["grating"], "grating", "#5B6B7F", "dash", 1.5)
    lo, hi = res.slit
    span = (z["det"] + 0.03) if full else (z["det"] - z["grating"]) * 1.1
    wz = 0.004 * span
    for y0, y1 in ((-ymax, lo * sc), (hi * sc, ymax)):
        fig.add_shape(type="rect", x0=z["slit"] - wz, x1=z["slit"], y0=y0, y1=y1,
                      fillcolor=C["slit"], line=dict(width=0), row=row, col=1)
    vline(z["sample"], "" if full else "sample", C["sample"], "solid", 2)
    b0, b1 = res.dbs
    fig.add_shape(type="rect", x0=z["dbs"] - wz / 2, x1=z["dbs"] + wz / 2, y0=b0 * sc, y1=b1 * sc,
                  fillcolor="#111827", line=dict(width=0), row=row, col=1)
    if not full:
        fig.add_annotation(x=z["slit"], y=-ymax * 0.96, text="slit ", showarrow=False, xanchor="right",
                           yanchor="bottom", font=dict(size=11, color=C["slit"]), row=row, col=1)
        fig.add_annotation(x=z["dbs"], y=b1 * sc, text="direct-beam stop", showarrow=False, yanchor="bottom",
                           yshift=4, font=dict(size=11), row=row, col=1)
    else:
        fig.add_annotation(x=z["sample"], y=ymax, text="slit · sample · stop", showarrow=False, yanchor="bottom",
                           xanchor="right", font=dict(size=11, color=C["sample"]), row=row, col=1)
    # CCD: the chip itself (side, centred where the sidebar puts it), with the camera plane dotted
    off, cw = m.get("ccd_offset_used_mm", p.ccd_offset_mm) * 1e-3, p.ccd_width_mm * 1e-3 / 2
    vline(z["det"], "CCD", "#94A3B8", "dot", 1)
    fig.add_shape(type="rect", x0=z["det"], x1=z["det"] + 1.5 * wz, y0=(off - cw) * sc, y1=(off + cw) * sc,
                  fillcolor="#1E3A5F", line=dict(color="#C9A227", width=1.5), row=row, col=1)


PROBE_ON = p.experiment == "optical" and "probe_mirror_x_mm" in m
GEO = sim.probe_geometry(res) if PROBE_ON else None


def probe_formula_md():
    """Phase-matching (Bragg) angle of the optical probe from the pump wavelength and crossing angle."""
    th2 = (m["angle_B_mrad"] - m["angle_A_mrad"]) / 2 * 1e-3
    lp, lx = p.probe_wavelength_nm, p.wavelength_nm
    sB = lp / lx * math.sin(th2)
    txt = (f"**Probe phase matching.** The pumps cross at θ = {2 * th2 * 1e3:.3f} mrad, so "
           f"Λ = λ_X / (2 sin(θ/2)) = {m['tg_surface_nm']:.1f} nm. The probe is Bragg-matched (in-plane wavevector "
           f"of the probe changed by exactly ±K) when **sin φ_B = λ_probe / (2Λ) = (λ_probe/λ_X)·sin(θ/2)** = "
           f"({lp:g} / {lx:g})·sin({th2 * 1e3:.3f} mrad) = {sB:.4f}")
    if sB < 1:
        txt += (f" → **φ_B = {math.degrees(math.asin(sB)):.2f}°** from the sample normal, in the horizontal plane; "
                f"inside the sample (n = {p.sample_n:g}) φ_in = {math.degrees(math.asin(sB / p.sample_n)):.2f}°. "
                f"The diffracted probe leaves at −φ_B. The probe in all drawings is set to "
                f"{m['probe_angle_used_deg']:.2f}°" + (" (Bragg angle)." if p.probe_auto_bragg else " (set by hand)."))
    else:
        txt += " > 1: no phase-matched probe at this wavelength (needs Λ > λ_probe/2)."
    return txt


def probe_at(zpos):
    """Optical probe beams crossing the plane z = zpos: (label, centre µm, FWHM across the plane µm, colour)."""
    if not PROBE_ON:
        return []
    S, w = GEO["S"], p.probe_fwhm_um()
    if abs(zpos - S[2]) < 1e-9:
        return [("optical probe (footprint)", S[0] * 1e6, w / GEO["d_in"][2], s3d.COL["probe"])]
    if zpos < S[2]:
        if zpos < GEO["mirror"][2]:
            return []
        d = GEO["d_in"]
        return [("optical probe (incoming)", (S[0] + d[0] * (zpos - S[2]) / d[2]) * 1e6, w / d[2], s3d.COL["probe"])]
    out = [("transmitted probe", (S[0] + GEO["d_tr"][0] * (zpos - S[2]) / GEO["d_tr"][2]) * 1e6, w / GEO["d_tr"][2],
            s3d.COL["probe"])]
    if GEO["d_df"] is not None:
        d = GEO["d_df"]
        out.append(("diffracted probe (signal)", (S[0] + d[0] * (zpos - S[2]) / d[2]) * 1e6, w / d[2], s3d.COL["probe_d"]))
    return out


with tab_bl:
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.13, row_heights=[0.45, 0.55],
                        subplot_titles=(f"Full beamline · split axis {p.split_axis} (focused by {foc})",
                                        "Zoom: grating → CCD"))
    seen = set()
    for name, tag, zz, lo, hi, wanted in env:
        col = TAG_COL[tag]
        if name.endswith("(blocked)"):
            alpha = 0.08
        else:
            alpha = 0.45 if tag in ("A", "B", "full") and wanted else 0.16
        lg = {"full": "full beam", "A": "beam A → A(−1)", "B": "beam B → B(+1)", "other": "other orders"}[tag]
        band(fig, zz, lo * 1e3, hi * 1e3, col, alpha, lg, row=1, showlegend=lg not in seen, group=lg)
        seen.add(lg)
        if zz[-1] >= z["grating"] - 1e-9:
            band(fig, zz, lo * 1e6, hi * 1e6, col, alpha, lg, row=2, group=lg)
    for name, zz, lo, hi in sdenv:
        band(fig, zz, lo * 1e6, hi * 1e6, C["SD"], 0.45, "self-diffracted (2kA−kB, 2kB−kA)", row=2,
             showlegend=name == "SD_A", group="SD")
    zr_, lr_, hr_ = sim.overlap_region(res)
    if len(zr_):
        fig.add_trace(go.Scatter(x=np.r_[zr_, zr_[::-1]], y=np.r_[hr_, lr_[::-1]] * 1e6, fill="toself", mode="lines",
                                 line=dict(color="#F2B705", width=2), fillcolor="rgba(242,183,5,0.35)",
                                 name="A/B overlap region (TG exists here)", hoverinfo="name"), row=2, col=1)
    ymax1 = 1.08 * max(max(np.max(np.abs(e[3])), np.max(np.abs(e[4]))) for e in env) * 1e3
    wanted_ext = [max(np.max(np.abs(e[3])), np.max(np.abs(e[4]))) for e in env
                  if e[5] and e[1] in ("A", "B") and e[2][0] >= z["grating"] - 1e-9]
    ymax2 = 1.08 * max([abs(m["det_SD_A_um"]) * 1e-6] + wanted_ext) * 1e6
    elements(fig, 1, 1e3, ymax1, True)
    elements(fig, 2, 1e6, ymax2, False)
    if PROBE_ON:
        # the optical probe at its lab position (the axes here are stretched ~1000:1, so a 26° beam is almost
        # vertical: it crosses this whole window within a few mm of z around the sample; to scale: close-up below)
        S_ = GEO["S"]
        for d_, sgn, col_, nm_ in ((GEO["d_in"], -1, s3d.COL["probe"], "optical probe (incoming)"),
                                   (GEO["d_tr"], 1, s3d.COL["probe"], "transmitted probe"),
                                   (GEO["d_df"], 1, s3d.COL["probe_d"], "diffracted probe (signal)")):
            if d_ is None or abs(d_[0]) < 1e-9:
                continue
            t_ = 1.2 * ymax2 * 1e-6 / abs(d_[0])            # path length to leave the window
            P1 = S_ + sgn * t_ * d_
            fig.add_trace(go.Scatter(x=[S_[2], P1[2]], y=[S_[0] * 1e6, P1[0] * 1e6], mode="lines", name=nm_,
                                     line=dict(color=col_, width=3, dash="dash" if nm_.startswith("trans") else None),
                                     legendgroup="probe"), row=2, col=1)
    x0 = z["grating"] - 0.06 * (z["det"] - z["grating"])
    fig.update_xaxes(range=[min(p.z_hm, p.z_vm) - 0.04, z["det"] + 0.03], row=1, col=1, **AX)
    fig.update_xaxes(range=[x0, z["det"] + 0.02 * (z["det"] - x0)], title="z from HM [m]", row=2, col=1, **AX)
    fig.update_yaxes(range=[-ymax1, ymax1], title=f"{p.split_axis} [mm]", row=1, col=1, **AX)
    fig.update_yaxes(range=[-ymax2, ymax2], title=f"{p.split_axis} [µm]", row=2, col=1, **AX)
    fig.update_layout(height=860, **LAYOUT)
    fig.update_layout(margin=dict(l=60, r=20, t=80, b=45), legend=dict(y=1.06))
    for an in fig.layout.annotations[:2]:          # subplot titles: left-aligned, above the labels
        an.update(x=0, xanchor="left", yshift=22, font=dict(size=14, color=C["ink"]))
    st.plotly_chart(fig, **WIDE)
    st.caption("Each half-beam is drawn as its FWHM band (11.95 % to 88.05 % quantiles, the same outline as in every other view), from the KB to the CCD, with the same rays "
               "throughout. The green outline before the beamstop meets their outer edges, and the parts of the full beam that "
               "become A and B are tinted. Yellow is the region where A(−1) and B(+1) overlap: the sample has to sit "
               "inside it. Grey bands are the orders the slit has to remove. "
               "The faint red and blue after the direct-beam stop show where the blocked direct beams would go. "
               "Purple shows the self-diffracted signal travelling to the CCD.")
    if PROBE_ON:
        st.markdown(probe_formula_md())
        st.plotly_chart(s3d.figure_pump_probe(res), key="pp_bl", **WIDE)
        st.caption("Close-up at true scale (equal axes): the pumps A(−1) and B(+1) (FWHM bands) cross at only "
                   f"±{abs(m['angle_A_mrad']):.2f} mrad, so they look parallel; the probe comes in at φ_B through the "
                   "gap of the order-selecting slit and overlaps them on the sample; the diffracted probe (signal) "
                   "leaves at −φ_B, the self-diffracted X-rays continue at ±1.5 θ. The long views above have "
                   "stretched transverse axes, where a 26° beam cannot be drawn at its real angle, so the probe is "
                   "shown here.")

# ─────────────────────────────────────────────────────────────────────────────
#  Beam size
# ─────────────────────────────────────────────────────────────────────────────
with tab_sz:
    v = sim.size_vs_z(res)
    o_ax = "y" if p.split_axis == "x" else "x"
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=v["z"], y=v["full_s"], name=f"full beam, {p.split_axis}", line=dict(color=C["full"], width=2.5)))
    f2.add_trace(go.Scatter(x=v["z"], y=v["full_o"], name=f"full beam, {o_ax}",
                            line=dict(color=C["full"], width=2.5, dash="dash")))
    f2.add_trace(go.Scatter(x=v["z"], y=v["A_s"], name=f"A → A(−1), {p.split_axis}", line=dict(color=C["A"], width=2.5)))
    f2.add_trace(go.Scatter(x=v["z"], y=v["B_s"], name=f"B → B(+1), {p.split_axis}",
                            line=dict(color=C["B"], width=2, dash="dot")))
    f2.add_trace(go.Scatter(x=v["z"], y=v["A_o"], name=f"A and B, {o_ax}", line=dict(color="#7C3AED", width=2, dash="dash")))
    for k, lab, pos in (("vm", "VM", "top"), ("bs", "beamstop", "top"), ("grating", "grating", "top"),
                        ("sample", "sample ", "bottom left"), ("dbs", " direct stop", "bottom right"),
                        ("det", "CCD", "top")):
        f2.add_vline(x=z[k], line=dict(color="#94A3B8", dash="dot", width=1),
                     annotation_text=lab, annotation_position=pos, annotation_font_size=11)
    f2.update_yaxes(type="log", title="FWHM [µm] (2.355 × rms)", **AX)
    f2.update_xaxes(title="z from HM [m]", **AX)
    f2.update_layout(height=470, **LAYOUT)
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(f2, **WIDE)
    rows = sim.size_table(res)
    names = dict(hm="HM", vm="VM", bs="Beamstop", grating="Grating", slit="Slit", sample="Sample",
                 dbs="Direct stop", det="CCD")
    tbl = []
    for r in rows:
        g = lambda key: f"{r[key]:.0f}" if key in r else "–"
        tbl.append({"Position": names[r["plane"]], "z [m]": f"{float(r['z_m']):.4f}",
                    f"Full {p.split_axis}": f"{r['full_s']:.0f}", f"Full {o_ax}": f"{r['full_o']:.0f}",
                    f"A/B {p.split_axis}": g("A_s"), f"A/B {o_ax}": g("A_o"), "A centre": g("A_centroid")})
    c2.markdown("##### Beam size at each element (µm FWHM)")
    c2.dataframe(tbl, hide_index=True, **WIDE)
    buf = io.StringIO()
    keys = list(tbl[0])
    buf.write(",".join(keys) + "\n")
    for r in tbl:
        buf.write(",".join(str(r[k]) for k in keys) + "\n")
    c2.download_button("⬇ Download table (CSV)", buf.getvalue(), "sdxtg_beam_sizes.csv", "text/csv")
    c2.caption("Full beam: the unsplit beam (shown everywhere for reference). A/B: half-beam A, then A(−1) after the "
               "grating (after the slit, only the rays that pass it). B is the exact mirror image of A, with the same size "
               "and the opposite centre. 'A centre' is the centroid along the split axis.")


# ─────────────────────────────────────────────────────────────────────────────
#  Profiles
# ─────────────────────────────────────────────────────────────────────────────
def profile_fig(zpos, bins, masked, title, shade=None, shade_inside=False, sd=False, height=360):
    R = res.rays
    f = go.Figure()
    pr = probe_at(zpos)
    if pr and abs(zpos - z["sample"]) > 1e-9:          # widen the window so the probe is in view (up to 5 mm)
        xs_ = [c for _, c, _, _ in pr if abs(c) < 5000]
        if xs_:
            lo_, hi_ = min(bins[0], min(xs_) - 150), max(bins[-1], max(xs_) + 150)
            bins = np.linspace(lo_, hi_, max(len(bins), 400))
    centers = 0.5 * (bins[1:] + bins[:-1])
    for b in res.bundles:
        k = b.idx[b.slit_mask] if masked else b.idx
        if len(k) < 5:
            continue
        x = R.s(zpos, k, b.order) * 1e6
        h, _ = np.histogram(x, bins, weights=np.full(len(k), b.eff / R.N))
        if h.max() <= 0:
            continue
        col = C[b.beam] if b.wanted else C["other"]
        f.add_trace(go.Scatter(x=centers, y=h, name=b.name, mode="lines",
                               line=dict(color=col, width=2.4 if b.wanted else 1.2, shape="spline"),
                               fill="tozeroy" if b.wanted else None, fillcolor=rgba(col, 0.15)))
    if sd:
        sA, sB = res.sd_slopes
        for b, extra, nm in [(bb, res.sd_extra[0] if bb.beam == "A" else res.sd_extra[1], "SD_" + bb.beam)
                             for bb in res.bundles if bb.wanted]:
            k = b.idx[b.slit_mask]
            x = (R.s(zpos, k, b.order) + extra * (zpos - z["sample"])) * 1e6
            h, _ = np.histogram(x, bins, weights=np.full(len(k), b.eff / R.N))
            f.add_trace(go.Scatter(x=centers, y=h, name=nm.replace("_", " "), mode="lines",
                                   line=dict(color=C["SD"], width=2.2, shape="spline"),
                                   fill="tozeroy", fillcolor=rgba(C["SD"], 0.12)))
    if pr:
        top = max([max(tr.y) for tr in f.data if tr.y is not None and len(tr.y)] + [1e-30])
        for i_, (lab, c, fw, col) in enumerate(pr):
            if abs(c) < 5000:
                xx = np.linspace(bins[0], bins[-1], 600)
                f.add_trace(go.Scatter(x=xx, y=top * np.exp(-4 * math.log(2) * ((xx - c) / fw) ** 2), name=lab,
                                       mode="lines", line=dict(color=col, width=2.2, dash="dash")))
            else:
                f.add_annotation(x=bins[0] if c < 0 else bins[-1], y=top * (0.55 - 0.15 * i_), text=f"{lab} at x = {c / 1e3:+.1f} mm",
                                 showarrow=True, ax=60 if c < 0 else -60, ay=0, font=dict(size=10, color=col),
                                 arrowcolor=col)
    if shade is not None:
        lo, hi = shade[0] * 1e6, shade[1] * 1e6
        if shade_inside:
            f.add_vrect(x0=lo, x1=hi, fillcolor="#111827", opacity=0.13, line_width=0)
        else:
            f.add_vrect(x0=bins[0], x1=lo, fillcolor=C["slit"], opacity=0.13, line_width=0)
            f.add_vrect(x0=hi, x1=bins[-1], fillcolor=C["slit"], opacity=0.13, line_width=0)
    f.update_layout(title=dict(text=title, font=dict(size=13)), height=height, **LAYOUT)
    f.update_layout(margin=dict(l=50, r=15, t=45, b=45),
                    legend=dict(orientation="v", x=1, xanchor="right", y=1, yanchor="top",
                                bgcolor="rgba(255,255,255,0.8)", font=dict(size=10)))
    f.update_xaxes(title=f"{p.split_axis} [µm]", **AX)
    f.update_yaxes(title="photons / bin (rel.)", showticklabels=False, **AX)
    return f


with tab_pr:
    c1, c2 = st.columns(2)
    lo, hi = res.slit[0] * 1e6, res.slit[1] * 1e6
    w = max(hi - lo, 1)
    c1.plotly_chart(profile_fig(z["slit"], np.linspace(lo - 1.6 * w, hi + 1.6 * w, 220), False,
                                f"At the slit (z = {z['slit']:.4f} m), before the blades: separation "
                                f"{m['sep_slit_um']:.0f} µm", shade=res.slit), **WIDE)
    ext = 2.6 * m["fwhm_A_s_um"]
    c2.plotly_chart(profile_fig(z["sample"], np.linspace(-ext, ext, 110), True,
                                f"At the sample (z = {z['sample']:.4f} m), after the slit: overlap "
                                f"{m['overlap']:.2f}, Λ = {m['tg_period_nm']:.0f} nm"), **WIDE)
    c3, c4 = st.columns(2)
    b0, b1 = res.dbs[0] * 1e6, res.dbs[1] * 1e6
    e3 = 1.4 * max(abs(m["det_SD_A_um"]) * (z["dbs"] - z["sample"]) / max(z["det"] - z["sample"], 1e-6) * 1.2,
                   b1 - b0, 150)
    c3.plotly_chart(profile_fig(z["dbs"], np.linspace(-e3, e3, 200), True,
                                f"At the direct-beam stop (z = {z['dbs']:.4f} m): SD clearance "
                                f"{m['sd_clear_dbs_um']:.0f} µm", shade=res.dbs, shade_inside=True, sd=True),
                    **WIDE)
    # CCD plane 2D
    f4 = go.Figure()
    rng = np.random.default_rng(0)
    for key, (x, y) in res.sd.items():
        kk = rng.choice(len(x), min(1500, len(x)), replace=False)
        col = C["SD"] if key.startswith("SD") else (C["A"] if key.startswith("A") else C["B"])
        label = key.replace("_", " ") if key.startswith("SD") else key + " direct (blocked)"
        f4.add_trace(go.Scattergl(x=x[kk] * 1e6, y=y[kk] * 1e6, mode="markers", name=label,
                                  marker=dict(size=3, color=col, opacity=0.25 if not key.startswith("SD") else 0.7)))
    f4.update_layout(title=dict(text=f"On the CCD (z = {z['det']:.3f} m): SD at ±{abs(m['det_SD_A_um']):.0f} µm, "
                                     f"direct at ±{abs(m['det_direct_A_um']):.0f} µm", font=dict(size=13)),
                     height=360, **LAYOUT)
    f4.update_layout(margin=dict(l=50, r=15, t=45, b=45),
                     legend=dict(orientation="h", y=-0.22, x=0, yanchor="top", font=dict(size=10)))
    f4.update_xaxes(title=None, **AX)
    f4.update_yaxes(title=f"{o_ax} [µm]", **AX)
    if PROBE_ON:
        for nm, (qx, qy) in GEO["on_ccd"].items():
            f4.add_annotation(x=1 if qx > 0 else 0, xref="paper", y=0.98 if qx > 0 else 0.90, yref="paper", text=f"{nm} probe at x = {qx:+.1f} mm →"
                              if qx > 0 else f"← {nm} probe at x = {qx:+.1f} mm", showarrow=False, xanchor="right"
                              if qx > 0 else "left", font=dict(size=10, color=s3d.COL["probe_d" if nm == "diffracted"
                                                                                       else "probe"]))
    c4.plotly_chart(f4, **WIDE)
    if PROBE_ON:
        # the whole CCD plane, to scale (mm): chip, X-ray spots and where the optical probes land
        lam_p = p.probe_wavelength_nm * 1e-9
        w0 = p.probe_waist_um * 1e-6
        zR = math.pi * w0 ** 2 / lam_p                          # Rayleigh range of the probe
        f5 = go.Figure()
        c0, cw = m.get("ccd_offset_used_mm", p.ccd_offset_mm), p.ccd_width_mm / 2
        f5.add_shape(type="rect", x0=c0 - cw, x1=c0 + cw, y0=-cw, y1=cw, line=dict(color="#334155", width=1.5),
                     fillcolor="rgba(51,65,85,0.10)")
        f5.add_annotation(x=c0, y=cw, text=f"CCD chip {p.ccd_width_mm:g} mm", showarrow=False, yshift=10,
                          font=dict(size=10, color="#334155"))
        for key, (x, y) in res.sd.items():
            col = C["SD"] if key.startswith("SD") else (C["A"] if key.startswith("A") else C["B"])
            f5.add_trace(go.Scatter(x=[float(np.mean(x)) * 1e3], y=[float(np.mean(y)) * 1e3], mode="markers",
                                    name=key.replace("_", " ") if key.startswith("SD") else key + " direct (blocked)",
                                    marker=dict(size=7, color=col, symbol="x" if not key.startswith("SD") else "circle")))
        th_ = np.linspace(0, 2 * np.pi, 80)
        for nm, (qx, qy) in GEO["on_ccd"].items():
            d = GEO["d_tr"] if nm == "transmitted" else GEO["d_df"]
            L = (p.z_det - GEO["S"][2]) / d[2]                   # path length sample → CCD plane
            wz = w0 * math.sqrt(1 + (L / zR) ** 2)               # probe radius there (Gaussian beam)
            fx, fy = 1.17741 * wz / d[2] * 1e3 / 2, 1.17741 * wz * 1e3 / 2   # FWHM/2 on the plane, mm
            col = s3d.COL["probe_d" if nm == "diffracted" else "probe"]
            f5.add_trace(go.Scatter(x=qx + fx * np.cos(th_), y=qy + fy * np.sin(th_), mode="lines", fill="toself",
                                    line=dict(color=col, width=1.5), fillcolor=rgba(col, 0.35),
                                    name=f"{nm} probe, FWHM {2 * fx * 1e3:.0f} µm"))
            f5.add_annotation(x=qx, y=qy, text=f"{nm} probe<br>x = {qx:+.1f} mm", showarrow=True, ay=-35,
                              font=dict(size=10, color=col), arrowcolor=col)
        xs_ = [c0 - cw, c0 + cw] + [q[0] for q in GEO["on_ccd"].values()]
        f5.update_layout(title=dict(text=f"CCD plane (z = {p.z_det:.3f} m), to scale: the optical probes miss the X-ray "
                                         f"chip and go to their own detector" if not any("CCD chip" in o for o in
                                         GEO["obstacles"]) else "CCD plane: a probe beam lands on the X-ray chip",
                                    font=dict(size=13)), height=330, **LAYOUT)
        f5.update_layout(margin=dict(l=50, r=15, t=45, b=45),
                         legend=dict(orientation="h", y=-0.25, x=0, yanchor="top", font=dict(size=10)))
        f5.update_xaxes(title=f"{p.split_axis} [mm]", range=[min(xs_) - 6, max(xs_) + 6], **AX)
        f5.update_yaxes(title=f"{o_ax} [mm]", scaleanchor="x", scaleratio=1, **AX)
        st.plotly_chart(f5, key="ccd_plane_probe", **WIDE)
        st.caption(f"Probe beams spread after the sample: Rayleigh range π w₀²/λ = {zR * 1e3:.1f} mm, so at the CCD plane "
                   f"the {p.probe_waist_um:g} µm waist has grown to the size drawn. The X-ray CCD sees only the SD beams; "
                   "the diffracted probe is detected with a photodiode outside the chip.")
    st.caption("Profiles along the split axis, with each order weighted by its grating efficiency. Shaded areas are the "
               "slit blades (top left) and the direct-beam stop (bottom left).")

# ─────────────────────────────────────────────────────────────────────────────
#  Correlation scan
# ─────────────────────────────────────────────────────────────────────────────
with tab_tg:
    t = sim.tg_in_sample(res)
    ev = t["ev_atom"] is not None
    cols = st.columns(4)
    kpi(cols[0], "Transient grating", f"Λ = {t['Ls']:.1f} nm", f"contrast V = {t['V']:.2f} · grating vector along "
        f"{p.split_axis} (horizontal), tilt {m['tg_tilt_mrad']:.2f} mrad from the surface")
    kpi(cols[1], "Depth of the grating", f"L_att = {t['La']:.0f} nm",
        f"{p.sample_formula} at {1239.842 / p.wavelength_nm:.0f} eV · L_eff = {t['L_eff']:.0f} nm · "
        f"{t['absorbed'] * 100:.1f} % absorbed in {t['d']:g} nm")
    kpi(cols[2], "Peak absorbed energy", f"{t['dose0'] * (1 + t['V']):.3g} J/cm³",
        (f"{t['ev_atom_peak'] * 1e3:.3g} meV/atom at the surface, fringe maxima" if ev else "formula needed for eV/atom"))
    kpi(cols[3], "Temperature jump (estimate)", f"{t['dT_peak']:.3g} K" if ev else "–",
        "Dulong–Petit, 3 k_B per atom, peak · fringe contrast in T equals V right after the pump")
    st.write("")
    unit = "meV/atom" if ev else "J/cm³"
    Z = t["ev_atom"] * 1e3 if ev else t["dose"]
    norm = st.radio("Colour scale", ["absolute", "per depth"], horizontal=True, key="tg_norm",
                    format_func=lambda v: "absolute" if v == "absolute" else "normalised at each depth (shows the fringes deep inside)")
    Zs, ut = (Z, unit) if norm == "absolute" else (Z / Z.max(axis=1, keepdims=True), "rel.")
    c1, c2 = st.columns([1.6, 1])
    fh = go.Figure(go.Heatmap(x=t["x"], y=t["z"], z=Zs, colorscale="Magma", colorbar=dict(title=ut, thickness=12),
                              hovertemplate="x %{x:.1f} nm<br>depth %{y:.1f} nm<br>%{z:.3g} " + ut + "<extra></extra>"))
    for zz, lab in ((t["La"], "L_att"), (t["d"], "back face")):
        if zz <= t["z"][-1]:
            fh.add_hline(y=zz, line=dict(color="white", dash="dot", width=1), annotation_text=lab,
                         annotation_font_color="white", annotation_position="bottom right")
    fh.update_layout(height=430, title=dict(text=f"Absorbed energy in the sample, cross-section in the {p.split_axis}–z "
                                                 "plane (X-rays enter from the top)", font=dict(size=13)),
                     **{k_: v_ for k_, v_ in LAYOUT.items() if k_ != "legend"})
    fh.update_xaxes(title=f"{p.split_axis} along the surface [nm]", **AX)
    fh.update_yaxes(title="depth z [nm]", autorange="reversed", **AX)
    c1.plotly_chart(fh, theme=None, **WIDE)
    prof_max = Z.max(axis=1)
    prof_min = Z.min(axis=1)
    fd = go.Figure()
    fd.add_trace(go.Scatter(x=prof_max, y=t["z"], name="fringe maxima", line=dict(color=C["A"], width=2)))
    fd.add_trace(go.Scatter(x=(prof_max + prof_min) / 2, y=t["z"], name="mean", line=dict(color=C["ink"], width=1.5, dash="dash")))
    fd.add_trace(go.Scatter(x=prof_min, y=t["z"], name="fringe minima", line=dict(color=C["B"], width=2)))
    fd.update_layout(height=430, title=dict(text="Depth profile", font=dict(size=13)), **LAYOUT)
    fd.update_xaxes(title=unit, **AX)
    fd.update_yaxes(title="depth z [nm]", autorange="reversed", **AX)
    c2.plotly_chart(fd, **WIDE)
    st.caption(f"Absorbed energy density = F/L_att · e^(−z/L_att) · (1 + V cos(K·r)) with F = {m['fluence_sum']:.3g} mJ/cm² "
               f"(peak fluence A+B), V = {t['V']:.2f} and L_att from the **Sample material** panel"
               + (f"; {t['n_atoms_cm3']:.3g} atoms/cm³ for {p.sample_formula} at {p.sample_density:g} g/cm³" if ev else "")
               + ". The temperature jump uses 3 k_B per atom (high-temperature limit), so it is an estimate; reflection "
               "losses are neglected (soft X-rays near normal incidence).")
    st.markdown(
        f"**What reads this grating.** Self-diffraction: the SD beams are generated over the same depth and must "
        f"leave through the back face, signal ∝ e^(−d/L_att)·L_eff² → relative factor **{m['sd_sample_rel']:.2g}** "
        f"(1 = membrane of L_att·ln 3 = {m['sd_best_thickness_nm']:.0f} nm); X-ray transmission "
        f"{m['sd_transmission']:.2g}; SD phase mismatch Δk_z = {m['sd_dkz_per_um']:.2f} µm⁻¹ → η = {m['sd_eta']:.2f}. "
        + (f"Optical probe: grating depth L_eff = {m['probe_L_nm']:.0f} nm, Klein parameter Q = {m['probe_Q']:.2f} "
           f"({m['probe_regime']})." if p.experiment == "optical" else ""))


with tab_an:
    tilt = p.sample_tilt_mrad * 1e-3
    kX = 2 * math.pi / (p.wavelength_nm * 1e-9)
    bA, bB = m["angle_A_mrad"] * 1e-3, m["angle_B_mrad"] * 1e-3
    Kx, Kz = kX * (math.sin(bB) - math.sin(bA)), kX * (math.cos(bB) - math.cos(bA))
    st.markdown(
        f"**Pump beams at the sample** (angles from the sample normal, in the split plane, + towards +{p.split_axis}):\n\n"
        f"| | angle [mrad] | angle [deg] |\n|---|---|---|\n"
        f"| A(−1) | {m['angle_A_mrad']:+.4f} | {m['angle_A_deg']:+.5f} |\n"
        f"| B(+1) | {m['angle_B_mrad']:+.4f} | {m['angle_B_deg']:+.5f} |\n"
        f"| crossing θ = B − A | {m['crossing_angle_mrad']:.4f} | {math.degrees(m['crossing_angle_mrad'] * 1e-3):.5f} |\n"
        f"| bisector (tilt of the pair) | {m['bisector_mrad']:+.4f} | {math.degrees(m['bisector_mrad'] * 1e-3):+.5f} |\n\n"
        f"Λ = λ / (2 sin(θ/2)) = {p.wavelength_nm} nm / (2 sin({m['crossing_angle_mrad'] / 2:.4f} mrad)) = "
        f"**{m['tg_period_nm']:.2f} nm**. Period along the sample surface {m['tg_surface_nm']:.2f} nm; grating vector "
        f"tilted {m['tg_tilt_mrad']:.3f} mrad from the surface (0 when the sample normal bisects A and B). "
        f"Each beam's angle comes from the KB convergence (A and B head for the focus) plus the grating kick "
        f"λ/p = {p.wavelength_nm / p.period_nm * 1e3:.3f} mrad.")
    st.markdown("---")
    lam_p = p.probe_wavelength_nm
    if p.experiment != "optical":
        st.info("Switch *Experiment* in the sidebar to **X-ray pump · optical probe** to set the probe and see its "
                "phase-matching curve here.", icon="🔦")
    else:
        L_nm = m["probe_L_nm"]
        st.markdown(
            f"**Optical probe, {lam_p:g} nm, transmission.** The probe's in-plane wavevector picks up ±K_x = ±2π/Λ_surface "
            f"exactly; the mismatch along the normal, Δk_z = k_pz ± K_z − k_dz (inside the sample, n = {p.sample_n:g}), "
            f"sets the efficiency. The TG amplitude decays into the sample as e^(−z/L_att) (X-ray attenuation length "
            f"L_att = {p.xray_atten_nm:g} nm) over the thickness d = {p.sample_thickness_nm:g} nm, so "
            f"η_rel = |∫₀ᵈ e^(−z/L_att) e^(iΔk_z z) dz|² / |∫₀ᵈ e^(−z/L_att) dz|²: **sinc²(Δk_z d/2) for a uniform slab "
            f"(d ≪ L_att), a Lorentzian 1/(1 + Δk_z² L_att²) when the pump is absorbed (d ≫ L_att)**. "
            f"Effective depth {L_nm:.0f} nm, Klein parameter Q = 2πλL/(nΛ²) = {m['probe_Q']:.2f}: {m['probe_regime']}. "
            f"The probe travels in the horizontal plane of the X-ray beams, so it diffracts in the same plane as the "
            f"self-diffracted X-rays. Bragg angle sin φ_B = λ_probe/(2Λ): "
            f"**φ_B = {m['probe_bragg_deg']:.3f}°**. A diffracted probe exists only "
            f"for Λ > λ_probe/2 = {lam_p / 2:g} nm.")
        if m["probe_ok"]:
            acc = m["probe_accept_deg"]
            half = 90.0 if not np.isfinite(acc) or acc > 30 else max(4 * acc, 2.0)
            ctr = m["probe_bragg_deg"] if np.isfinite(m["probe_bragg_deg"]) else m["probe_angle_used_deg"]
            lo_, hi_ = max(ctr - half, -89.5), min(ctr + half, 89.5)
            ph = np.linspace(lo_, hi_, 3001)
            cur = sim.probe_curve(p, Kx, Kz, ph)
            f = go.Figure()
            for mo, col in ((-1, C["A"]), (1, C["B"])):
                f.add_trace(go.Scatter(x=ph, y=cur[mo][0], name=f"diffracted order m = {mo:+d}",
                                       line=dict(color=col, width=2)))
            for xv, lab, col in ((m["probe_bragg_deg"], "Bragg", C["good"]), (-m["probe_bragg_deg"], "−Bragg", C["good"]),
                                 (m["probe_angle_used_deg"], "probe", C["ink"])):
                if lo_ <= xv <= hi_:
                    f.add_vline(x=xv, line=dict(color=col, width=1.4, dash="dot"), annotation_text=lab)
            f.update_layout(title=dict(text=f"Phase matching vs probe angle: FWHM {acc:.2f}°", font=dict(size=13)),
                            height=400, **LAYOUT)
            f.update_layout(margin=dict(l=60, r=20, t=50, b=90), legend=dict(orientation="h", y=-0.22, yanchor="top"))
            f.update_xaxes(title="probe incidence angle from the sample normal [deg]", range=[lo_, hi_], **AX)
            f.update_yaxes(title="η_rel (phase-matching factor)", range=[0, 1.05], **AX)
            c1, c2 = st.columns(2)
            c1.plotly_chart(f, **WIDE)
            # the shape itself, against the mismatch: this depth profile vs a uniform slab of the same depth
            dk = np.linspace(-6 * math.pi, 6 * math.pi, 1201) / (L_nm * 1e-9)
            eta_p, _ = sim.depth_factor(dk, p.xray_atten_nm * 1e-9, p.sample_thickness_nm * 1e-9)
            fs = go.Figure()
            fs.add_trace(go.Scatter(x=dk * L_nm * 1e-9 / 2, y=np.sinc(dk * L_nm * 1e-9 / 2 / math.pi) ** 2,
                                    name=f"uniform slab, d = {L_nm:.0f} nm: sinc²", line=dict(color=C["muted"], dash="dash")))
            fs.add_trace(go.Scatter(x=dk * L_nm * 1e-9 / 2, y=eta_p, name="this sample (e^(−z/L_att) over d)",
                                    line=dict(color=C["SD"], width=2.5)))
            # the mismatch that the physical angle range (−90°…90°) can actually reach
            full = np.linspace(-89.5, 89.5, 3581)
            k0p = 2 * math.pi / (lam_p * 1e-9)
            kk = p.sample_n * k0p
            kpx = k0p * np.sin(np.radians(full))
            mo_ = -1 if Kx > 0 else 1
            kdx = kpx + mo_ * Kx
            okk = np.abs(kdx) < kk
            dkr = (np.sqrt(kk ** 2 - kpx ** 2) + mo_ * Kz - np.sqrt(np.clip(kk ** 2 - kdx ** 2, 0, None)))[okk]
            if dkr.size:
                fs.add_vrect(x0=max(dkr.min() * L_nm * 1e-9 / 2, -3 * math.pi), x1=min(dkr.max() * L_nm * 1e-9 / 2, 3 * math.pi),
                             fillcolor=C["good"], opacity=0.08, line_width=0,
                             annotation_text="reachable by tilting the probe", annotation_position="top left")
            fs.update_layout(title=dict(text="Phase-matching shape vs mismatch Δk_z·L_eff/2", font=dict(size=13)),
                             height=400, **LAYOUT)
            fs.update_layout(margin=dict(l=60, r=20, t=50, b=90), legend=dict(orientation="h", y=-0.22, yanchor="top"))
            fs.update_xaxes(title="Δk_z · L_eff / 2  [rad]", range=[-3 * math.pi, 3 * math.pi], **AX)
            fs.update_yaxes(title="η_rel", range=[0, 1.05], **AX)
            c2.plotly_chart(fs, **WIDE)
            st.caption("The green band is the part of the curve the probe can reach between −90° and +90°. "
                       "A sinc² with visible side lobes needs the grating to be deep and uniform: d ≲ L_att and "
                       "d ≳ 10 Λ (acceptance inside the sample ≈ 0.9 Λ/d). With a soft X-ray pump the grating is only "
                       "L_att deep, the curve is a broad Lorentzian and phase matching is not critical.")
            g = sim.probe_geometry(res)
            st.markdown("**Probe layout.** The steering mirror sits on the probe line, "
                        f"{p.probe_mirror_dist_mm:g} mm before the sample; it is fed "
                        + ("along the beamline (+z)." if p.probe_feed == "z" else "from the side.")
                        + f" Mirror centre at x = {m['probe_mirror_x_mm']:+.1f} mm, y = {m['probe_mirror_y_mm']:+.1f} mm, "
                        f"z = {m['probe_mirror_z']:.4f} m ({(z_s_ := res.z['sample'] - m['probe_mirror_z']) * 1e3:.1f} mm "
                        f"before the sample); angle of incidence on the mirror {m['probe_mirror_aoi_deg']:.1f}°; "
                        f"clearance from the X-ray beams {m['probe_mirror_clear_mm']:.1f} mm.")
            if g["obstacles"]:
                for o in g["obstacles"]:
                    st.warning("Probe: " + o, icon="🚧")
            else:
                st.success("The probe path is clear: no slit blade, grating chip or beamstop in the way, and the "
                           "mirror clears the X-ray beams.", icon="✅")
            st.markdown(probe_formula_md())
            c1, c2 = st.columns(2)
            c1.plotly_chart(s3d.figure_pump_probe(res), key="pp_an", **WIDE)
            c2.plotly_chart(s3d.figure_probe_2d(res, "xz"), **WIDE)
            # Bragg angle and acceptance vs TG period, for orientation
            tgs = np.geomspace(max(lam_p / 2 * 1.001, 5), max(lam_p * 20, m["tg_period_nm"] * 2), 300)
            fb = go.Figure(go.Scatter(x=tgs, y=np.degrees(np.arcsin(lam_p / (2 * tgs))), line=dict(color=C["SD"], width=2),
                                      name="φ_B"))
            fb.add_trace(go.Scatter(x=[m["tg_period_nm"]] * 2, y=[0, 90], mode="lines+text", text=["", "this Λ"],
                                    textposition="top right", line=dict(color=C["ink"], dash="dot", width=1.4)))
            fb.update_layout(title=dict(text=f"Bragg angle of a {lam_p:g} nm probe vs TG period", font=dict(size=13)),
                             height=320, showlegend=False, **{k_: v_ for k_, v_ in LAYOUT.items() if k_ != "legend"})
            fb.update_xaxes(title="Λ [nm]", type="log", **AX)
            fb.update_yaxes(title="φ_B [deg]", **AX)
            st.plotly_chart(fb, **WIDE)
            st.caption(f"Probe at {m['probe_angle_used_deg']:.3f}° → η_rel = {m['probe_eta_rel']:.3f}; the stronger "
                       f"diffracted order m = {m['probe_order']:+d} leaves at {m['probe_diff_angle_deg']:.3f}°. "
                       "Thin grating (L ≪ Λ²n/λ) = broad curve (Raman–Nath); thick = narrow (Bragg).")
        else:
            st.error(f"Λ = {m['tg_period_nm']:.1f} nm is below λ_probe/2 = {lam_p / 2:g} nm: no diffracted probe order. "
                     "Raise the TG target range or use a shorter probe wavelength.", icon="⛔")


def do_scan(pdict, zr, yv, kind, nz, rays):
    bar = st.progress(0.0, text="Running correlation scan …")
    maps = sim.scan(sim.Params(**pdict), np.linspace(*zr, nz), yv, kind=kind, n_rays=rays,
                    progress=lambda f: bar.progress(f, text=f"Running correlation scan … {f * 100:.0f} %"))
    bar.empty()
    return maps


def apply_point(kind, zz, yy):
    d = dict(z_grating=round(float(zz), 4))
    if kind == "tg":
        d.update(period_mode="target", tg_set_nm=round(float(yy), 2))
    elif kind == "period":
        d.update(period_mode="manual", period_nm=round(float(yy), 1))
    else:
        d.update(bs_width_mm=round(float(yy), 2))
    return d


def log_ticks(lo, hi):
    t = [m_ * 10 ** e for e in range(0, 5) for m_ in (1, 2, 5)]
    return [v for v in t if lo <= v <= hi]


with tab_sc:
    st.markdown("Each map below is a grid of full simulations over grating position × a second parameter. "
                "**Every other parameter is taken from the sidebar as it is now.** Settings whose TG period is "
                "outside the target range in the sidebar are **discarded** (grey). Click any pixel to apply it.")
    degenerate = math.isclose(_lo, _hi)
    kinds = (["bs_width", "period"] if degenerate else ["tg", "period", "bs_width"])
    kind_txt = {"tg": "TG period Λ over the target range (grating period solved)",
                "period": "grating period (Λ follows; out-of-range points discarded)",
                "bs_width": f"beamstop width (grating period solved for Λ = {p.tg_set_nm:g} nm)"}
    kind = st.radio("Second axis", kinds, format_func=lambda k: kind_txt[k], horizontal=True,
                    key=f"scan_kind_{int(degenerate)}")
    cc = st.columns([2.2, 2.2, 1, 1, 1.2, 1.2])
    zr = cc[0].slider("Grating z range [m]", 0.6, 1.995, (max(round(p.z_bs + 0.02, 3), 1.0), 1.99), step=0.005)
    if kind == "tg":
        cc[1].markdown(f"**Λ range:** {_lo:g}–{_hi:g} nm (log grid), from the sidebar target.")
        y_lo, y_hi = _lo, _hi
    elif kind == "period":
        p_lo = sim.solve_period(p, _lo, res.rays)
        p_hi = sim.solve_period(p, _hi, res.rays)
        d_lo = float(np.round(p_lo * 0.7 if np.isfinite(p_lo) else 20.0, 0))
        d_hi = float(np.round(p_hi * 1.4 if np.isfinite(p_hi) else 5000.0, 0))
        y_lo, y_hi = cc[1].slider("Grating period range [nm] (log grid)", 5.0, 20000.0,
                                  (max(d_lo, 5.0), min(max(d_hi, d_lo + 10), 20000.0)), step=1.0)
    else:
        D = p.beam_fwhm(p.split_axis)
        y_lo, y_hi = cc[1].slider("Beamstop width range [mm]", 0.0, 8.0,
                                  (round(0.3 * D, 2), round(min(1.6 * D, 8.0), 2)), step=0.01)
    nz = cc[2].number_input("z points", 10, 250, 90, step=10)
    npp = cc[3].number_input("y points", 10, 250, 90, step=10)
    rays = cc[4].select_slider("rays / sim", [1000, 2000, 3000, 5000, 8000], value=3000)
    cc[5].write("")
    go_btn = cc[5].button("▶ Run scan", type="primary", **WIDE)
    chosen = st.multiselect("Maps to show", sim.SCAN_KEYS, default=sim.default_maps(p),
                            format_func=lambda k: sim.SCAN_INFO[k][0])
    st.caption(f"{nz} × {npp} = {nz * npp} simulations, about {nz * npp * rays / 3000 * 0.0065:.0f} s on 4 cores.")
    if go_btn:
        ylog = sim.Y_KINDS[kind][2]
        yv = np.geomspace(y_lo, y_hi, int(npp)) if ylog else np.linspace(y_lo, y_hi, int(npp))
        ss["scan"] = (do_scan(sim.asdict(p), zr, yv, kind, int(nz), int(rays)), sim.asdict(p))
    if "scan" in ss:
        maps, pd_scan = ss["scan"]
        ps = sim.Params(**pd_scan)
        mk = maps.get("kind", "period")
        ylab, yunit, ylog = sim.Y_KINDS[mk]
        skip = {"z_grating", "period_nm", "n_rays", "z_sample", "tg_set_nm", "period_mode"}
        if mk == "bs_width":
            skip.add("bs_width_mm")
        changed = [k for k in FIELDS if k not in skip and pd_scan[k] != getattr(p, k)]
        if changed:
            st.info("The sidebar has changed since this scan (" + ", ".join(changed[:6]) +
                    ("…" if len(changed) > 6 else "") + "). Press *Run scan* to update it.", icon="ℹ️")
        ok = sim.feasible_mask(maps)
        zg, yv = maps["z_grating"], maps["y"]
        nd = int(maps["discarded"].sum())
        if ok.any():
            fk = "opt_fom" if maps.get("experiment") == "optical" else "sd_fom"
            fom = np.where(ok, maps[fk], np.nan)
            j, i = np.unravel_index(np.nanargmax(fom), fom.shape)
            b1, b2 = st.columns([4, 1])
            b1.success(f"Feasible region: {ok.sum()} of {ok.size} settings ({nd} discarded for Λ outside the target). "
                       f"Highest {sim.SCAN_INFO[fk][0]} at grating z = {zg[i]:.4f} m, {ylab} {yv[j]:.4g} {yunit}: "
                       f"Λ = {maps['tg_period_nm'][j, i]:.1f} nm, grating period {maps['grating_period_nm'][j, i]:.1f} nm, "
                       f"fluence {maps['fluence_sum'][j, i]:.3g} mJ/cm², overlap {maps['overlap'][j, i]:.2f}.",
                       icon="✅")
            if b2.button("Apply best point", **WIDE):
                ss["pending"] = apply_point(mk, zg[i], yv[j])
                st.rerun()
        else:
            st.error(f"No feasible setting in this range ({nd} of {ok.size} discarded for Λ outside the target): "
                     + ("Λ in target, Λ > λ_probe/2, separation > 20 µm, unwanted light < 1 %, overlap > 0.5 and "
                        "sample inside the A/B overlap are never all met. " if maps.get("experiment") == "optical" else
                        "Λ in target, separation > 20 µm, unwanted light < 1 %, overlap > 0.5, SD clearance > 20 µm, "
                        "sample inside the A/B overlap and SD beam on the CCD are never all met. ") +
                     "Look at the maps to see which one fails.", icon="⛔")
        st.caption(sim.scan_caption(ps, maps))
        Zg, Yy = np.meshgrid(zg, yv)
        ycur = sim.y_current(p, mk)
        cols = st.columns(3)
        for n, k in enumerate(chosen):
            A = maps[k]
            title, unit, desc = sim.SCAN_INFO[k]
            logc = k in ("contamination_pct", "sd_fom", "fluence_sum", "P_1st_total", "opt_fom")
            if logc:                                   # zeros (e.g. no leakage) at the bottom of the scale
                pos = A[np.isfinite(A) & (A > 0)]
                floor = max(pos.min(), pos.max() * 1e-4) if pos.size else 1e-6
                Z = np.log10(np.where(np.isfinite(A), np.maximum(A, floor), np.nan))
            else:
                Z = A
            cs, zmid = "Viridis", None
            if k in ("sep_slit_um", "sd_clear_dbs_um", "sample_ov_margin_mm", "sd_ccd_margin_mm"):
                fin = A[np.isfinite(A)]
                if fin.size and fin.min() < 0 < fin.max():
                    cs, zmid = "RdBu", 0
            if k == "contamination_pct":
                cs = "Magma_r"
            cbar = dict(title=dict(text=("log₁₀ " if logc else "") + unit, side="right"), thickness=12)
            hm = go.Heatmap(x=zg, y=yv, z=Z, colorscale=cs, zmid=zmid, colorbar=cbar, hoverinfo="skip")
            f = go.Figure(hm)
            if ok.any() and not ok.all():
                f.add_trace(go.Contour(x=zg, y=yv, z=ok.astype(float), contours=dict(start=0.5, end=0.5, size=1,
                                       coloring="none"), line=dict(color="white", width=3), showscale=False,
                                       hoverinfo="skip"))
                f.add_trace(go.Contour(x=zg, y=yv, z=ok.astype(float), contours=dict(start=0.5, end=0.5, size=1,
                                       coloring="none"), line=dict(color="black", width=1.2), showscale=False,
                                       hoverinfo="skip"))
            f.add_trace(go.Scatter(x=[p.z_grating], y=[ycur], mode="markers", hoverinfo="skip",
                                   marker=dict(symbol="star", size=15, color="gold", line=dict(color="black", width=1))))
            # invisible grid of points so that a click returns (z, y)
            f.add_trace(go.Scatter(x=Zg.ravel(), y=Yy.ravel(), mode="markers", name="pick",
                                   customdata=np.stack([A.ravel(), maps["grating_period_nm"].ravel(),
                                                        maps["tg_period_nm"].ravel()], axis=1),
                                   marker=dict(size=7, opacity=0.0),
                                   hovertemplate="z = %{x:.4f} m<br>" + ylab + " = %{y:.4g} " + yunit
                                                 + "<br>Λ = %{customdata[2]:.1f} nm, p = %{customdata[1]:.1f} nm<br>"
                                                 + title + " = %{customdata[0]:.4g} " + unit
                                                 + "<br><i>click to apply</i><extra></extra>"))
            f.update_layout(title=dict(text=f"<b>({chr(97 + n)}) {title}</b>", font=dict(size=13)), height=380,
                            showlegend=False, clickmode="event+select", **{k2: v2 for k2, v2 in LAYOUT.items()
                                                                           if k2 != "legend"})
            f.update_layout(margin=dict(l=55, r=10, t=40, b=45), plot_bgcolor="#D9DDE3")
            dz = (zg[-1] - zg[0]) / max(len(zg) - 1, 1) / 2
            f.update_xaxes(title="grating z [m]", range=[zg[0] - dz, zg[-1] + dz], dtick=0.1, **AX)
            if ylog:
                ticks = log_ticks(yv[0], yv[-1])
                f.update_yaxes(title=f"{ylab} [{yunit}]", type="log", tickvals=ticks,
                               ticktext=[f"{t:g}" for t in ticks],
                               range=[math.log10(yv[0]) - 0.01, math.log10(yv[-1]) + 0.01], **AX)
            else:
                f.update_yaxes(title=f"{ylab} [{yunit}]", **AX)
            with cols[n % 3]:
                if CAN_SELECT:
                    ev = st.plotly_chart(f, **WIDE, on_select="rerun", selection_mode="points",
                                         key=f"map_{k}_{ss.get('map_nonce', 0)}")
                else:
                    ev = None
                    st.plotly_chart(f, **WIDE, key=f"map_{k}_{ss.get('map_nonce', 0)}")
                st.markdown(f"<div class='mapdesc'>{desc}</div>", unsafe_allow_html=True)
                pts = (ev or {}).get("selection", {}).get("points", []) if ev else []
                if pts:
                    pt = pts[0]
                    # apply once, then give the charts fresh keys so the click is not re-applied later
                    ss["pending"] = apply_point(mk, pt["x"], pt["y"])
                    ss["map_nonce"] = ss.get("map_nonce", 0) + 1
                    st.rerun()
        lines = [f"z_grating_m,{mk}_{yunit},discarded," + ",".join(sim.SCAN_KEYS)]
        for j in range(len(yv)):
            for i in range(len(zg)):
                lines.append(f"{zg[i]:.4f},{yv[j]:.4g},{int(maps['discarded'][j, i])}," +
                             ",".join(f"{maps[k][j, i]:.5g}" for k in sim.SCAN_KEYS))
        st.download_button("⬇ Download scan (CSV)", "\n".join(lines), "sdxtg_scan.csv", "text/csv")

# ─────────────────────────────────────────────────────────────────────────────
#  Wave-optics check
# ─────────────────────────────────────────────────────────────────────────────
with tab_wv:
    st.markdown("The ray model above is geometric. This tab propagates the actual complex field along the split axis "
                "with the **Fresnel diffraction integral** (scaled for the converging beam, each grating order carried as "
                "envelope × exact carrier so periods of tens of nm work), so it includes edge "
                "diffraction from the beamstop and slit, partial coherence (Gaussian Schell model), bandwidth, and the real "
                "interference of A(−1) and B(+1), which is the transient grating itself. Use it to check the ray "
                "numbers for the current settings.")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.2])
    nxi = c1.select_slider("source points (coherence)", [1, 5, 9, 15], value=9)
    nl = c2.select_slider("wavelengths (bandwidth)", [1, 3, 5, 9], value=5)
    car_on = c3.toggle("x–z intensity map", value=True)
    c4.write("")
    run_w = c4.button("▶ Run wave-optics check", type="primary", **WIDE)
    if run_w:
        bar = st.progress(0.0, text="Propagating …")
        try:
            W = wave.wave_check(p, res, n_xi=nxi, n_lam=nl, carpet=car_on,
                                progress=lambda f: bar.progress(min(f, 1.0), text=f"Propagating … {f * 100:.0f} %"))
            ss["wave"] = (W, sim.asdict(p))
        except Exception as e:                                      # noqa: BLE001
            st.error(f"Wave check not possible for this setting: {e}")
        bar.empty()
    if "wave" in ss:
        W, pd_w = ss["wave"]
        if pd_w != sim.asdict(p):
            st.info("The parameters have changed since this wave check. Press *Run* to update it.", icon="ℹ️")
        rw = run(pd_w)
        mw = rw.metrics
        cols = st.columns(4)
        kpi(cols[0], "TG contrast: wave", f"{W.visibility:.2f}", "partially coherent, with bandwidth",
            "good" if W.visibility > 0.3 else "bad")
        kpi(cols[1], "TG contrast: ray model", f"{mw['tg_contrast']:.2f}",
            f"A–B coherence {mw['coherence_AB']:.2f} · overlap factor", "neutral")
        kpi(cols[2], "Fully coherent contrast", f"{W.visibility_coh:.2f}", "one source point, central λ")
        kpi(cols[3], "Λ: wave / ray", f"{W.tg_period_nm:.2f} / {W.info['tg_ray_nm']:.2f} nm",
            f"envelope grid {W.info['N_grating']:,} pts · sample grid {W.info['N_sample']:,} pts, dx "
            f"{W.info['dx_sample_nm']:.1f} nm · {W.info['n_xi']}×{W.info['n_lam']} components")
        st.write("")

        def compare(plane, zpos, masked, title):
            xw, Iw = W.planes[plane]
            fwhm = mw["fwhm_A_s_um"] * 1e-6
            if plane == "slit":
                lo, hi = rw.slit
                ext = 1.6 * (hi - lo)
                xmin, xmax = lo - ext, hi + ext
            else:
                xmin, xmax = -2.6 * fwhm, 2.6 * fwhm
            edges = np.linspace(xmin, xmax, 161)
            h, _ = np.histogram(xw, edges, weights=Iw)
            c, _ = np.histogram(xw, edges)
            Iwb = np.where(c > 0, h / np.maximum(c, 1), 0)              # averaged over the TG fringes
            xr, Ir = wave.ray_profile(rw, zpos, edges, masked)
            cx = 0.5 * (edges[1:] + edges[:-1]) * 1e6
            f = go.Figure()
            f.add_trace(go.Scatter(x=cx, y=Iwb / max(Iwb.max(), 1e-30), name="wave optics (fringe-averaged)",
                                   line=dict(color=C["ink"], width=2)))
            f.add_trace(go.Scatter(x=cx, y=Ir / max(Ir.max(), 1e-30), name="ray model",
                                   line=dict(color=C["B"], width=2, dash="dash")))
            if plane == "slit":
                f.add_vrect(x0=cx[0], x1=rw.slit[0] * 1e6, fillcolor=C["slit"], opacity=0.12, line_width=0)
                f.add_vrect(x0=rw.slit[1] * 1e6, x1=cx[-1], fillcolor=C["slit"], opacity=0.12, line_width=0)
            f.update_layout(title=dict(text=title, font=dict(size=13)), height=340, **LAYOUT)
            f.update_layout(margin=dict(l=50, r=15, t=45, b=45),
                            legend=dict(orientation="v", x=1, xanchor="right", y=1, bgcolor="rgba(255,255,255,0.8)"))
            f.update_xaxes(title=f"{p.split_axis} [µm]", **AX)
            f.update_yaxes(title="intensity (norm.)", **AX)
            return f

        c1, c2 = st.columns(2)
        c1.plotly_chart(compare("slit", rw.z["slit"], False, "At the slit, before the blades: wave vs ray"), **WIDE)
        c2.plotly_chart(compare("sample", rw.z["sample"], True, "At the sample, after the slit: wave vs ray"), **WIDE)
        c1, c2 = st.columns(2)
        xw, Iw = W.planes["sample"]
        win = 6 * W.tg_period_nm * 1e-9
        sel = np.abs(xw) < win
        f = go.Figure(go.Scatter(x=xw[sel] * 1e9, y=Iw[sel] / Iw[sel].max(), mode="lines",
                                 line=dict(color=C["sample"], width=2), name="I(x)"))
        f.update_layout(title=dict(text=f"TG fringes at the sample centre: Λ = {W.tg_period_nm:.0f} nm, "
                                        f"contrast {W.visibility:.2f}", font=dict(size=13)), height=340, **LAYOUT)
        f.update_layout(margin=dict(l=50, r=15, t=45, b=45), showlegend=False)
        f.update_xaxes(title=f"{p.split_axis} [nm]", **AX)
        f.update_yaxes(title="intensity (norm.)", range=[0, 1.05], **AX)
        c1.plotly_chart(f, **WIDE)
        if W.carpet is not None:
            zc, xc, img = W.carpet
            f = go.Figure(go.Heatmap(x=zc, y=xc * 1e6, z=np.log10(img.T / np.nanmax(img) + 1e-4), colorscale="Inferno",
                                     zmin=-4, zmax=0,
                                     colorbar=dict(title=dict(text="log₁₀ I", side="right"), thickness=12),
                                     hovertemplate="z = %{x:.4f} m<br>x = %{y:.0f} µm<extra></extra>"))
            for k_, col_, lab_ in (("slit", "#9AE6FF", "slit"), ("sample", "#FF6B9A", "sample")):
                f.add_vline(x=rw.z[k_], line=dict(color=col_, width=1.5, dash="dot"), annotation_text=lab_,
                            annotation_font_color=col_)
            f.update_layout(title=dict(text="Coherent intensity from the grating to past the sample (fringes averaged "
                                            "per pixel)", font=dict(size=13)), height=340, **LAYOUT)
            f.update_layout(margin=dict(l=55, r=10, t=45, b=45))
            f.update_layout(plot_bgcolor="#000004")
            f.update_xaxes(title="z from HM [m]", **AX)
            f.update_yaxes(title=f"{p.split_axis} [µm]", **AX)
            c2.plotly_chart(f, **WIDE)
        st.caption("Wave model: scalar, paraxial Fresnel propagation along the split axis with thin elements and an ideal KB "
                   "lens. It does not include the 40 µm focal blur of the ray model, which comes from mirror errors and "
                   "source properties that are not modelled, so the wave profiles have slightly sharper edges. Contrast = "
                   "2·|Ĩ(1/Λ)| / Ĩ(0) over the sample.")

# ─────────────────────────────────────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────────────────────────────────────
with tab_md:
    c1, c2 = st.columns([3, 2])
    c1.markdown(f"""
**Geometry.** z is measured from the horizontal KB mirror. HM at z = {p.z_hm} m with f_H = {p.f_h} m and VM at
z = {p.z_vm} m with f_V = {p.f_v} m. Each focal length is measured from its own mirror, so the foci are at
{p.z_hm + p.f_h:.3f} m (H) and {p.z_vm + p.f_v:.3f} m (V), and the CCD is at {p.z_det} m.

**Splitting.** A bar beamstop at z = {p.z_bs} m cuts the central {p.bs_width_mm} mm out of the converging beam along
**{p.split_axis}**. Half-beam A (+{p.split_axis}) and half-beam B (−{p.split_axis}) still head for the common focus.
For the zero orders to stay out of the slit, the beamstop has to be wider than the beam FWHM where it sits
(about 1.3–1.5×).

**Grating.** Thin binary grating with Fourier coefficients
E_n = (t_bar − t_gap)·a·sinc(a·n) + t_gap·δ_n0, the `EnxPhase` expression from the Talbot tutorial. The order n
picks up the angle n·λ/p. A(−1) and B(+1) bend towards the axis and cross at the sample.

**Slit.** Two blades {p.slit_upstream_mm} mm before the sample. In auto mode they open to the 1–99 % extent of the
two wanted orders.

**Self-diffraction.** The SD beams leave the sample along 2k_A − k_B and 2k_B − k_A. A beamstop
{p.dbs_after_sample_mm} mm after the sample blocks the two transmitted 1st orders, and the SD beams pass beside it
to the CCD.

**Coherence.** A and B must be mutually coherent to form a grating. The rays of A(−1) and B(+1) that meet at a
sample point left the KB a distance Δu apart. With a Gaussian-Schell coherence length ℓ = {p.coh_ratio} × rms beam
size, their coherence is exp(−Δu²/2ℓ²). A wide beamstop pushes A and B into the far tails of the beam, so Δu grows
and the TG contrast falls. Bandwidth does not wash out the fringes: Λ = (p/2)·R_sample/R_grating for every
wavelength, which the wave-optics check confirms.

**Ray model.** Paraxial Monte-Carlo. Every ray has a mirror-plane position (the beam profile), a focal-plane
position (the focal-spot blur) and a wavelength (the bandwidth). Rays are sampled mirror-symmetrically, so with
a centred beamstop A and B are exact mirror images. The model leaves out Fresnel edge diffraction from the
beamstop and slit (√(λL) ≈ 50 µm over 1 m), so keep that much margin on the clearances.
""")
    orders = np.arange(-5, 6)
    eff = sim.order_efficiencies(p, orders)
    fe = go.Figure(go.Bar(x=orders, y=eff * 100, marker_color=[C["A"] if n == -1 else C["B"] if n == 1 else C["other"]
                                                                for n in orders],
                          text=[f"{e * 100:.1f}" for e in eff], textposition="outside"))
    fe.update_layout(title="Grating order efficiency |E_n|² [%]", height=360, **LAYOUT)
    fe.update_xaxes(title="order n", dtick=1, **AX)
    fe.update_yaxes(title="%", **AX)
    c2.plotly_chart(fe, **WIDE)
    c2.code(sim.summary(res), language=None)

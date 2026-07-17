"""
Helper module for loading and applying custom electron MVA model.
Loads pre-trained XGBoost model and applies it to electron events.

Feature list is controlled by ELECTRON_FEATURES below.
Comment/uncomment features to match your training configuration.
All computations are always performed; only listed features enter the matrix.

FIXES APPLIED (relative to previous version):
  1. pratio / ntracks now use physically-motivated defaults for leptons with
     no matched jet ("lepton is its own jet" convention), instead of 0.0.
  2. Feature-matrix construction now hard-fails (RuntimeError) if any feature
     name in the saved/declared feature list is missing from the computed
     dictionary, instead of silently zero-filling that column.
  3. The exact feature order actually used for inference is logged once per
     process, so mismatches between training and inference are easy to spot.
"""

import os
import pickle
import logging
from columnflow.util import maybe_import

logger = logging.getLogger(__name__)

# Model paths (relative to this module or absolute)
_MODEL_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/../data/mva_model"
_MODEL_PATH    = os.path.join(_MODEL_DIR, "ele_xgb_clf.pkl")
_SCALER_PATH   = os.path.join(_MODEL_DIR, "ele_scaler.pkl")
_FEATURES_PATH = os.path.join(_MODEL_DIR, "ele_features.pkl")

# ──────────────────────────────────────────────────────────────────────────────
# FEATURE LIST — edit here to match your training run.
# Comment out any feature to exclude it from the input matrix.
# All values are still computed; only the uncommented ones are used.
# ──────────────────────────────────────────────────────────────────────────────
ELECTRON_FEATURES = [
        'pt', 
        'eta',
        'Irel_neutral', 
        'Irel_charged',  
        'pfRelIso03_all', 
        #'pfRelIso03_chg', 'pfRelIso04_all', 
        'btagDeepFlavB' ,
        #'jetDF' ,
        'jetNDauCharged', 'jetPtRelv2', 'pratio',
        #'jetRelIso', --> low ks than pratio
        #'ntracks',  'prel_T', 

        #'btagPNetB',
        
        'log_dxy', 
        'log_dz', 
        'sip3d',
        #'Irel_all', #--> redundant with Irel_neutral + Irel_charged
        # 'promptMVA', #--> lepton ID MVA with isolation, so redundant with Irel_neutral + Irel_charged
        # 'dxy', 'dz',
        
        

        'hoe', 
        'sieie', 
        #'nMissingHits', 
        'eInvMinusPInv', 
        'mvaNoIso',
        #'deltaEtaSC', 
        #'convVeto', #--> low Ks
        #'mvaIso', #--> highly correlated with mvaNoIso

]

# Singleton cache
_model    = None
_scaler   = None
_features = None
_logged_feature_order = False  # only log once per process


def _load_model():
    """Load and cache the trained model and scaler with robust fallbacks."""
    global _model, _scaler, _features
    joblib = maybe_import("joblib")
    if _model is None:
        model_error = None
        if os.path.exists(_MODEL_PATH):
            try:
                with open(_MODEL_PATH, "rb") as f:
                    _model = pickle.load(f, encoding="latin1")
            except Exception as e:
                model_error = e
                _model = None
        else:
            model_error = FileNotFoundError(f"Model not found at {_MODEL_PATH}")

        scaler_error = None
        if os.path.exists(_SCALER_PATH):
            try:
                _scaler = joblib.load(_SCALER_PATH)
            except Exception as e:
                scaler_error = e
                _scaler = None
        else:
            scaler_error = FileNotFoundError(f"Scaler not found at {_SCALER_PATH}")

        if os.path.exists(_FEATURES_PATH):
            try:
                with open(_FEATURES_PATH, "rb") as f:
                    _features = pickle.load(f, encoding="latin1")
            except Exception:
                _features = None
        else:
            _features = None

        if model_error is not None:
            raise RuntimeError(f"Failed to load model from {_MODEL_PATH}: {model_error}")
        if scaler_error is not None:
            raise RuntimeError(f"Failed to load scaler from {_SCALER_PATH}: {scaler_error}")

    return _model, _scaler, _features


def compute_electron_mva_score(events) -> "ak.Array":  # noqa: F821
    """
    Compute custom electron MVA scores using trained XGBoost model.

    The active feature set is controlled by ELECTRON_FEATURES at the top of this file
    (or by the feature list saved alongside the model, if present).

    Full variable catalogue (all always computed):
        pt, eta
        Irel_neutral, Irel_charged, Irel_all
        pfRelIso03_all, pfRelIso03_chg, pfRelIso04_all
        jetDF, jetNDauCharged, jetPtRelv2, pratio, prel_T, ntracks, jetRelIso
        btagPNetB, btagDeepFlavB
        log_dxy, log_dz, sip3d, dxy, dz
        hoe, sieie, deltaEtaSC, eInvMinusPInv, mvaNoIso, mvaIso
        nMissingHits, convVeto, promptMVA

    Args:
        events: NanoAOD-like awkward array with Electron and Jet collections

    Returns:
        Awkward array with electron MVA scores (per-electron, same shape as events.Electron.pt)
    """
    import numpy as np
    import awkward as ak

    global _logged_feature_order

    model, scaler, saved_features = _load_model()

    electron = events.Electron
    jet      = events.Jet

    def _flat(branch):
        return ak.to_numpy(ak.flatten(branch)).astype(np.float32)

    # ── basic kinematics ──────────────────────────────────────────────────────
    el_pt  = _flat(electron.pt)
    el_eta = _flat(electron.eta)
    el_phi = _flat(electron.phi)
    el_dxy = _flat(electron.dxy)
    el_dz  = _flat(electron.dz)
    el_sip3d = _flat(electron.sip3d)

    # ── safe branch reader ────────────────────────────────────────────────────
    def _safe(branch_fn, fallback=None):
        try:
            return _flat(branch_fn())
        except (AttributeError, ValueError):
            return np.zeros_like(el_pt) if fallback is None else np.full_like(el_pt, fallback)

    # ── mini-isolation ────────────────────────────────────────────────────────
    el_iso_all = _safe(lambda: electron.miniPFRelIso_all)
    el_iso_chg = _safe(lambda: electron.miniPFRelIso_chg)

    # ── PF-cone isolation ─────────────────────────────────────────────────────
    el_pfRelIso03_all = _safe(lambda: electron.pfRelIso03_all)
    el_pfRelIso03_chg = _safe(lambda: electron.pfRelIso03_chg)
    el_pfRelIso04_all = _safe(lambda: electron.pfRelIso04_all)

    # ── jet activity stored directly on lepton (producer pre-computes these) ──
    el_jetDF          = _safe(lambda: electron.jetDF)
    el_jetNDauCharged = _safe(lambda: electron.jetNDauCharged)
    el_jetPtRelv2     = _safe(lambda: electron.jetPtRelv2)
    el_jetRelIso      = _safe(lambda: electron.jetRelIso)
    el_promptMVA      = _safe(lambda: electron.promptMVA)

    # ── electron-ID variables ─────────────────────────────────────────────────
    el_hoe           = _safe(lambda: electron.hoe)
    el_sieie         = _safe(lambda: electron.sieie)
    el_deltaEtaSC    = _safe(lambda: electron.deltaEtaSC)
    el_eInvMinusPInv = _safe(lambda: electron.eInvMinusPInv)
    el_mvaNoIso      = _safe(lambda: electron.mvaNoIso)
    el_mvaIso        = _safe(lambda: electron.mvaIso)
    el_nMissingHits  = _safe(lambda: electron.lostHits)       # lostHits == nMissingHits
    el_convVeto      = _safe(lambda: electron.convVeto, fallback=1.0)

    # ── jet matching ──────────────────────────────────────────────────────────
    el_jetidx_ak = ak.fill_none(electron.jetIdx, -1)
    el_jetidx_ak = ak.values_astype(el_jetidx_ak, np.int32)

    n_jets_ak           = ak.num(jet)
    n_jets_per_electron = ak.values_astype(
        ak.broadcast_arrays(n_jets_ak, el_jetidx_ak)[0], np.int32)
    valid_ak            = (el_jetidx_ak >= 0) & (el_jetidx_ak < n_jets_per_electron)
    jidx_safe_ak        = ak.where(valid_ak, el_jetidx_ak, 0)
    max_jets            = int(ak.max(n_jets_ak)) + 1 if ak.max(n_jets_ak) >= 0 else 1

    def _gather_jet(branch):
        if branch is None:
            return np.zeros_like(el_pt)
        try:
            padded   = ak.pad_none(branch, max_jets, clip=True)
            gathered = padded[jidx_safe_ak]
            return _flat(ak.fill_none(gathered, 0.0))
        except (AttributeError, ValueError, TypeError):
            return np.zeros_like(el_pt)

    matched_jpt  = _gather_jet(jet.pt)
    matched_jphi = _gather_jet(jet.phi)

    try:   bpnet_branch = jet.btagPNetB
    except (AttributeError, ValueError): bpnet_branch = None
    matched_bpnet = _gather_jet(bpnet_branch)

    try:   bdeep_branch = jet.btagDeepFlavB
    except (AttributeError, ValueError): bdeep_branch = None
    matched_bdeep = _gather_jet(bdeep_branch)

    try:   ncon_branch = jet.nConstituents
    except (AttributeError, ValueError): ncon_branch = None
    matched_ncon = _gather_jet(ncon_branch)

    valid_flat = ak.to_numpy(ak.flatten(valid_ak)).astype(bool)

    # ── derived jet variables ─────────────────────────────────────────────────
    # NOTE: defaults for jet-less leptons (pratio, ntracks -> 0.0) are chosen to
    # EXACTLY MATCH the training-data producer (lepton_producer_v2.py::_compute_jet_vars),
    # which uses `z = np.float32(0.0)` as the fallback for every jet-derived
    # quantity when `valid` (i.e. a matched jetIdx) is False. Do NOT change
    # these defaults without also changing (and retraining against) the
    # producer -- any mismatch here silently shifts the score distribution
    # for every jet-less lepton relative to what the model was trained on.
    with np.errstate(divide="ignore", invalid="ignore"):
        pratio = np.where(
            valid_flat & (matched_jpt > 0), el_pt / matched_jpt, 0.0,
        ).astype(np.float32)

    def _delta_phi(phi1, phi2):
        dphi = np.abs(phi1 - phi2)
        return np.where(dphi > np.pi, 2.0 * np.pi - dphi, dphi)

    dphi   = _delta_phi(el_phi, matched_jphi)
    prel_T = np.where(valid_flat, np.abs(el_pt * np.sin(dphi)), 0.0).astype(np.float32)

    btagPNetB     = np.where(valid_flat, matched_bpnet, 0.0).astype(np.float32)
    btagDeepFlavB = np.where(valid_flat, matched_bdeep, 0.0).astype(np.float32)
    ntracks       = np.where(valid_flat, matched_ncon, 0.0).astype(np.float32)

    # ── isolation decomposition ───────────────────────────────────────────────
    Irel_charged = el_iso_chg
    Irel_neutral = (el_iso_all - el_iso_chg).astype(np.float32)
    Irel_all     = el_iso_all

    # ── log IP ────────────────────────────────────────────────────────────────
    log_dxy = np.log(np.abs(el_dxy) + 1e-10).astype(np.float32)
    log_dz  = np.log(np.abs(el_dz)  + 1e-10).astype(np.float32)

    # ── full computed dictionary (every variable, always) ────────────────────
    computed = {
        "pt":              el_pt,
        "eta":             el_eta,
        "Irel_neutral":    Irel_neutral,
        "Irel_charged":    Irel_charged,
        "Irel_all":        Irel_all,
        "pfRelIso03_all":  el_pfRelIso03_all,
        "pfRelIso03_chg":  el_pfRelIso03_chg,
        "pfRelIso04_all":  el_pfRelIso04_all,
        "jetDF":           el_jetDF,
        "jetNDauCharged":  el_jetNDauCharged,
        "jetPtRelv2":      el_jetPtRelv2,
        "jetRelIso":       el_jetRelIso,
        "pratio":          pratio,
        "prel_T":          prel_T,
        "ntracks":         ntracks,
        "btagPNetB":       btagPNetB,
        "btagDeepFlavB":   btagDeepFlavB,
        "log_dxy":         log_dxy,
        "log_dz":          log_dz,
        "dxy":             el_dxy,
        "dz":              el_dz,
        "sip3d":           el_sip3d,
        "hoe":             el_hoe,
        "sieie":           el_sieie,
        "deltaEtaSC":      el_deltaEtaSC,
        "eInvMinusPInv":   el_eInvMinusPInv,
        "mvaNoIso":        el_mvaNoIso,
        "mvaIso":          el_mvaIso,
        "nMissingHits":    el_nMissingHits,
        "convVeto":        el_convVeto,
        "promptMVA":       el_promptMVA,
    }

    # ── build feature matrix from ELECTRON_FEATURES list ─────────────────────
    # If model was saved with a feature list, use that; otherwise fall back to
    # the ELECTRON_FEATURES constant defined at the top of this file.
    feat_order = saved_features if saved_features is not None else ELECTRON_FEATURES

    # FIX #3: hard-fail instead of silently zero-filling missing features.
    # A zero-filled column fed through a StandardScaler fit on real data
    # shifts every event's score in a correlated, hard-to-detect way.
    missing = [feat for feat in feat_order if feat not in computed]
    if missing:
        raise RuntimeError(
            f"Electron MVA: feature(s) {missing} required by the active feature "
            f"list are not present in the computed variable dictionary. "
            f"Available computed variables: {sorted(computed.keys())}. "
            f"This almost always means a naming mismatch between ele_features.pkl "
            f"(or ELECTRON_FEATURES) and this producer -- fix the name before proceeding, "
            f"do not let this fall back to a zero-filled column."
        )

    # FIX #4: log (once) the exact feature order used for inference, so any
    # mismatch vs. the training-time feature order is easy to spot from logs.
    if not _logged_feature_order:
        source = "saved_features (ele_features.pkl)" if saved_features is not None else "ELECTRON_FEATURES (in-file)"
        logger.info(f"Electron MVA: using {len(feat_order)} features from {source}: {feat_order}")
        _logged_feature_order = True

    X_list = [computed[feat] for feat in feat_order]

    X        = np.column_stack(X_list).astype(np.float32)
    X_scaled = scaler.transform(X)

    try:
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_scaled)[:, 1]
        else:
            import xgboost as xgb
            scores = model.predict(xgb.DMatrix(X_scaled))
    except Exception as e:
        raise RuntimeError(f"Failed to generate predictions from model: {e}")
    logger.info(f"Electron MVA raw score stats: min={scores.min():.4f} max={scores.max():.4f} mean={scores.mean():.4f}")
    # ── diagnostic logging (helps debug sudden event-yield drops) ────────────
    n_total = len(el_pt)
    if n_total > 0:
        n_no_jet = int(np.sum(~valid_flat))
        pct_no_jet = 100.0 * n_no_jet / n_total

        pt_lo_mask = el_pt < 20.0
        n_pt_lo = int(np.sum(pt_lo_mask))

        s_all = scores
        s_lo  = scores[pt_lo_mask] if n_pt_lo > 0 else np.array([])

        def _pass_frac(arr, cut):
            return 100.0 * np.mean(arr > cut) if len(arr) > 0 else float("nan")

        logger.info(
            f"Electron MVA batch: n={n_total}  "
            f"no_matched_jet={n_no_jet} ({pct_no_jet:.1f}%)  "
            f"pT<20={n_pt_lo} ({100.0 * n_pt_lo / n_total:.1f}%)"
        )
        logger.info(
            f"Electron MVA score (all): "
            f"min={s_all.min():.3f} p10={np.percentile(s_all, 10):.3f} "
            f"median={np.median(s_all):.3f} p90={np.percentile(s_all, 90):.3f} "
            f"max={s_all.max():.3f} mean={s_all.mean():.3f}"
        )
        if n_pt_lo > 0:
            logger.info(
                f"Electron MVA score (pT<20 only, n={n_pt_lo}): "
                f"min={s_lo.min():.3f} median={np.median(s_lo):.3f} "
                f"max={s_lo.max():.3f} mean={s_lo.mean():.3f}"
            )
        logger.info(
            f"Electron MVA pass-fraction @ cuts -- "
            f"0.3: all={_pass_frac(s_all, 0.3):.1f}% pT<20={_pass_frac(s_lo, 0.3):.1f}% | "
            f"0.5: all={_pass_frac(s_all, 0.5):.1f}% pT<20={_pass_frac(s_lo, 0.5):.1f}% | "
            f"0.6: all={_pass_frac(s_all, 0.6):.1f}% pT<20={_pass_frac(s_lo, 0.6):.1f}%"
        )
    else:
        logger.info("Electron MVA batch: no electrons in this chunk")

    scores = ak.unflatten(scores, ak.num(electron.pt))
    return scores
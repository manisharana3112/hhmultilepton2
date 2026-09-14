"""
Helper module for loading and applying custom muon MVA model.
Loads pre-trained XGBoost model and applies it to muon events.
"""

import os
import sys
import pickle
from columnflow.util import maybe_import


# Model paths (relative to this module or absolute)
_MODEL_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/../data/mva_model/v1"
_MODEL_PATH = os.path.join(_MODEL_DIR, "mu_xgb_clf.pkl")
_SCALER_PATH = os.path.join(_MODEL_DIR, "mu_scaler.pkl")
_FEATURES_PATH = os.path.join(_MODEL_DIR, "mu_features.pkl")

# Default feature list (fallback if loading fails). Order MUST match mu_features.pkl / the
# trained model's booster feature order.
_DEFAULT_MUON_FEATURES = [
    "pt", "eta",
    "Irel_neutral", "Irel_charged",
    "pfRelIso03_all", "btagDeepFlavB", "jetNDauCharged", "jetPtRelv2",
    "pratio", "log_dxy", "log_dz", "sip3d",
    "segmentComp", "isTracker", "isGlobal", "nStations",
]

# Singleton cache for model and scaler (loaded once)
_model = None
_scaler = None
_features = None

# names of NanoAOD branches we've already warned about being missing (warn once each)
_warned_missing_branches = set()

# guard so the MVA_FEATURE_DEBUG parity table prints only once per process
_parity_printed = False


def _load_model():
    """Load and cache the trained model and scaler with robust fallbacks."""
    global _model, _scaler, _features
    joblib = maybe_import("joblib")  # Lazy import for joblib
    if _model is None:
        # Try to load model
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

        # Try to load scaler (saved via joblib.dump in training)
        scaler_error = None
        if os.path.exists(_SCALER_PATH):
            try:
                _scaler = joblib.load(_SCALER_PATH)
            except Exception as e:
                scaler_error = e
                _scaler = None
        else:
            scaler_error = FileNotFoundError(f"Scaler not found at {_SCALER_PATH}")

        # Try to load features
        if os.path.exists(_FEATURES_PATH):
            try:
                with open(_FEATURES_PATH, "rb") as f:
                    _features = pickle.load(f, encoding="latin1")
            except Exception:
                _features = None
        else:
            _features = None

        # If model or scaler failed, raise an exception
        # (features failing is non-critical, we have defaults)
        if model_error is not None:
            raise RuntimeError(f"Failed to load model from {_MODEL_PATH}: {model_error}")
        if scaler_error is not None:
            raise RuntimeError(f"Failed to load scaler from {_SCALER_PATH}: {scaler_error}")

    return _model, _scaler, _features


def compute_muon_mva_score(events) -> "ak.Array":  # noqa: F821
    """
    Compute custom muon MVA scores using trained XGBoost model.

    Expected muon features (12 total):
    'pt', 'eta', 'Irel_neutral', 'Irel_charged',
    'pratio', 'ntracks', 'btagPNetB',
    'log_dxy', 'log_dz', 'sip3d',
    'segmentComp', 'nTrackerLayers'

    Features computed from NanoAOD following same recipe as training in Lepton-MVA-Run3/src/lepton_producer.py

    Args:
        events: NanoAOD-like awkward array with Muon collection

    Returns:
        Awkward array with muon MVA scores (per-muon, same structure as events.Muon.pt)
    """
    # Lazy imports - only load when function is called
    import numpy as np
    import awkward as ak

    model, scaler, features = _load_model()

    muon = events.Muon
    jet = events.Jet

    # Flatten muons first to avoid shape mismatch issues
    def _flat(branch):
        return ak.to_numpy(ak.flatten(branch)).astype(np.float32)

    # Extract basic muon properties (flatten to 1D)
    mu_pt = _flat(muon.pt)
    mu_eta = _flat(muon.eta)
    mu_dxy = _flat(muon.dxy)
    mu_dz = _flat(muon.dz)
    mu_sip3d = _flat(muon.sip3d)

    # Optional branches: read coll.<field> or, if the field is absent (most commonly because it
    # was not declared in the selector's `uses`, so columnflow never loaded it), WARN LOUDLY and
    # fall back to zeros. A silent zero-fill here previously masked missing inputs and silently
    # degraded the score, so a missing branch must never be quiet again.
    def _opt_flat(coll, field):
        try:
            return _flat(getattr(coll, field))
        except (AttributeError, ValueError, KeyError):
            if field not in _warned_missing_branches:
                _warned_missing_branches.add(field)
                import sys
                print(
                    f"[compute_muon_mva_score] WARNING: branch '{field}' not available; "
                    f"feature filled with ZEROS (degrades the MVA). "
                    f"Declare it in the selector's `uses`.",
                    file=sys.stderr,
                )
            return np.zeros_like(mu_pt)

    mu_iso_all = _opt_flat(muon, "miniPFRelIso_all")
    mu_iso_chg = _opt_flat(muon, "miniPFRelIso_chg")
    mu_seg = _opt_flat(muon, "segmentComp")
    mu_nlayers = _opt_flat(muon, "nTrackerLayers")
    # muon-quality features the model was trained on (bool branches flatten to 0.0/1.0)
    mu_is_tracker = _opt_flat(muon, "isTracker")
    mu_n_stations = _opt_flat(muon, "nStations")
    mu_is_global = _opt_flat(muon, "isGlobal")
    # v2 model inputs (direct per-muon NanoAOD branches)
    mu_pfreliso03 = _opt_flat(muon, "pfRelIso03_all")
    mu_jetndau = _opt_flat(muon, "jetNDauCharged")
    mu_jetptrelv2 = _opt_flat(muon, "jetPtRelv2")
    # DIAGNOSTIC (test A): the v2 scaler expects jetPtRelv2 ~ 0 (train std 0.017); feeding the
    # real ~GeV branch pushes it ~350 sigma out of distribution and collapses the model. Set
    # MVA_ZERO_JETPTRELV2=1 to feed 0 (== training mean, in-distribution) and see if AUC recovers.
    if os.environ.get("MVA_ZERO_JETPTRELV2"):
        mu_jetptrelv2 = np.zeros_like(mu_pt)

    # -------------------------------------------------------------------------
    # Jet matching: fill None in jetIdx BEFORE any boolean operations.
    # muon.jetIdx comes as an option-type (?int32) in awkward when events have
    # no matched jet, causing bitwise_and to fail on None-typed arrays.
    # Filling None -> -1 converts it to a plain integer array first.
    # -------------------------------------------------------------------------
    mu_jetidx_ak = ak.fill_none(muon.jetIdx, -1)

    # Cast to int32 explicitly to guarantee a plain (non-option) integer type
    mu_jetidx_ak = ak.values_astype(mu_jetidx_ak, np.int32)

    n_jets_ak = ak.num(jet)  # Per-event number of jets

    # Broadcast n_jets to match muon structure (per-muon) and cast to int32
    n_jets_per_muon = ak.values_astype(
        ak.broadcast_arrays(n_jets_ak, mu_jetidx_ak)[0],
        np.int32,
    )

    # Check validity: jetIdx >= 0 and jetIdx < num_jets (both per-muon)
    # Both sides are now plain int32 arrays — bitwise_and is safe
    valid_ak = (mu_jetidx_ak >= 0) & (mu_jetidx_ak < n_jets_per_muon)
    jidx_safe_ak = ak.where(valid_ak, mu_jetidx_ak, 0)  # 0 as safe fallback

    # Pad jets to avoid index out of bounds
    max_jets = int(ak.max(n_jets_ak)) + 1 if ak.max(n_jets_ak) >= 0 else 1

    def _gather_jet(branch):
        """Safely gather jet properties matched to muons."""
        if branch is None:
            return np.zeros_like(mu_pt)
        try:
            padded = ak.pad_none(branch, max_jets, clip=True)
            gathered = padded[jidx_safe_ak]
            filled = ak.fill_none(gathered, 0.0)
            return _flat(filled)
        except (AttributeError, ValueError, TypeError):
            return np.zeros_like(mu_pt)

    # Get matched jet properties
    matched_jpt = _gather_jet(jet.pt)

    def _opt_jet_branch(field):
        try:
            return getattr(jet, field)
        except (AttributeError, ValueError, KeyError):
            if field not in _warned_missing_branches:
                _warned_missing_branches.add(field)
                import sys
                print(
                    f"[compute_muon_mva_score] WARNING: jet branch '{field}' not available; "
                    f"matched-jet feature filled with ZEROS (degrades the MVA). "
                    f"Declare 'Jet.{field}' in the selector's `uses`.",
                    file=sys.stderr,
                )
            return None

    matched_bpnet = _gather_jet(_opt_jet_branch("btagPNetB"))
    matched_ncon = _gather_jet(_opt_jet_branch("nConstituents"))
    matched_btagdeepflavb = _gather_jet(_opt_jet_branch("btagDeepFlavB"))

    # Flatten valid mask to 1D numpy bool
    valid_flat = ak.to_numpy(ak.flatten(valid_ak)).astype(bool)

    # pratio = muon_pt / jet_pt (0 if no matched jet)
    with np.errstate(divide="ignore", invalid="ignore"):
        pratio = np.where(
            valid_flat & (matched_jpt > 0),
            mu_pt / matched_jpt,
            0.0,
        ).astype(np.float32)

    # Isolation features
    Irel_charged = mu_iso_chg
    Irel_neutral = (mu_iso_all - mu_iso_chg).astype(np.float32)

    # log-transformed IP variables
    log_dxy = np.log(np.abs(mu_dxy) + 1e-10).astype(np.float32)
    log_dz = np.log(np.abs(mu_dz) + 1e-10).astype(np.float32)

    # B-tagging and nTracks (0 if no matched jet)
    btagPNetB = np.where(valid_flat, matched_bpnet, 0.0).astype(np.float32)
    ntracks = np.where(valid_flat, matched_ncon, 0.0).astype(np.float32)
    btagDeepFlavB = np.where(valid_flat, matched_btagdeepflavb, 0.0).astype(np.float32)

    # Build feature dictionary with all computed features. feat_order (from the loaded
    # *_features.pkl) selects which of these the current model actually consumes, so leaving
    # extra (e.g. v1-only) keys here is harmless.
    computed = {
        "pt": mu_pt,
        "eta": mu_eta,
        "Irel_neutral": Irel_neutral,
        "Irel_charged": Irel_charged,
        "pratio": pratio,
        "ntracks": ntracks,
        "btagPNetB": btagPNetB,
        "log_dxy": log_dxy,
        "log_dz": log_dz,
        "sip3d": mu_sip3d,
        "segmentComp": mu_seg,
        "nTrackerLayers": mu_nlayers,
        "isTracker": mu_is_tracker,
        "nStations": mu_n_stations,
        "isGlobal": mu_is_global,
        # v2 inputs
        "pfRelIso03_all": mu_pfreliso03,
        "btagDeepFlavB": btagDeepFlavB,  # nearest-jet btagDeepFlavB (via jetIdx matching)
        "jetNDauCharged": mu_jetndau,
        "jetPtRelv2": mu_jetptrelv2,
    }

    # Build feature matrix in correct order
    # Use saved features if available, otherwise use defaults
    feat_order = _features if _features is not None else _DEFAULT_MUON_FEATURES

    X_list = []
    for feat in feat_order:
        if feat in computed:
            X_list.append(computed[feat])
        else:
            # Missing feature - fill with zeros
            X_list.append(np.zeros_like(mu_pt))

    X = np.column_stack(X_list).astype(np.float32)

    # ------------------------------------------------------------------
    # Optional feature-parity diagnostic (see compute_electron_mva_score for details).
    # Compares the in-situ feature distribution against the training distribution stored in
    # the scaler (StandardScaler.mean_/scale_). Enable with:  MVA_FEATURE_DEBUG=1
    # ------------------------------------------------------------------
    global _parity_printed
    if os.environ.get("MVA_FEATURE_DEBUG") and not _parity_printed:
        _parity_printed = True
        tr_mean = getattr(scaler, "mean_", None)
        tr_std = getattr(scaler, "scale_", None)
        print(f"[MVA feature parity] muon (n={X.shape[0]}, features_loaded={_features is not None})",
              file=sys.stderr)
        print("  %-16s %12s %12s | %12s %12s | %7s"
              % ("feature", "insitu_mean", "insitu_std", "train_mean", "train_std", "pull"),
              file=sys.stderr)
        for i, feat in enumerate(feat_order):
            im, isd = float(np.mean(X[:, i])), float(np.std(X[:, i]))
            tm = float(tr_mean[i]) if tr_mean is not None else float("nan")
            ts = float(tr_std[i]) if tr_std is not None else float("nan")
            pull = (im - tm) / ts if ts else float("nan")
            flag = "  <== OFF" if (ts and abs(pull) > 0.5) else ""
            print("  %-16s %12.5f %12.5f | %12.5f %12.5f | %7.2f%s"
                  % (feat, im, isd, tm, ts, pull, flag), file=sys.stderr)

    # Apply scaler (trained on same features)
    X_scaled = scaler.transform(X)

    # Get predictions (probabilities for positive class = prompt muon)
    try:
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_scaled)[:, 1]
        else:
            import xgboost as xgb
            dmatrix = xgb.DMatrix(X_scaled)
            scores = model.predict(dmatrix)
    except Exception as e:
        raise RuntimeError(f"Failed to generate predictions from model: {e}")

    # Reshape back to awkward structure (per-muon)
    num_muons = ak.num(muon.pt)
    scores = ak.unflatten(scores, num_muons)

    return scores

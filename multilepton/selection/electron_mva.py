"""
Helper module for loading and applying custom electron MVA model.
Loads pre-trained XGBoost model and applies it to electron events.
"""

import os
import sys
import pickle
from columnflow.util import maybe_import


# Model paths (relative to this module or absolute)
_MODEL_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/../data/mva_model/v1"
_MODEL_PATH = os.path.join(_MODEL_DIR, "ele_xgb_clf.pkl")
_SCALER_PATH = os.path.join(_MODEL_DIR, "ele_scaler.pkl")
_FEATURES_PATH = os.path.join(_MODEL_DIR, "ele_features.pkl")

# Default feature list (fallback if loading fails). Order MUST match ele_features.pkl / the
# trained model's booster feature order. This is the v2 model's feature set.
_DEFAULT_ELECTRON_FEATURES = [
    "pt", "eta",
    "Irel_neutral", "Irel_charged",
    "pfRelIso03_all", "btagDeepFlavB", "jetNDauCharged", "jetPtRelv2",
    "pratio", "log_dxy", "log_dz", "sip3d",
    "hoe", "sieie", "eInvMinusPInv", "mvaNoIso",
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


def compute_electron_mva_score(events) -> "ak.Array":  # noqa: F821
    """
    Compute custom electron MVA scores using trained XGBoost model.

    Expected electron features (16 total):
    'pt', 'eta', 'Irel_neutral', 'Irel_charged',
    'pratio', 'prel_T', 'ntracks', 'btagPNetB',
    'log_dxy', 'log_dz', 'sip3d',
    'hoe', 'sieie', 'deltaEtaSC', 'eInvMinusPInv', 'mvaNoIso'

    Features computed from NanoAOD following same recipe as training in Lepton-MVA-Run3/src/lepton_producer.py

    Args:
        events: NanoAOD-like awkward array with Electron collection

    Returns:
        Awkward array with electron MVA scores (per-electron, same structure as events.Electron.pt)
    """
    # Lazy imports - only load when function is called
    import numpy as np
    import awkward as ak

    model, scaler, features = _load_model()

    electron = events.Electron
    jet = events.Jet

    # Flatten electrons first to avoid shape mismatch issues
    def _flat(branch):
        return ak.to_numpy(ak.flatten(branch)).astype(np.float32)

    # Extract basic electron properties (flatten to 1D)
    el_pt = _flat(electron.pt)
    el_eta = _flat(electron.eta)
    el_phi = _flat(electron.phi)
    el_dxy = _flat(electron.dxy)
    el_dz = _flat(electron.dz)
    el_sip3d = _flat(electron.sip3d)

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
                    f"[compute_electron_mva_score] WARNING: branch '{field}' not available; "
                    f"feature filled with ZEROS (degrades the MVA). "
                    f"Declare it in the selector's `uses`.",
                    file=sys.stderr,
                )
            return np.zeros_like(el_pt)

    el_iso_all = _opt_flat(electron, "miniPFRelIso_all")
    el_iso_chg = _opt_flat(electron, "miniPFRelIso_chg")
    el_hoe = _opt_flat(electron, "hoe")
    el_sieie = _opt_flat(electron, "sieie")
    el_deltaEtaSC = _opt_flat(electron, "deltaEtaSC")
    el_eInvMinusPInv = _opt_flat(electron, "eInvMinusPInv")
    el_mvaNoIso = _opt_flat(electron, "mvaNoIso")
    # v2 model inputs (direct per-electron NanoAOD branches)
    el_pfreliso03 = _opt_flat(electron, "pfRelIso03_all")
    el_jetndau = _opt_flat(electron, "jetNDauCharged")
    el_jetptrelv2 = _opt_flat(electron, "jetPtRelv2")
    # DIAGNOSTIC (test A): the v2 scaler expects jetPtRelv2 ~ 0 (train std 0.017); feeding the
    # real ~GeV branch pushes it ~350 sigma out of distribution and collapses the model. Set
    # MVA_ZERO_JETPTRELV2=1 to feed 0 (== training mean, in-distribution) and see if AUC recovers.
    if os.environ.get("MVA_ZERO_JETPTRELV2"):
        el_jetptrelv2 = np.zeros_like(el_pt)

    # -------------------------------------------------------------------------
    # Jet matching: fill None in jetIdx BEFORE any boolean operations.
    # electron.jetIdx comes as an option-type (?int32) in awkward when events have
    # no matched jet, causing bitwise_and to fail on None-typed arrays.
    # Filling None -> -1 converts it to a plain integer array first.
    # -------------------------------------------------------------------------
    el_jetidx_ak = ak.fill_none(electron.jetIdx, -1)

    # Cast to int32 explicitly to guarantee a plain (non-option) integer type
    el_jetidx_ak = ak.values_astype(el_jetidx_ak, np.int32)

    n_jets_ak = ak.num(jet)  # Per-event number of jets

    # Broadcast n_jets to match electron structure (per-electron) and cast to int32
    n_jets_per_electron = ak.values_astype(
        ak.broadcast_arrays(n_jets_ak, el_jetidx_ak)[0],
        np.int32,
    )

    # Check validity: jetIdx >= 0 and jetIdx < num_jets (both per-electron)
    # Both sides are now plain int32 arrays — bitwise_and is safe
    valid_ak = (el_jetidx_ak >= 0) & (el_jetidx_ak < n_jets_per_electron)
    jidx_safe_ak = ak.where(valid_ak, el_jetidx_ak, 0)  # 0 as safe fallback

    # Pad jets to avoid index out of bounds
    max_jets = int(ak.max(n_jets_ak)) + 1 if ak.max(n_jets_ak) >= 0 else 1

    def _gather_jet(branch):
        """Safely gather jet properties matched to electrons."""
        if branch is None:
            return np.zeros_like(el_pt)
        try:
            padded = ak.pad_none(branch, max_jets, clip=True)
            gathered = padded[jidx_safe_ak]
            filled = ak.fill_none(gathered, 0.0)
            return _flat(filled)
        except (AttributeError, ValueError, TypeError):
            return np.zeros_like(el_pt)

    # Get matched jet properties
    matched_jpt = _gather_jet(jet.pt)
    matched_jphi = _gather_jet(jet.phi)

    def _opt_jet_branch(field):
        try:
            return getattr(jet, field)
        except (AttributeError, ValueError, KeyError):
            if field not in _warned_missing_branches:
                _warned_missing_branches.add(field)
                import sys
                print(
                    f"[compute_electron_mva_score] WARNING: jet branch '{field}' not available; "
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

    # Compute features
    def _delta_phi(phi1, phi2):
        """Compute delta-phi in [-pi, pi]."""
        dphi = np.abs(phi1 - phi2)
        return np.where(dphi > np.pi, 2.0 * np.pi - dphi, dphi)

    # pratio = electron_pt / jet_pt (0 if no matched jet)
    with np.errstate(divide="ignore", invalid="ignore"):
        pratio = np.where(
            valid_flat & (matched_jpt > 0),
            el_pt / matched_jpt,
            0.0,
        ).astype(np.float32)

    # prel_T = abs(electron_pt * sin(delta_phi))
    dphi = _delta_phi(el_phi, matched_jphi)
    prel_T = np.where(
        valid_flat,
        np.abs(el_pt * np.sin(dphi)),
        0.0,
    ).astype(np.float32)

    # Isolation features
    Irel_charged = el_iso_chg
    Irel_neutral = (el_iso_all - el_iso_chg).astype(np.float32)

    # log-transformed IP variables
    log_dxy = np.log(np.abs(el_dxy) + 1e-10).astype(np.float32)
    log_dz = np.log(np.abs(el_dz) + 1e-10).astype(np.float32)

    # B-tagging and nTracks (0 if no matched jet)
    btagPNetB = np.where(valid_flat, matched_bpnet, 0.0).astype(np.float32)
    ntracks = np.where(valid_flat, matched_ncon, 0.0).astype(np.float32)
    btagDeepFlavB = np.where(valid_flat, matched_btagdeepflavb, 0.0).astype(np.float32)

    # Build feature dictionary with all computed features. feat_order (from the loaded
    # *_features.pkl) selects which of these the current model actually consumes, so leaving
    # extra (e.g. v1-only) keys here is harmless.
    computed = {
        "pt": el_pt,
        "eta": el_eta,
        "Irel_neutral": Irel_neutral,
        "Irel_charged": Irel_charged,
        "pratio": pratio,
        "prel_T": prel_T,
        "ntracks": ntracks,
        "btagPNetB": btagPNetB,
        "log_dxy": log_dxy,
        "log_dz": log_dz,
        "sip3d": el_sip3d,
        "hoe": el_hoe,
        "sieie": el_sieie,
        "deltaEtaSC": el_deltaEtaSC,
        "eInvMinusPInv": el_eInvMinusPInv,
        "mvaNoIso": el_mvaNoIso,
        # v2 inputs
        "pfRelIso03_all": el_pfreliso03,
        "btagDeepFlavB": btagDeepFlavB,  # nearest-jet btagDeepFlavB (via jetIdx matching)
        "jetNDauCharged": el_jetndau,
        "jetPtRelv2": el_jetptrelv2,
    }

    # Build feature matrix in correct order
    # Use saved features if available, otherwise use defaults
    feat_order = _features if _features is not None else _DEFAULT_ELECTRON_FEATURES

    X_list = []
    for feat in feat_order:
        if feat in computed:
            X_list.append(computed[feat])
        else:
            # Missing feature - fill with zeros
            X_list.append(np.zeros_like(el_pt))

    X = np.column_stack(X_list).astype(np.float32)

    # ------------------------------------------------------------------
    # Optional feature-parity diagnostic.
    # A validated model that scores well offline but poorly in-situ almost always
    # means the inference features differ from the training features. The scaler
    # (StandardScaler) stores the per-feature training mean_/scale_, so we can compare
    # the in-situ feature distribution against training directly. Any feature whose
    # in-situ mean is far from the training mean (large |pull|) — or a feature that is
    # silently all-zero because its NanoAOD branch was missing — is the culprit.
    # Enable with:  MVA_FEATURE_DEBUG=1
    # ------------------------------------------------------------------
    global _parity_printed
    if os.environ.get("MVA_FEATURE_DEBUG") and not _parity_printed:
        _parity_printed = True
        tr_mean = getattr(scaler, "mean_", None)
        tr_std = getattr(scaler, "scale_", None)
        print(f"[MVA feature parity] electron (n={X.shape[0]}, features_loaded={_features is not None})",
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

    # Get predictions (probabilities for positive class = prompt electron)
    try:
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_scaled)[:, 1]
        else:
            import xgboost as xgb
            dmatrix = xgb.DMatrix(X_scaled)
            scores = model.predict(dmatrix)
    except Exception as e:
        raise RuntimeError(f"Failed to generate predictions from model: {e}")

    # Reshape back to awkward structure (per-electron)
    num_electrons = ak.num(electron.pt)
    scores = ak.unflatten(scores, num_electrons)

    return scores

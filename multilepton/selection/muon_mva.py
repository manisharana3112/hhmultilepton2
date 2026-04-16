"""
Helper module for loading and applying custom muon MVA model.
Loads pre-trained XGBoost model and applies it to muon events.
"""

import os
import pickle
import numpy as np
import awkward as ak


# Model paths (relative to this module or absolute)
_MODEL_DIR = os.path.expandvars("${PWD}/../../../Lepton-MVA-Run3/models")
_MODEL_PATH = os.path.join(_MODEL_DIR, "mu_xgb_clf.pkl")
_SCALER_PATH = os.path.join(_MODEL_DIR, "mu_scaler.pkl")
_FEATURES_PATH = os.path.join(_MODEL_DIR, "mu_features.pkl")

# Singleton cache for model and scaler (loaded once)
_model = None
_scaler = None
_features = None


def _load_model():
    """Load and cache the trained model and scaler."""
    global _model, _scaler, _features
    
    if _model is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {_MODEL_PATH}")
        if not os.path.exists(_SCALER_PATH):
            raise FileNotFoundError(f"Scaler not found at {_SCALER_PATH}")
        if not os.path.exists(_FEATURES_PATH):
            raise FileNotFoundError(f"Features not found at {_FEATURES_PATH}")
        
        with open(_MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        
        with open(_SCALER_PATH, "rb") as f:
            _scaler = pickle.load(f)
        
        with open(_FEATURES_PATH, "rb") as f:
            _features = pickle.load(f)
    
    return _model, _scaler, _features


def compute_muon_mva_score(events: ak.Array) -> ak.Array:
    """
    Compute custom muon MVA scores using trained XGBoost model.
    
    Expected muon features (13 total):
    'pdgId', 'pt', 'eta', 'pratio', 'prel_T', 'Irel_charged', 'Irel_neutral',
    'dxy', 'dz', 'sip3d', 'segmentComp', 'btagDeepFlavB', 'ntracks'
    
    Features computed from NanoAOD following same recipe as training in Lepton-MVA-Run3/src/lepton_producer.py
    
    Args:
        events: NanoAOD-like awkward array with Muon collection
        
    Returns:
        Awkward array with muon MVA scores (per-muon, same structure as events.Muon.pt)
    """
    model, scaler, features = _load_model()
    
    muon = events.Muon
    jet = events.Jet
    
    # Compute features following the same calculation as in lepton_producer.py training
    feature_dict = {}
    
    for feat in features:
        if feat == 'pdgId':
            feature_dict[feat] = muon.pdgId
        
        elif feat == 'pt':
            feature_dict[feat] = muon.pt
        
        elif feat == 'eta':
            feature_dict[feat] = muon.eta
        
        elif feat == 'pratio':
            # pratio = lepton_pt / jet_pt (from associated jet)
            # Computed as: lep_pt / jets_pt[jidx]
            jet_indices = muon.jetIdx
            jet_pts = ak.fill_none(ak.where(jet_indices >= 0, jet[jet_indices].pt, None), 0.0)
            pratio = ak.where(jet_pts > 0, muon.pt / jet_pts, 0.0)
            feature_dict[feat] = pratio
        
        elif feat == 'prel_T':
            # prel_T = abs(lep_pt * sin(delta_phi)) where delta_phi = angle between lepton and jet
            # Computed as: delta_phi = (lep_phi - jets_phi + pi) % (2*pi) - pi
            #              prelT = lep_pt * sin(delta_phi)
            jet_indices = muon.jetIdx
            
            # Get jet phi values (0.0 if no associated jet)
            jet_phi = ak.fill_none(ak.where(jet_indices >= 0, jet[jet_indices].phi, None), 0.0)
            
            # Compute delta phi
            delta_phi = (muon.phi - jet_phi + np.pi) % (2 * np.pi) - np.pi
            
            # Compute prel_T
            prel_T = ak.abs(muon.pt * np.sin(delta_phi))
            feature_dict[feat] = prel_T
        
        elif feat == 'Irel_charged':
            # Charged particle isolation: miniPFRelIso_chg
            if hasattr(muon, 'miniPFRelIso_chg'):
                feature_dict[feat] = muon.miniPFRelIso_chg
            else:
                feature_dict[feat] = ak.zeros_like(muon.pt)
        
        elif feat == 'Irel_neutral':
            # Neutral particle isolation: miniPFRelIso_all - miniPFRelIso_chg
            if hasattr(muon, 'miniPFRelIso_all') and hasattr(muon, 'miniPFRelIso_chg'):
                feature_dict[feat] = muon.miniPFRelIso_all - muon.miniPFRelIso_chg
            else:
                feature_dict[feat] = ak.zeros_like(muon.pt)
        
        elif feat == 'dxy':
            feature_dict[feat] = muon.dxy
        
        elif feat == 'dz':
            feature_dict[feat] = muon.dz
        
        elif feat == 'sip3d':
            feature_dict[feat] = muon.sip3d
        
        elif feat == 'segmentComp':
            # Segment compatibility (CMS variable, muon-specific)
            if hasattr(muon, 'segmentComp'):
                feature_dict[feat] = muon.segmentComp
            else:
                feature_dict[feat] = ak.zeros_like(muon.pt)
        
        elif feat == 'btagDeepFlavB':
            # B-tagging score from associated jet
            jet_indices = muon.jetIdx
            btag = ak.fill_none(ak.where(jet_indices >= 0, jet[jet_indices].btagDeepFlavB, None), 0.0)
            feature_dict[feat] = btag
        
        elif feat == 'ntracks':
            # Number of tracks in associated jet: jet.nConstituents
            jet_indices = muon.jetIdx
            ntracks = ak.fill_none(ak.where(jet_indices >= 0, jet[jet_indices].nConstituents, None), 0.0)
            feature_dict[feat] = ntracks
    
    # Flatten awkward arrays to 1D numpy (muon by muon)
    num_muons = ak.num(muon.pt)
    X_list = []
    
    for feat in features:
        arr = feature_dict[feat]
        # Flatten to 1D
        flat = ak.to_numpy(ak.flatten(arr))
        X_list.append(flat)
    
    X = np.column_stack(X_list)
    
    # Apply scaler (trained on same features)
    X_scaled = scaler.transform(X)
    
    # Get predictions (probabilities for positive class = prompt muon)
    scores = model.predict_proba(X_scaled)[:, 1]  # probability of prompt muon
    
    # Reshape back to awkward structure (per-muon)
    scores = ak.unflatten(scores, num_muons)
    
    return scores

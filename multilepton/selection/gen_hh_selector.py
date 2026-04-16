from columnflow.selection import Selector, SelectionResult, selector
from columnflow.columnar_util import set_ak_column, full_like
from columnflow.util import maybe_import

import law

np = maybe_import("numpy")
ak = maybe_import("awkward")
logger = law.logger.get_logger(__name__)


@selector(
    uses={
        "GenPart.{pt,eta,phi,mass,pdgId,genPartIdxMother}",
    },
    produces={
        "hh_decay_mode",
        "mHH_gen",
        "ptHH",
        "ptH1",
        "ptH2",
        "acoplanarity",
        "costheta_star",
    },
    exposed=False,
)
def hh_truth_selector(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:

    gen = events.GenPart

    # ---------------------------------------------------------
    # 1️⃣ Identify Higgs bosons
    # ---------------------------------------------------------

    higgs = gen[np.abs(gen.pdgId) == 25]
    higgs_padded = ak.pad_none(higgs, 2)

    h1 = higgs_padded[:, 0]
    h2 = higgs_padded[:, 1]

    # ---------------------------------------------------------
    # 2️⃣ Di-Higgs kinematics
    # ---------------------------------------------------------

    hh_system = h1 + h2

    mHH = ak.fill_none(hh_system.mass, -999.0)
    ptHH = ak.fill_none(hh_system.pt, -999.0)
    ptH1 = ak.fill_none(h1.pt, -999.0)
    ptH2 = ak.fill_none(h2.pt, -999.0)

    # Acoplanarity
    delta_phi = np.abs(h1.phi - h2.phi)
    delta_phi = np.where(delta_phi > np.pi, 2*np.pi - delta_phi, delta_phi)
    acoplanarity = 1.0 - delta_phi / np.pi
    acoplanarity = ak.fill_none(acoplanarity, -999.0)

    # cos(theta*) approximation
    delta_eta = h1.eta - h2.eta
    costheta_star = np.tanh(delta_eta / 2.0)
    costheta_star = ak.fill_none(costheta_star, -999.0)

    # ---------------------------------------------------------
    # 3️⃣ Higgs decay classification 
    # ---------------------------------------------------------

    mother_idx = gen.genPartIdxMother
    higgs_mask = (gen.pdgId == 25)

    # Identify direct daughters of Higgs
    is_higgs_daughter = higgs_mask[mother_idx]

    # Count vector bosons and taus from Higgs
    nW = ak.sum(is_higgs_daughter & (np.abs(gen.pdgId) == 24), axis=1)
    nZ = ak.sum(is_higgs_daughter & (np.abs(gen.pdgId) == 23), axis=1)
    nTau = ak.sum(is_higgs_daughter & (np.abs(gen.pdgId) == 15), axis=1)

    # Convert to numpy for classification
    nW_np = ak.to_numpy(nW)
    nZ_np = ak.to_numpy(nZ)
    nTau_np = ak.to_numpy(nTau)

    decay_mode = np.full(len(events), "other", dtype="U20")

    # 4V sample
    decay_mode = np.where((nW_np == 4) & (nZ_np == 0), "HH_WWWW", decay_mode)
    decay_mode = np.where((nW_np == 2) & (nZ_np == 2), "HH_WWZZ", decay_mode)
    decay_mode = np.where((nW_np == 0) & (nZ_np == 4), "HH_ZZZZ", decay_mode)

    # 2V2Tau sample
    decay_mode = np.where((nW_np == 2) & (nTau_np == 2), "HH_WWtautau", decay_mode)
    decay_mode = np.where((nZ_np == 2) & (nTau_np == 2), "HH_ZZtautau", decay_mode)

    # 4Tau
    decay_mode = np.where((nTau_np == 4), "HH_tautautautau", decay_mode)

    # ---------------------------------------------------------
    # Store outputs
    # ---------------------------------------------------------

    events = set_ak_column(events, "hh_decay_mode", decay_mode)
    events = set_ak_column(events, "mHH_gen", mHH)
    events = set_ak_column(events, "ptHH", ptHH)
    events = set_ak_column(events, "ptH1", ptH1)
    events = set_ak_column(events, "ptH2", ptH2)
    events = set_ak_column(events, "acoplanarity", acoplanarity)
    events = set_ak_column(events, "costheta_star", costheta_star)

    return events, SelectionResult(
        steps={
            "hh_truth": full_like(events.event, True, dtype=bool),
        }
    )

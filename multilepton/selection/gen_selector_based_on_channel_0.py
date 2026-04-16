# coding: utf-8

"""
Gen-matching selection: match selected leptons/taus to generator-level particles.
"""

from __future__ import annotations

import law

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.columnar_util import set_ak_column, full_like
from columnflow.util import maybe_import

np = maybe_import("numpy")
ak = maybe_import("awkward")
logger = law.logger.get_logger(__name__)


def _delta_phi(phi1: ak.Array, phi2: ak.Array) -> ak.Array:
    return np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))


def _match_by_dr(reco: ak.Array, gen: ak.Array, dr_max: float) -> ak.Array:
    if len(gen) == 0:
        return ak.zeros_like(reco.pt, dtype=bool)
    dphi = _delta_phi(reco.phi[:, :, None], gen.phi[:, None, :])
    deta = reco.eta[:, :, None] - gen.eta[:, None, :]
    dr = np.sqrt(dphi * dphi + deta * deta)
    return ak.any(dr < dr_max, axis=2)


def _get_gen_props(reco: ak.Array, gen_part: ak.Array) -> tuple[ak.Array, ak.Array, ak.Array]:
    """Return (is_matched, gen_pdg, gen_pt) for reco objects using genPartIdx."""
    fake_array = ak.fill_none(ak.zeros_like(reco.pt, dtype=bool), False)
    if not hasattr(reco, "genPartIdx"):
        return fake_array, ak.zeros_like(reco.pt), ak.zeros_like(reco.pt)

    gen_idx = reco.genPartIdx
    is_matched = gen_idx >= 0
    gen_idx_safe = ak.where(is_matched, gen_idx, 0)
    gen_pdg = ak.where(
        is_matched & (gen_idx_safe < ak.num(gen_part.pdgId)),
        gen_part.pdgId[gen_idx_safe],
        0,
    )
    gen_pt = ak.where(
        is_matched & (gen_idx_safe < ak.num(gen_part.pt)),
        gen_part.pt[gen_idx_safe],
        0,
    )
    return is_matched, gen_pdg, gen_pt


def _get_charge_flip_mask(reco: ak.Array, gen_pdg: ak.Array) -> ak.Array:
    if not hasattr(reco, "charge"):
        return ak.zeros_like(gen_pdg, dtype=bool)
    reco_charge = reco.charge
    # Use np.sign to avoid division-by-zero warnings; sign(0)==0 naturally
    gen_charge = np.sign(gen_pdg)
    return (reco_charge != 0) & (gen_charge != 0) & (reco_charge != gen_charge)



@selector(
    uses={
        "GenPart.{pt,pdgId}",
        "GenVisTau.{pt,eta,phi}",
        "GenJet.{pt,eta,phi}",
        "Electron.genPartIdx",
        "Electron.charge",
        "Muon.genPartIdx",
        "Muon.charge",
        "Tau.genPartIdx",
        "Tau.charge",
        "channel_id",
        "ElectronLoose", "MuonLoose", "TauIso",
        "Jet.btagPNetB",
    },
    produces={
        "gen_match_category",                    # fakes/flips/conversions/nonfakes
        "passes_bveto",                          # b-jet veto flag (same WP as categorization)
        "gen_match_tau_category",                # gentau/faketau/none
        "selLeptons_numGenMatchedLeptons",
        "selLeptons_numChargeFlippedGenMatchedLeptons",
        "selLeptons_numGenMatchedPhotons",
        "selLeptons_numGenMatchedHadTaus",
        "selLeptons_numGenMatchedJets",
        "selHadTaus_numGenMatchedHadTaus",
        "selHadTaus_numChargeFlippedGenMatchedHadTaus",
        "selHadTaus_numGenMatchedElectrons",
        "selHadTaus_numGenMatchedMuons",
        "selHadTaus_numGenMatchedJets",
    },
    exposed=False,
)
def gen_matching_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """
    Run2-style gen-matching classification:
    - leptons matched to gen leptons/photons/hadTaus/jets
    - hadTaus matched to gen hadTaus/electrons/muons/jets
    - optional charge-flip handling
    - event category: fakes, flips, conversions, nonfakes (+ gentau/faketau)
    """
    
    # Config flags (default to False unless set)
    apply_lepton_gen_matching = bool(self.config_inst.x("apply_lepton_gen_matching", True))
    apply_hadTau_gen_matching = bool(self.config_inst.x("apply_hadTau_gen_matching", True))
    use_flips = bool(self.config_inst.x("use_flips", False))
    use_flips_hadtau = bool(self.config_inst.x("use_flips_hadtau", False))
    use_gen_tau_and_fake_tau = bool(self.config_inst.x("use_gen_tau_and_fake_tau", False))

    # Channel IDs from config
    channel_ids = {ch.name: ch.id for ch in self.config_inst.channels}
    channel_id = events.channel_id

    # Get all leptons and gen particles
    e_sel = events.ElectronLoose
    mu_sel = events.MuonLoose
    tau_sel = events.TauIso
    gen_part = events.GenPart

    # Lepton matching (Run2 logic — NO mother tracing, purely gen-match type)
    e_is_matched, e_gen_pdg, e_gen_pt = _get_gen_props(e_sel, gen_part)
    mu_is_matched, mu_gen_pdg, _      = _get_gen_props(mu_sel, gen_part)
    tau_is_matched, tau_gen_pdg, _    = _get_gen_props(tau_sel, gen_part)

    # ── Electrons ────────────────────────────────────────────────────────────
    # Genuine: matched to gen e (11) or gen tau (15)
    e_is_lepton  = e_is_matched & (np.abs(e_gen_pdg) == 11)
    e_is_gentau  = e_is_matched & (np.abs(e_gen_pdg) == 15)
    # Conversion: matched to gen photon (22) with pt > 0.5 * reco pt
    e_is_photon  = (
        e_is_matched
        & ~e_is_lepton
        & ~e_is_gentau
        & (np.abs(e_gen_pdg) == 22)
        & (e_gen_pt > (0.5 * e_sel.pt))
    )
    # Fake: everything else (unmatched, other pdgId, low-pt photon, ...)
    e_is_jet     = ~(e_is_lepton | e_is_gentau | e_is_photon)

    # ── Muons ────────────────────────────────────────────────────────────────
    # Genuine: matched to gen mu (13) or gen tau (15)
    mu_is_lepton = mu_is_matched & (np.abs(mu_gen_pdg) == 13)
    mu_is_gentau = mu_is_matched & (np.abs(mu_gen_pdg) == 15)
    mu_is_photon = ak.zeros_like(mu_is_matched, dtype=bool)   # muons have no photon conversion
    mu_is_jet    = ~(mu_is_lepton | mu_is_gentau)

    # ── HadTau matching (priority order via DR) ───────────────────────────────
    # 1. ΔR < 0.3 to GenVisTau → genuine hadtau
    tau_match_genvistau = _match_by_dr(tau_sel, events.GenVisTau, dr_max=0.3)
    # 2. gen e or gen mu (via genPartIdx) → lepton→tau fake
    tau_is_e    = tau_is_matched & ~tau_match_genvistau & (np.abs(tau_gen_pdg) == 11)
    tau_is_mu   = tau_is_matched & ~tau_match_genvistau & ~tau_is_e & (np.abs(tau_gen_pdg) == 13)
    # 3. ΔR < 0.3 to GenJet → jet→tau fake
    tau_match_genjet = _match_by_dr(tau_sel, events.GenJet, dr_max=0.3)
    tau_is_jet  = ~tau_match_genvistau & ~tau_is_e & ~tau_is_mu & tau_match_genjet
    # 4. Everything else → fake (unclassified)
    tau_is_hadtau = tau_match_genvistau
    tau_is_other_fake = ~tau_is_hadtau & ~tau_is_e & ~tau_is_mu & ~tau_is_jet
    # consolidate: jet category absorbs other_fake too
    tau_is_jet  = tau_is_jet | tau_is_other_fake

    # Per-type genuine masks (non-jet)
    e_is_genuine   = ~e_is_jet
    mu_is_genuine  = ~mu_is_jet
    tau_is_genuine = ~tau_is_jet

    # Charge flips
    e_is_flip = _get_charge_flip_mask(e_sel, e_gen_pdg)
    mu_is_flip = _get_charge_flip_mask(mu_sel, mu_gen_pdg)
    tau_is_flip = _get_charge_flip_mask(tau_sel, tau_gen_pdg)

    # Event-level counts
    selLeptons_numGenMatchedLeptons = ak.sum(e_is_lepton, axis=1) + ak.sum(mu_is_lepton, axis=1)
    selLeptons_numChargeFlippedGenMatchedLeptons = ak.sum(e_is_flip, axis=1) + ak.sum(mu_is_flip, axis=1)
    selLeptons_numGenMatchedPhotons = ak.sum(e_is_photon, axis=1) + ak.sum(mu_is_photon, axis=1)
    # leptons matched to gen tau (e.g. lepton from tau decay mistaken for prompt lepton)
    selLeptons_numGenMatchedHadTaus = ak.sum(e_is_gentau, axis=1) + ak.sum(mu_is_gentau, axis=1)
    selLeptons_numGenMatchedJets = ak.sum(e_is_jet, axis=1) + ak.sum(mu_is_jet, axis=1)

    selHadTaus_numGenMatchedHadTaus = ak.sum(tau_is_hadtau, axis=1)
    selHadTaus_numChargeFlippedGenMatchedHadTaus = ak.sum(tau_is_flip, axis=1)
    selHadTaus_numGenMatchedElectrons = ak.sum(tau_is_e, axis=1)
    selHadTaus_numGenMatchedMuons = ak.sum(tau_is_mu, axis=1)
    selHadTaus_numGenMatchedJets = ak.sum(tau_is_jet, axis=1)

    # Event category (Run2 priority)
    is_fakes = (
        (apply_lepton_gen_matching & (selLeptons_numGenMatchedJets >= 1)) |
        (apply_hadTau_gen_matching & (selHadTaus_numGenMatchedJets >= 1))
    )
    is_flips = (
        (use_flips & (selLeptons_numChargeFlippedGenMatchedLeptons >= 1)) |
        (use_flips_hadtau & (selHadTaus_numChargeFlippedGenMatchedHadTaus >= 1))
    )
    is_conversions = selLeptons_numGenMatchedPhotons >= 1

    gen_match_category_np = np.full(len(selLeptons_numGenMatchedLeptons), "nonfakes", dtype="U16")
    is_conversions_np = ak.to_numpy(is_conversions)
    is_flips_np = ak.to_numpy(is_flips)
    is_fakes_np = ak.to_numpy(is_fakes)
    gen_match_category_np[is_conversions_np] = "conversions"
    gen_match_category_np[is_flips_np] = "flips"
    gen_match_category_np[is_fakes_np] = "fakes"
    gen_match_category = ak.Array(gen_match_category_np)

    gen_match_tau_category_np = np.full(len(selLeptons_numGenMatchedLeptons), "none", dtype="U16")
    if use_gen_tau_and_fake_tau:
        tau_fake_np = ak.to_numpy(selHadTaus_numGenMatchedJets >= 1)
        gen_match_tau_category_np[tau_fake_np] = "faketau"
        gen_match_tau_category_np[~tau_fake_np] = "gentau"
    gen_match_tau_category = ak.Array(gen_match_tau_category_np)

    # Convenience columns
    nFake = selLeptons_numGenMatchedJets + selHadTaus_numGenMatchedJets

    # ── B-jet veto (same working points as categorization/default.py) ─────────
    # passes_bveto = True when the event would pass the b-veto used in the SR:
    #   nLooseBjets < 2  AND  nMediumBjets < 1
    wp_loose  = self.config_inst.x.btag_working_points["particleNet"]["loose"]
    wp_medium = self.config_inst.x.btag_working_points["particleNet"]["medium"]
    tagged_loose  = events.Jet.btagPNetB > wp_loose
    tagged_medium = events.Jet.btagPNetB > wp_medium
    passes_bveto  = (ak.sum(tagged_loose, axis=1) < 2) & (ak.sum(tagged_medium, axis=1) < 1)

    # Store columns
    events = set_ak_column(events, "gen_match_category", gen_match_category)
    events = set_ak_column(events, "gen_match_tau_category", gen_match_tau_category)
    events = set_ak_column(events, "selLeptons_numGenMatchedLeptons", selLeptons_numGenMatchedLeptons)
    events = set_ak_column(
        events,
        "selLeptons_numChargeFlippedGenMatchedLeptons",
        selLeptons_numChargeFlippedGenMatchedLeptons,
    )
    events = set_ak_column(events, "selLeptons_numGenMatchedPhotons", selLeptons_numGenMatchedPhotons)
    events = set_ak_column(events, "selLeptons_numGenMatchedHadTaus", selLeptons_numGenMatchedHadTaus)
    events = set_ak_column(events, "selLeptons_numGenMatchedJets", selLeptons_numGenMatchedJets)
    events = set_ak_column(events, "selHadTaus_numGenMatchedHadTaus", selHadTaus_numGenMatchedHadTaus)
    events = set_ak_column(
        events,
        "selHadTaus_numChargeFlippedGenMatchedHadTaus",
        selHadTaus_numChargeFlippedGenMatchedHadTaus,
    )
    events = set_ak_column(events, "selHadTaus_numGenMatchedElectrons", selHadTaus_numGenMatchedElectrons)
    events = set_ak_column(events, "selHadTaus_numGenMatchedMuons", selHadTaus_numGenMatchedMuons)
    events = set_ak_column(events, "selHadTaus_numGenMatchedJets", selHadTaus_numGenMatchedJets)
    events = set_ak_column(events, "passes_bveto", passes_bveto)
    
    return events, SelectionResult(
        steps={
            "gen_matching": full_like(events.event, True, dtype=bool),  # No event rejection
        },
    )
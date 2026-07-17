# coding: utf-8

"""
Gen-matching selection: classify selected leptons/taus using NanoAOD genPartFlav
(Run3-style, no GenPart/GenVisTau/GenJet dR matching).
"""

from __future__ import annotations

import law

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.columnar_util import set_ak_column, full_like
from columnflow.util import maybe_import

np = maybe_import("numpy")
ak = maybe_import("awkward")
logger = law.logger.get_logger(__name__)


# ====================================================================
# genPartFlav reference (NanoAOD v12+, Run3)
# ====================================================================
# Electron_genPartFlav:
#   0  = unmatched (jet fake)
#   1  = prompt electron               -> genuine
#  15  = from tau decay                -> genuine
#   3  = photon conversion (old conv.) -> fake (conversion)
#  22  = photon conversion             -> fake (conversion)
#
# Muon_genPartFlav:
#   0  = unmatched (jet fake)
#   1  = prompt muon                   -> genuine
#  15  = from tau decay                -> genuine
#   3  = light meson decay             -> fake
#   4  = kaon decay                    -> fake
#   5  = heavy flavor (b/c) decay      -> fake
#
# Tau_genPartFlav:
#   5  = genuine hadronic tau
#   1,3 = electron faking tau
#   2,4 = muon faking tau
#   0  = jet faking tau
# ====================================================================


@selector(
    uses={
        "Electron.genPartFlav",
        "Electron.charge",
        "Muon.genPartFlav",
        "Muon.charge",
        "Tau.genPartFlav",
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
    Run3-style gen-matching classification using genPartFlav only:
    ...
    """

    # Run only on MC
    if not self.dataset_inst.is_mc:
        return events, SelectionResult(
            steps={
                "gen_matching": full_like(events.event, True, dtype=bool),
            },
        )

    logger.info("Running Run3 genPartFlav-based gen_matching_selection")
    apply_lepton_gen_matching = bool(self.config_inst.x("apply_lepton_gen_matching", True))
    apply_hadTau_gen_matching = bool(self.config_inst.x("apply_hadTau_gen_matching", True))
    use_flips = bool(self.config_inst.x("use_flips", False))
    use_flips_hadtau = bool(self.config_inst.x("use_flips_hadtau", False))
    use_gen_tau_and_fake_tau = bool(self.config_inst.x("use_gen_tau_and_fake_tau", False))

    # Get selected objects
    e_sel = events.ElectronLoose
    mu_sel = events.MuonLoose
    tau_sel = events.TauIso

    e_flav = e_sel.genPartFlav
    mu_flav = mu_sel.genPartFlav
    tau_flav = tau_sel.genPartFlav

    # ── Electrons ────────────────────────────────────────────────────────────
    e_is_lepton = (e_flav == 1)                               # prompt e
    e_is_gentau = (e_flav == 15)                              # e from tau decay -> genuine
    e_is_photon = (e_flav == 3) | (e_flav == 22)              # conversion -> fake  # FIX E221
    e_is_jet = ~(e_is_lepton | e_is_gentau | e_is_photon)     # catch-all, includes flav 4/5 too

    # ── Muons ────────────────────────────────────────────────────────────────
    mu_is_lepton = (mu_flav == 1)                             # prompt mu
    mu_is_gentau = (mu_flav == 15)                            # mu from tau decay -> genuine  # FIX E221
    mu_is_photon = ak.zeros_like(mu_flav, dtype=bool)         # muons: no photon conversion category
    mu_is_jet = (mu_flav == 0) | (mu_flav == 3) | (mu_flav == 4) | (mu_flav == 5)

    # ── HadTaus ──────────────────────────────────────────────────────────────
    tau_is_hadtau = (tau_flav == 5)                           # genuine hadronic tau  # FIX E221
    tau_is_e = (tau_flav == 1) | (tau_flav == 3)              # e -> tau fake         # FIX E221
    tau_is_mu = (tau_flav == 2) | (tau_flav == 4)             # mu -> tau fake        # FIX E221
    tau_is_jet = (tau_flav == 0)                              # jet -> tau fake

    # FIX F841: removed unused e_is_genuine, mu_is_genuine, tau_is_genuine
    # (charge-flip not derivable from genPartFlav alone; flip counters set to zero below)

    # Charge flips: not derivable from genPartFlav alone -> set to zero
    e_is_flip = ak.zeros_like(e_flav, dtype=bool)
    mu_is_flip = ak.zeros_like(mu_flav, dtype=bool)
    tau_is_flip = ak.zeros_like(tau_flav, dtype=bool)

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

    # Event category (Run2-style priority, unchanged)
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

    # ── B-jet veto (same working points as categorization/default.py) ─────────
    # passes_bveto = True when the event would pass the b-veto used in the SR:
    #   nLooseBjets < 2  AND  nMediumBjets < 1
    # B-tagger selection depends on year (same as categorization/default.py)
    year = self.config_inst.campaign.x.year

    if year in {2024, 2025, 2026}:
        btag_tagger = "UParTAK4"
        btag_discriminator = "btagUParTAK4B"
    else:
        btag_tagger = "particleNet"
        btag_discriminator = "btagPNetB"

    wp_loose = self.config_inst.x.btag_working_points[btag_tagger]["loose"]
    wp_medium = self.config_inst.x.btag_working_points[btag_tagger]["medium"]

    btag_score = events.Jet[btag_discriminator]

    tagged_loose = btag_score > wp_loose
    tagged_medium = btag_score > wp_medium
    passes_bveto = (ak.sum(tagged_loose, axis=1) < 2) & (ak.sum(tagged_medium, axis=1) < 1)

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

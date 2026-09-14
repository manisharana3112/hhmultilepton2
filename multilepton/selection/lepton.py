# coding: utf-8

"""
Lepton selection methods.
"""

from __future__ import annotations

import os
import law

from operator import or_
from functools import reduce
from collections import defaultdict

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.columnar_util import (
    set_ak_column, sorted_indices_from_mask, flat_np_view, full_like,
)
from columnflow.util import maybe_import
from columnflow.production.cms.jet import jet_id

from multilepton.util import (
    IF_NANO_V9, IF_NANO_GE_V10, IF_NANO_V12, IF_NANO_V14, IF_NANO_V15, IF_NOT_NANO_V15,
)
from multilepton.selection.muon_mva import compute_muon_mva_score
from multilepton.selection.electron_mva import compute_electron_mva_score
from multilepton.config.util import Trigger

np = maybe_import("numpy")
ak = maybe_import("awkward")
logger = law.logger.get_logger(__name__)


class TIDGroups:
    def __init__(self, tid_tags):
        self._groups = defaultdict(list)
        for tid, tags in tid_tags.items():
            for tag in tags:
                self._groups[tag].append(tid)
            if {"cross_tau_tau", "cross_tau_tau_jet", "cross_tau_tau_vbf"} & tags:
                self._groups["cross_tau_tau_any"].append(tid)

    def __getattr__(self, name):
        # Always return empty list if missing
        return self._groups.get(name, [])

    def __getitem__(self, name):
        return self._groups.get(name, [])


def trigger_object_matching(
    vectors1: ak.Array,
    vectors2: ak.Array,
    /,
    *,
    threshold: float = 0.5,
    axis: int = 2,
    event_mask: ak.Array | type(Ellipsis) | None = None,
) -> ak.Array:
    """
    Helper to check per object in *vectors1* if there is at least one object in *vectors2* that
        leads to a delta R metric below *threshold*. The final reduction is applied over *axis* of the
    resulting metric table containing the full combinatorics. If an *event_mask* is given, the
    the matching is performed only for those events, but a full object mask with the same shape as
    that of *vectors1* is returned, which all objects set to *False* where not matching was done.
    """
    # handle event masks
    used_event_mask = event_mask is not None and event_mask is not Ellipsis
    event_mask = Ellipsis if event_mask is None else event_mask
    # delta_r for all combinations
    dr = vectors1[event_mask].metric_table(vectors2[event_mask])
    # check per element in vectors1 if there is at least one matching element in vectors2
    any_match = ak.any(dr < threshold, axis=axis)
    # expand to original shape if an event mask was given
    if used_event_mask:
        full_any_match = full_like(vectors1.pt, False, dtype=bool)
        flat_full_any_match = flat_np_view(full_any_match)
        flat_full_any_match[flat_np_view(full_any_match | event_mask)] = flat_np_view(any_match)
        any_match = full_any_match
    return any_match


def update_channel_ids(
    events: ak.Array,
    previous_channel_ids: ak.Array,
    correct_channel_id: int,
    channel_mask: ak.Array,
) -> ak.Array:
    """
    Check if the events in the is_mask can be inside the given channel
    or have already been sorted in another channel before.
    """
    events_not_in_channel = (previous_channel_ids != 0) & (previous_channel_ids != correct_channel_id)
    channel_id_overwrite = events_not_in_channel & channel_mask
    if ak.any(channel_id_overwrite):
        raise ValueError(
            "The channel_ids of some events are being set to two different values. "
            "The first event of this chunk concerned has index",
            ak.where(channel_id_overwrite)[0],
        )
    return ak.where(channel_mask, correct_channel_id, previous_channel_ids)


def get_cone_pt_from_jetidx(
    lepton_pt: ak.Array,
    lepton_eta: ak.Array,
    lepton_phi: ak.Array,
    jet_pt: ak.Array,
    jet_eta: ak.Array,
    jet_phi: ak.Array,
    closestjet_indicies: ak.Array,
    tight_mask: ak.Array,
    pfRelIso_03_or_04_all: ak.Array,
) -> ak.Array:
    """
    - if the lepton is tight:
        cone_pt = lepton_pt
    - else, if the associated jet exists and DeltaR(lepton, jet) < 0.4:
        cone_pt = 0.9 * jet_pt
    - else:
        cone_pt = 0.9 * lepton_pt * (1 + pfRelIso_03_or_04_all)
    """

    good_indicies = closestjet_indicies >= 0
    n_jets = ak.to_numpy(ak.num(jet_pt, axis=1))

    # Defining a global index for the closest jet in each event
    jet_offsets = np.cumsum(n_jets) - n_jets
    global_closestjet_indicies = closestjet_indicies + jet_offsets[:, np.newaxis]
    safe_global_closestjet_indicies = ak.where(
        good_indicies,
        global_closestjet_indicies,
        0,
    )
    flat_safe_global_closestjet_indicies = ak.to_numpy(ak.flatten(safe_global_closestjet_indicies, axis=1))
    # Flatenning closest jet associated quantities and associating them with the global indices
    flat_jet_pt = ak.flatten(jet_pt, axis=1)
    flat_jet_eta = ak.flatten(jet_eta, axis=1)
    flat_jet_phi = ak.flatten(jet_phi, axis=1)
    selected_flat_jet_pt = flat_jet_pt[flat_safe_global_closestjet_indicies]
    selected_flat_jet_eta = flat_jet_eta[flat_safe_global_closestjet_indicies]
    selected_flat_jet_phi = flat_jet_phi[flat_safe_global_closestjet_indicies]
    # Associating each lepton with the corresponding nearest jet
    # We need to unflatten the arrays, as the cone-pT need to have the same shape as lepton.pT
    lepton_counts = ak.num(lepton_pt, axis=1)
    closest_jet_pt = ak.unflatten(
        selected_flat_jet_pt,
        lepton_counts,
    )
    closest_jet_eta = ak.unflatten(
        selected_flat_jet_eta,
        lepton_counts,
    )
    closest_jet_phi = ak.unflatten(
        selected_flat_jet_phi,
        lepton_counts,
    )
    # Now we define the lepton's nearby jet kinematics in the jagged structure
    closest_jet_pt = ak.where(
        good_indicies,
        closest_jet_pt,
        lepton_pt,
    )
    closest_jet_eta = ak.where(
        good_indicies,
        closest_jet_eta,
        lepton_eta,
    )
    closest_jet_phi = ak.where(
        good_indicies,
        closest_jet_phi,
        lepton_phi,
    )
    # Computing DeltaR (metric table does not work here)
    delta_phi = lepton_phi - closest_jet_phi
    delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi
    delta_eta = lepton_eta - closest_jet_eta
    closest_jet_DR = np.sqrt(delta_eta**2 + delta_phi**2)
    # See if the nearby jet is within a 0.4 cone with respect to the lepton
    has_nearby_jet = good_indicies & (closest_jet_DR < 0.4)
    # If lepton is tight or there are no nearby jets, cone-pT resolves to lepton pT
    cone_pt = ak.where(
        tight_mask,
        lepton_pt,
        ak.where(
            has_nearby_jet,
            0.9 * closest_jet_pt,
            0.9 * lepton_pt * (1 + pfRelIso_03_or_04_all),
        ),
    )

    return cone_pt


def hzz_iso_wp(electron, cuts=None):
    """
    Rebuilds Electron_mvaIso_WPHZZ from Electron_mvaHZZIso, for nano versions that contains the
    score but not the WP flag (e.g. v12).
    """
    # mvaEleID-Winter22-HZZ-V1 working point, from cms-sw/cmssw
    # RecoEgamma/ElectronIdentification/python/Identification/mvaElectronID_Winter22_HZZ_V1_cff.py
    # nano stores tanh(raw) in Electron_mvaHZZIso (see MVAValueMapProducer.h in CMSSW)
    if cuts is None:
        cuts = np.tanh([
            1.633973689084034,    # EB1, 5 < pt < 10
            1.5499076306249353,   # EB2, 5 < pt < 10
            2.0629564440753247,   # EE,  5 < pt < 10
            0.3685228146685872,   # EB1, pt >= 10
            0.2662407818935475,   # EB2, pt >= 10
            -0.5444837363886459,  # EE,  pt >= 10
        ])

    # nano stores deltaEtaSC = superCluster().eta() - eta(), so superCluster().eta()
    abs_sc_eta = abs(electron.eta + electron.deltaEtaSC)

    categories = [
        (electron.pt < 10) & (abs_sc_eta < 0.800),
        (electron.pt < 10) & (abs_sc_eta >= 0.800) & (abs_sc_eta < 1.479),
        (electron.pt < 10) & (abs_sc_eta >= 1.479),
        (electron.pt >= 10) & (abs_sc_eta < 0.800),
        (electron.pt >= 10) & (abs_sc_eta >= 0.800) & (abs_sc_eta < 1.479),
        (electron.pt >= 10) & (abs_sc_eta >= 1.479),
    ]

    passed = ak.zeros_like(electron.pt, dtype=bool)
    for category, cut in zip(categories, cuts):
        passed = passed | (category & (electron.mvaHZZIso > cut))

    return passed


@selector(
    uses={
        "Electron.{pt,eta,phi,dxy,dz}",
        "Electron.{pfRelIso03_all,seediEtaOriX,seediPhiOriY,sip3d,miniPFRelIso_all,sieie}",
        "Electron.{hoe,eInvMinusPInv,convVeto,lostHits,jetPtRelv2,jetIdx}",
        # custom electron LeptonMVA input branches: without these declared here columnflow does
        # not load them, so compute_electron_mva_score silently fed zeros -> degraded score.
        "Electron.{miniPFRelIso_chg,deltaEtaSC,mvaNoIso}", "Jet.nConstituents",
        # v2 model inputs; btagDeepFlavB is read off the matched jet (Electron.jetIdx), not a
        # per-lepton branch
        "Electron.{pfRelIso03_all,jetNDauCharged,jetPtRelv2}",
        "Jet.{pt,eta,phi,btagDeepFlavB}",
        IF_NANO_V12("Electron.{mvaTTH,mvaHZZIso}", "Jet.btagPNetB"),
        IF_NANO_V14("Electron.{promptMVA,mvaIso_WPHZZ}", "Jet.btagPNetB"),
        IF_NANO_V15("Electron.{promptMVA,mvaIso_WPHZZ}", "Jet.{btagPNetB,btagUParTAK4B}"),
        IF_NANO_V9("Electron.mvaFall17V2{Iso_WP80,Iso_WP90}"),
        IF_NANO_GE_V10("Electron.{mvaIso_WP80,mvaIso_WP90}"),
    },
    exposed=False,
)
def electron_selection(
    self: Selector,
    events: ak.Array,
    trigger: Trigger,
    **kwargs,
) -> tuple[ak.Array, ak.Array, ak.Array, ak.Array]:
    """
    Electron selection returning three sets of masks and the cone-pT.
    See https://twiki.cern.ch/twiki/bin/view/CMS/EgammaNanoAOD?rev=4
    """
    # ch_key = kwargs.get("ch_key", None)
    # is_2016 = self.config_inst.campaign.x.year == 2016
    is_2022_post = (
        self.config_inst.campaign.x.year == 2022 and
        self.config_inst.campaign.has_tag("postEE")
    )
    is_single = trigger.has_tag("single_e") or trigger.has_tag("single_mu")
    is_cross = trigger.has_tag("cross_e_tau")

    if self.config_inst.campaign.x.year in {2024, 2025, 2026}:
        btag_tagger = "UParTAK4"
        btag_discriminator = "btagUParTAK4B"
    else:
        btag_tagger = "particleNet"
        btag_discriminator = "btagPNetB"

    # btagcut_loose = self.config_inst.x.btag_working_points[btag_tagger]["loose"]
    # btagcut_medium = self.config_inst.x.btag_working_points[btag_tagger]["medium"]
    btagcut_tight = self.config_inst.x.btag_working_points[btag_tagger]["tight"]

    # obtain mva flags, which might be located at different routes, depending on the nano version
    if "mvaIso_WP80" in events.Electron.fields:
        # >= nano v10
        # beware that the available Iso should be mvaFall17V2 for run2 files, not Winter22V1,
        # check this in original root files if necessary
        mva_iso_wp80 = events.Electron.mvaIso_WP80
        mva_iso_wp90 = events.Electron.mvaIso_WP90
        if "mvaIso_WPHZZ" in events.Electron.fields:
            mva_iso_wphzz = events.Electron.mvaIso_WPHZZ
        elif "mvaHZZIso" in events.Electron.fields:
            # v12 carry the score but not the WPHZZ flag, so we apply the WP by hand
            mva_iso_wphzz = hzz_iso_wp(events.Electron)
        else:
            mva_iso_wphzz = None
    else:
        # <= nano v9
        mva_iso_wp80 = events.Electron.mvaFall17V2Iso_WP80
        mva_iso_wp90 = events.Electron.mvaFall17V2Iso_WP90
        mva_iso_wphzz = None

    # Get electron MVA source from config (default: "custom")
    # Options:
    #   "custom"  - XGBoost trained model from Lepton-MVA-Run3/models
    #   "nanoaod" - Default NanoAOD MVA (promptMVA for v14+, mvaTTH for v<14)
    electron_mva_source = getattr(self.config_inst.x, "electron_mva_source", "nanoaod")

    # Select electron MVA based on configured source
    if electron_mva_source == "custom":
        # Try to use custom trained XGBoost model
        try:
            promptMVA = compute_electron_mva_score(events)

        except Exception as e:
            # Fallback to NanoAOD MVA if custom model fails
            logger.warning_once(f"Failed to load custom electron MVA model ({e}), falling back to NanoAOD MVA")
            if "promptMVA" in events.Electron.fields:
                promptMVA = events.Electron.promptMVA
                logger.warning_once("Using NanoAOD promptMVA (v14+) as fallback for electron selection")
            else:
                promptMVA = events.Electron.mvaTTH
                logger.warning_once("Using NanoAOD mvaTTH (v<14) as fallback for electron selection")

    elif electron_mva_source == "nanoaod":
        # Use NanoAOD default MVA based on version
        if "promptMVA" in events.Electron.fields:
            # >= nano v14
            promptMVA = events.Electron.promptMVA
            logger.warning_once("Using NanoAOD promptMVA (v14+) for electron selection")
        else:
            # nano <v14
            promptMVA = events.Electron.mvaTTH
            logger.warning_once("Using NanoAOD mvaTTH (v<14) for electron selection")

    else:
        raise ValueError(f"Invalid electron_mva_source '{electron_mva_source}'. "
                       f"Choose from: 'custom' (XGBoost model), 'nanoaod' (version-based default)")
    # =========================================
    # MVA comparison debugging (gen-matched ROC check)
    # =========================================
    # NOTE: counting how many electrons pass a fixed *numeric* cut (e.g. 0.3) on both
    # scores is NOT a valid comparison: the custom and nanoAOD scores live on different
    # scales, so 0.3 is a different working point for each, and the sample here mixes
    # prompt (signal) and fake (background) electrons. A ROC gain only shows up when you
    # (a) separate signal/background via truth and (b) compare at a MATCHED efficiency.
    if os.environ.get("MVA_FEATURE_DEBUG") and self.dataset_inst.is_mc and "genPartFlav" in events.Electron.fields:
        try:
            custom = ak.to_numpy(ak.flatten(compute_electron_mva_score(events))).astype(np.float64)
        except Exception as cmp_e:
            logger.warning_once(f"[Comparison electron] custom electron MVA failed ({cmp_e})")
            custom = None

        # same nanoAOD score the selection actually uses
        if "promptMVA" in events.Electron.fields:
            nano = ak.to_numpy(ak.flatten(events.Electron.promptMVA)).astype(np.float64)
        else:
            nano = ak.to_numpy(ak.flatten(events.Electron.mvaTTH)).astype(np.float64)

        flav = ak.to_numpy(ak.flatten(events.Electron.genPartFlav))
        is_sig = (flav == 1)          # prompt electron -> signal
        is_bkg = (flav == 0)          # jet fake        -> background
        keep = is_sig | is_bkg        # drop tau/photon-conversion electrons for a clean 2-class ROC

        if custom is not None and is_sig.sum() > 0 and is_bkg.sum() > 0:
            y = is_sig[keep].astype(int)
            xc, xn = custom[keep], nano[keep]
            n_sig, n_bkg = int(y.sum()), int((y == 0).sum())

            # threshold-free discrimination: AUC (this is what the ROC "gain" reflects)
            from sklearn.metrics import roc_auc_score
            auc_c, auc_n = roc_auc_score(y, xc), roc_auc_score(y, xn)

            # apples-to-apples working point: take nano's signal eff at its 0.3 cut, find the
            # custom threshold giving the SAME signal eff, then compare background efficiency
            sig_eff_nano = float((xn[y == 1] > 0.3).mean())
            bkg_eff_nano = float((xn[y == 0] > 0.3).mean())
            thr_c = float(np.quantile(xc[y == 1], 1.0 - sig_eff_nano))
            bkg_eff_custom = float((xc[y == 0] > thr_c).mean())

            logger.info_once(
                f"[Comparison electron] gen-matched n_sig={n_sig} n_bkg={n_bkg} | \n"
                f"AUC custom={auc_c:.4f} nano={auc_n:.4f} | \n"
                f"@ matched sig-eff={sig_eff_nano:.3f}: bkg-eff nano={bkg_eff_nano:.4f} \n"
                f"custom={bkg_eff_custom:.4f} (custom thr={thr_c:.3f})",
            )

            # ---- event-level yield comparison (nano vs custom) ----
            # Count events keeping >=1 electron passing each MVA. For a FAIR comparison we use
            # nano at its analysis cut (0.3) and custom at the threshold matched to the SAME
            # signal efficiency (thr_c), so any yield difference reflects extra fake rejection,
            # not just a looser/tighter numeric cut. (Stats here are one file only.)
            n_per_evt = ak.to_numpy(ak.num(events.Electron.pt, axis=1))
            custom_jag = ak.unflatten(custom, n_per_evt)
            nano_jag = ak.unflatten(nano, n_per_evt)
            nano_thr, custom_thr = 0.3, thr_c

            n_evt = len(events)
            evt_nano = int(ak.sum(ak.any(nano_jag > nano_thr, axis=1)))
            evt_custom = int(ak.sum(ak.any(custom_jag > custom_thr, axis=1)))
            sig_keep_nano = int(((nano > nano_thr) & is_sig).sum())
            sig_keep_custom = int(((custom > custom_thr) & is_sig).sum())
            fake_keep_nano = int(((nano > nano_thr) & is_bkg).sum())
            fake_keep_custom = int(((custom > custom_thr) & is_bkg).sum())

            # S/sqrt(B) significance-like figure: at matched signal eff S is ~equal, so the gain
            # is driven by fake (B) reduction. Reported as a single number + relative gain.
            z_nano = sig_keep_nano / np.sqrt(fake_keep_nano) if fake_keep_nano > 0 else float("inf")
            z_custom = sig_keep_custom / np.sqrt(fake_keep_custom) if fake_keep_custom > 0 else float("inf")
            z_gain = (z_custom / z_nano - 1.0) * 100.0 if np.isfinite(z_nano) and z_nano > 0 else float("nan")

            logger.info_once(
                f"[Yield electron] nano>{nano_thr:.3f} vs custom>{custom_thr:.3f} (matched sig-eff) | \n"
                f"events(>=1 sel e)/{n_evt}: nano={evt_nano} custom={evt_custom} \n"
                f"(delta={evt_custom - evt_nano:+d}) | \n"
                f"prompt-e kept: nano={sig_keep_nano} custom={sig_keep_custom} | \n"
                f"fake-e kept: nano={fake_keep_nano} custom={fake_keep_custom} \n"
                f"(fakes removed by custom: {fake_keep_nano - fake_keep_custom:+d}) | \n"
                f"S/sqrt(B): nano={z_nano:.2f} custom={z_custom:.2f} (gain={z_gain:+.1f}%)",
            )
    # =========================================

    # default electron mask
    tight_mask = None
    fakeable_mask = None
    if is_single or is_cross or True:  # investigate why trigger dependence on providing masks
        # min_pt = 26.0 if is_2016 else (31.0 if is_single else 25.0)
        # max_eta = 2.5 if is_single else 2.1

        closestjet_indicies = events.Electron.jetIdx[:, :]
        bad_indicies = (closestjet_indicies == -1)  # set btag to 0 if no closest jet
        btag_pad = ak.fill_none(ak.pad_none(events.Jet[btag_discriminator], 1, axis=1), 0.0)
        btag_values = ak.where(
            bad_indicies, 0.0, btag_pad[ak.where(bad_indicies, 0, closestjet_indicies)],
        )
        abs_sc_eta = abs(events.Electron.eta + events.Electron.deltaEtaSC)
        sieie_max = ak.where(abs_sc_eta > 1.479, 0.030, 0.011)  # endcap, barrel

        atleast_loose = ((mva_iso_wp80 == 1) | (mva_iso_wp90 == 1))
        if mva_iso_wphzz is not None:
            atleast_loose = atleast_loose | (mva_iso_wphzz == 1)
        tight_mask = (
            (events.Electron.pt > 10) &
            (abs(events.Electron.eta) < 2.5) &
            (abs(events.Electron.dxy) < 0.5) &
            (abs(events.Electron.dz) < 1) &
            (events.Electron.sip3d < 8) &
            (events.Electron.miniPFRelIso_all < 0.4) &
            (events.Electron.sieie < sieie_max) &
            (events.Electron.hoe < 0.1) &
            (events.Electron.eInvMinusPInv > -0.04) &
            (events.Electron.convVeto == 1) &
            (events.Electron.lostHits == 0) &
            atleast_loose &
            (promptMVA > 0.3) &
            (btag_values < btagcut_tight)
        )

        cone_pt = get_cone_pt_from_jetidx(
            events.Electron.pt,
            events.Electron.eta,
            events.Electron.phi,
            events.Jet.pt,
            events.Jet.eta,
            events.Jet.phi,
            closestjet_indicies,
            tight_mask,
            events.Electron.pfRelIso03_all,
        )

        loose_mask = (
            (events.Electron.pt > 7.0) &
            (abs(events.Electron.eta) < 2.5) &
            (abs(events.Electron.dxy) < 0.5) &
            (abs(events.Electron.dz) < 1) &
            (events.Electron.sip3d < 8) &
            (events.Electron.miniPFRelIso_all < 0.4) &
            (events.Electron.lostHits <= 1) &
            atleast_loose
        )
        idlepmvapassed = (atleast_loose & (promptMVA > 0.3))
        idlepmvafailed = ((mva_iso_wp90 == 1) & (promptMVA <= 0.3))
        jetisolepmvapassed = (promptMVA > 0.3)
        jetisolepmvafailed = ((promptMVA <= 0.3) & (events.Electron.jetPtRelv2 < (1. / 1.7)))
        fakeable_mask = (
            (events.Electron.pt > 10) &
            (cone_pt > 10.0) &
            (abs(events.Electron.eta) < 2.5) &
            (abs(events.Electron.dxy) < 0.5) &
            (abs(events.Electron.dz) < 1) &
            (events.Electron.sip3d < 8) &
            (events.Electron.miniPFRelIso_all < 0.4) &
            (events.Electron.sieie < sieie_max) &
            (events.Electron.hoe < 0.1) &
            (events.Electron.eInvMinusPInv > -0.04) &
            (events.Electron.convVeto == 1) &
            (events.Electron.lostHits == 0) &
            (idlepmvapassed | idlepmvafailed) &
            (btag_values < btagcut_tight) &
            (jetisolepmvapassed | jetisolepmvafailed)
        )
        if is_2022_post:
            tight_mask = tight_mask & ~(
                (events.Electron.eta > 1.556) &
                (events.Electron.seediEtaOriX < 45) &
                (events.Electron.seediPhiOriY > 72)
            )
            fakeable_mask = fakeable_mask & ~(
                (events.Electron.eta > 1.556) &
                (events.Electron.seediEtaOriX < 45) &
                (events.Electron.seediPhiOriY > 72)
            )

    return tight_mask, fakeable_mask, loose_mask, cone_pt


@electron_selection.init
def electron_selection_init(self) -> None:
    if self.config_inst.campaign.x.run == 3 and self.config_inst.campaign.x.year == 2022:
        self.shifts |= {
            shift_inst.name for shift_inst in self.config_inst.shifts
            if shift_inst.has_tag(("ees", "eer"))
        }


@selector(
    uses={"{Electron,TrigObj}.{pt,eta,phi}"},
    exposed=False,
)
def electron_trigger_matching(
    self: Selector,
    events: ak.Array,
    trigger: Trigger,
    trigger_fired: ak.Array,
    leg_masks: dict[str, ak.Array],
    **kwargs,
) -> tuple[ak.Array]:
    """
    Electron trigger matching.
    """
    is_single = trigger.has_tag("single_e")
    is_cross = trigger.has_tag("cross_e_tau")

    # catch config errors
    assert is_single or is_cross
    assert trigger.n_legs == len(leg_masks) == (1 if is_single else 2)
    assert abs(trigger.legs["e"].pdg_id) == 11
    return trigger_object_matching(
        events.Electron,
        events.TrigObj[leg_masks["e"]],
        event_mask=trigger_fired,
    )


@selector(
    uses={
        "Muon.{pt,eta,phi,looseId,mediumId,tightId}",
        "Muon.{pfRelIso04_all,dxy,dz,sip3d,miniPFRelIso_all,jetPtRelv2,jetIdx}",
        # custom muon LeptonMVA input branches: without these declared here columnflow does not
        # load them, so compute_muon_mva_score silently fed zeros -> degraded score.
        "Muon.{miniPFRelIso_chg,nTrackerLayers,segmentComp,isTracker,nStations,isGlobal}",
        "Jet.nConstituents",
        # v2 model inputs; btagDeepFlavB is read off the matched jet (Muon.jetIdx), not a
        # per-lepton branch
        "Muon.{pfRelIso03_all,jetNDauCharged,jetPtRelv2}",
        "Jet.{pt,eta,phi,btagDeepFlavB}",
        IF_NANO_V12("Muon.mvaTTH", "Jet.btagPNetB"),
        IF_NANO_V14("Muon.promptMVA", "Jet.btagPNetB"),
        IF_NANO_V15("Muon.promptMVA", "Jet.{btagPNetB,btagUParTAK4B}"),
    },
    exposed=False,
)
def muon_selection(
    self: Selector,
    events: ak.Array,
    trigger: Trigger,
    **kwargs,
) -> tuple[ak.Array, ak.Array, ak.Array, ak.Array]:
    """
    Muon selection returning three sets of masks and the cone-pT.
    References:
    - Isolation working point: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonIdRun2?rev=59
    - ID und ISO : https://twiki.cern.ch/twiki/bin/view/CMS/MuonUL2017?rev=15
    relaxed for multilepton, to be replaced with lepMVA later on
    """
    # ch_key = kwargs.get("ch_key", None)
    # is_2016 = self.config_inst.campaign.x.year == 2016
    is_single = trigger.has_tag("single_mu") or trigger.has_tag("single_e")
    is_cross = trigger.has_tag("cross_mu_tau")

    if self.config_inst.campaign.x.year in {2024, 2025, 2026}:
        btag_tagger = "UParTAK4"
        btag_discriminator = "btagUParTAK4B"
    else:
        btag_tagger = "particleNet"
        btag_discriminator = "btagPNetB"

    # btagcut_loose = self.config_inst.x.btag_working_points[btag_tagger]["loose"]
    # btagcut_medium = self.config_inst.x.btag_working_points[btag_tagger]["medium"]
    btagcut_tight = self.config_inst.x.btag_working_points[btag_tagger]["tight"]

    # Get muon MVA source from config (default: "custom")
    # Options:
    #   "custom"  - XGBoost trained model from Lepton-MVA-Run3/models
    #   "nanoaod" - Default NanoAOD MVA (promptMVA for v14+, mvaTTH for v<14)
    muon_mva_source = getattr(self.config_inst.x, "muon_mva_source", "nanoaod")

    # default muon mask
    tight_mask = None
    fakeable_mask = None
    if is_single or is_cross or True:  # investigate why trigger dependence on providing masks at all
        # if is_2016:
        #    min_pt = 23.0 if is_single else 20.0
        # else:
        #    min_pt = 26.0 if is_single else 22.0

        # Select muon MVA based on configured source
        if muon_mva_source == "custom":
            # Try to use custom trained XGBoost model
            try:
                promptMVA = compute_muon_mva_score(events)

            except Exception as e:
                # Fallback to NanoAOD MVA if custom model fails
                logger.warning(f"Failed to load custom muon MVA model ({e}), falling back to NanoAOD MVA")
                if "promptMVA" in events.Muon.fields:
                    promptMVA = events.Muon.promptMVA
                    logger.info_once("Using NanoAOD promptMVA (v14+) as fallback for muon selection")
                else:
                    promptMVA = events.Muon.mvaTTH
                    logger.info_once("Using NanoAOD mvaTTH (v<14) as fallback for muon selection")

        elif muon_mva_source == "nanoaod":
            # Use NanoAOD default MVA based on version
            if "promptMVA" in events.Muon.fields:
                # >= nano v14
                promptMVA = events.Muon.promptMVA
                logger.info_once("Using NanoAOD promptMVA (v14+) for muon selection")
            else:
                # nano <v14
                promptMVA = events.Muon.mvaTTH
                logger.info_once("Using NanoAOD mvaTTH (v<14) for muon selection")

        else:
            raise ValueError(f"Invalid muon_mva_source '{muon_mva_source}'. "
                           f"Choose from: 'custom' (XGBoost model), 'nanoaod' (version-based default)")

        # =========================================
        # MVA comparison debugging (gen-matched ROC check) -- mirrors the electron block.
        # Judge success by AUC / fake-rate at matched signal efficiency, NOT by counts at a
        # fixed numeric cut (the two scores are on different scales). Muon analysis cut is 0.5.
        # =========================================
        debug = True
        if debug and self.dataset_inst.is_mc and "genPartFlav" in events.Muon.fields:
            try:
                mu_custom = ak.to_numpy(ak.flatten(compute_muon_mva_score(events))).astype(np.float64)
            except Exception as cmp_e:
                logger.warning_once(f"[Comparison muon] custom muon MVA failed ({cmp_e})")
                mu_custom = None

            # same nanoAOD score the selection actually uses
            if "promptMVA" in events.Muon.fields:
                mu_nano = ak.to_numpy(ak.flatten(events.Muon.promptMVA)).astype(np.float64)
            else:
                mu_nano = ak.to_numpy(ak.flatten(events.Muon.mvaTTH)).astype(np.float64)

            mu_flav = ak.to_numpy(ak.flatten(events.Muon.genPartFlav))
            mu_is_sig = (mu_flav == 1)          # prompt muon -> signal
            mu_is_bkg = (mu_flav == 0)          # jet fake    -> background
            mu_keep = mu_is_sig | mu_is_bkg

            if mu_custom is not None and mu_is_sig.sum() > 0 and mu_is_bkg.sum() > 0:
                y = mu_is_sig[mu_keep].astype(int)
                xc, xn = mu_custom[mu_keep], mu_nano[mu_keep]
                n_sig, n_bkg = int(y.sum()), int((y == 0).sum())

                from sklearn.metrics import roc_auc_score
                auc_c, auc_n = roc_auc_score(y, xc), roc_auc_score(y, xn)

                # nano's signal eff at its 0.5 cut, then the custom threshold giving the SAME
                # signal eff -> compare background efficiency
                sig_eff_nano = float((xn[y == 1] > 0.5).mean())
                bkg_eff_nano = float((xn[y == 0] > 0.5).mean())
                thr_c = float(np.quantile(xc[y == 1], 1.0 - sig_eff_nano))
                bkg_eff_custom = float((xc[y == 0] > thr_c).mean())

                logger.info_once(
                    f"[Comparison muon] gen-matched n_sig={n_sig} n_bkg={n_bkg} | \n"
                    f"AUC custom={auc_c:.4f} nano={auc_n:.4f} | \n"
                    f"@ matched sig-eff={sig_eff_nano:.3f}: bkg-eff nano={bkg_eff_nano:.4f} \n"
                    f"custom={bkg_eff_custom:.4f} (custom thr={thr_c:.3f})",
                )

                # ---- event-level yield comparison (nano vs custom) at matched signal eff ----
                n_per_evt = ak.to_numpy(ak.num(events.Muon.pt, axis=1))
                custom_jag = ak.unflatten(mu_custom, n_per_evt)
                nano_jag = ak.unflatten(mu_nano, n_per_evt)
                nano_thr, custom_thr = 0.5, thr_c

                n_evt = len(events)
                evt_nano = int(ak.sum(ak.any(nano_jag > nano_thr, axis=1)))
                evt_custom = int(ak.sum(ak.any(custom_jag > custom_thr, axis=1)))
                sig_keep_nano = int(((mu_nano > nano_thr) & mu_is_sig).sum())
                sig_keep_custom = int(((mu_custom > custom_thr) & mu_is_sig).sum())
                fake_keep_nano = int(((mu_nano > nano_thr) & mu_is_bkg).sum())
                fake_keep_custom = int(((mu_custom > custom_thr) & mu_is_bkg).sum())

                # S/sqrt(B) significance-like figure (see electron block for rationale)
                z_nano = sig_keep_nano / np.sqrt(fake_keep_nano) if fake_keep_nano > 0 else float("inf")
                z_custom = sig_keep_custom / np.sqrt(fake_keep_custom) if fake_keep_custom > 0 else float("inf")
                z_gain = (z_custom / z_nano - 1.0) * 100.0 if np.isfinite(z_nano) and z_nano > 0 else float("nan")

                logger.info_once(
                    f"[Yield muon] nano>{nano_thr:.3f} vs custom>{custom_thr:.3f} (matched sig-eff) | \n"
                    f"events(>=1 sel mu)/{n_evt}: nano={evt_nano} custom={evt_custom} \n"
                    f"(delta={evt_custom - evt_nano:+d}) | \n"
                    f"prompt-mu kept: nano={sig_keep_nano} custom={sig_keep_custom} | \n"
                    f"fake-mu kept: nano={fake_keep_nano} custom={fake_keep_custom} \n"
                    f"(fakes removed by custom: {fake_keep_nano - fake_keep_custom:+d}) | \n"
                    f"S/sqrt(B): nano={z_nano:.2f} custom={z_custom:.2f} (gain={z_gain:+.1f}%)",
                )
        # =========================================

        closestjet_indicies = events.Muon.jetIdx[:, :]
        bad_indicies = (closestjet_indicies == -1)  # set btag to 0 if no closest jet
        btag_pad = ak.fill_none(ak.pad_none(events.Jet[btag_discriminator], 1, axis=1), 0.0)
        btag_values = ak.where(
            bad_indicies, 0.0, btag_pad[ak.where(bad_indicies, 0, closestjet_indicies)],
        )
        atleast_medium = ((events.Muon.mediumId == 1) | (events.Muon.tightId == 1))
        atleast_loose = ((events.Muon.looseId == 1) | (events.Muon.mediumId == 1) | (events.Muon.tightId == 1))
        tight_mask = (
            (events.Muon.pt > 10) &
            (abs(events.Muon.eta) < 2.4) &
            (abs(events.Muon.dxy) < 0.05) &
            (abs(events.Muon.dz) < 0.1) &
            (events.Muon.sip3d < 8) &
            (events.Muon.miniPFRelIso_all < 0.4) &
            atleast_medium &
            (btag_values < btagcut_tight) &
            (promptMVA > 0.5)
        )

        cone_pt = get_cone_pt_from_jetidx(
            events.Muon.pt,
            events.Muon.eta,
            events.Muon.phi,
            events.Jet.pt,
            events.Jet.eta,
            events.Jet.phi,
            closestjet_indicies,
            tight_mask,
            events.Muon.pfRelIso04_all,
        )

        loose_mask = (
            (events.Muon.pt > 5) &
            (abs(events.Muon.eta) < 2.4) &
            (abs(events.Muon.dxy) < 0.05) &
            (abs(events.Muon.dz) < 0.1) &
            (events.Muon.sip3d < 8) &
            (events.Muon.miniPFRelIso_all < 0.4) &
            atleast_loose
        )
        fakeable_mask = (
            (events.Muon.pt > 10) &
            (cone_pt > 10) &
            (abs(events.Muon.eta) < 2.4) &
            (abs(events.Muon.dxy) < 0.05) &
            (abs(events.Muon.dz) < 0.1) &
            (events.Muon.sip3d < 8) &
            (events.Muon.miniPFRelIso_all < 0.4) &
            atleast_loose &
            (btag_values < btagcut_tight) &
            ((promptMVA > 0.5) | ((promptMVA <= 0.5) & (events.Muon.jetPtRelv2 < (1. / 1.8))))
        )

    return tight_mask, fakeable_mask, loose_mask, cone_pt


@selector(
    uses={"{Muon,TrigObj}.{pt,eta,phi}"},
    exposed=False,
)
def muon_trigger_matching(
    self: Selector,
    events: ak.Array,
    trigger: Trigger,
    trigger_fired: ak.Array,
    leg_masks: dict[str, ak.Array],
    **kwargs,
) -> tuple[ak.Array]:
    """
    Muon trigger matching.
    """
    is_single = trigger.has_tag("single_mu")
    is_cross = trigger.has_tag("cross_mu_tau")

    assert is_single or is_cross
    assert trigger.n_legs == len(leg_masks) == (1 if is_single else 2)
    assert abs(trigger.legs["mu"].pdg_id) == 13
    return trigger_object_matching(
        events.Muon,
        events.TrigObj[leg_masks["mu"]],
        event_mask=trigger_fired,
    )


@selector(
    uses={
        "Tau.{pt,eta,phi,dz,decayMode}",
        "{Electron,Muon,TrigObj}.{pt,eta,phi}",
    },
    # shifts are declared dynamically below in tau_selection_init
    exposed=False,
)
def tau_selection(
    self: Selector,
    events: ak.Array,
    trigger: Trigger,
    electron_mask: ak.Array | None,
    muon_mask: ak.Array | None,
    **kwargs,
) -> tuple[ak.Array, ak.Array]:
    """
    Tau selection returning a masks for taus that are at least VVLoose isolated (vs jet)
    and a second mask to select isolated ones, eventually to separate normal and iso inverted taus
    for QCD estimations.
    """
    # return empty mask if no tagged taus exists in the chunk
    if ak.all(ak.num(events.Tau) == 0):
        logger.info("no taus found in event chunk")
        false_mask = full_like(events.Tau.pt, False, dtype=bool)
        return false_mask, false_mask

    # is_single_e = trigger.has_tag("single_e")
    # is_single_mu = trigger.has_tag("single_mu")
    is_cross_e = trigger.has_tag("cross_e_tau")
    is_cross_mu = trigger.has_tag("cross_mu_tau")
    is_cross_tau = trigger.has_tag("cross_tau_tau")
    is_cross_tau_vbf = trigger.has_tag("cross_tau_tau_vbf")
    is_cross_tau_jet = trigger.has_tag("cross_tau_tau_jet")
    is_2016 = self.config_inst.campaign.x.year == 2016
    is_run3 = self.config_inst.campaign.x.run == 3
    tagger = self.config_inst.x.tau_tagger
    col_prefix = getattr(self.config_inst.x, "tau_tagger_column_prefix", "id")
    get_tau_tagger = lambda tag: f"{col_prefix}{tagger}VS{tag}"
    wp_config = self.config_inst.x.tau_id_working_points

    # determine minimum pt and maximum eta
    max_eta = 2.5
    base_pt = 20.0
    # if is_single_e or is_single_mu:
    if is_cross_e:
        # only existing after 2016
        min_pt = 0.0 if is_2016 else 35.0
    elif is_cross_mu:
        min_pt = 25.0 if is_2016 else 32.0
    elif is_cross_tau:
        min_pt = 40.0
    elif is_cross_tau_vbf:
        # only existing after 2016
        min_pt = 0.0 if is_2016 else 25.0
    elif is_cross_tau_jet:
        min_pt = None if not is_run3 else 35.0
    else:
        min_pt = 20.0

    # no_id mask for tagge rindependent tests
    noid_mask = (
        (abs(events.Tau.eta) < max_eta) &
        (events.Tau.pt > base_pt) &
        (abs(events.Tau.dz) < 0.2)
    )

    # Decay modes: Run 3 PNet includes DM=2, Run 2 HPS does not
    dm_modes = (0, 1, 2, 10, 11) if is_run3 else (0, 1, 10, 11)

    # base tau mask for default and qcd sideband tau (Fakeable selection)
    base_mask = noid_mask & (
        reduce(or_, [events.Tau.decayMode == mode for mode in dm_modes]) &
        (events.Tau[get_tau_tagger("jet")] >= wp_config.tau_vs_jet.vvvloose)
        # vs e and mu cuts are channel dependent and thus applied in the overall lepton selection
    )

    # remove taus with too close spatial separation to previously selected leptons
    if electron_mask is not None:
        base_mask = base_mask & ak.all(events.Tau.metric_table(events.Electron[electron_mask]) > 0.3, axis=2)
    if muon_mask is not None:
        base_mask = base_mask & ak.all(events.Tau.metric_table(events.Muon[muon_mask]) > 0.3, axis=2)

    # trigger dependent cuts
    trigger_specific_mask = base_mask & (events.Tau.pt > min_pt)
    # compute the isolation mask separately as it is used to defined (qcd) categories later on
    iso_mask = events.Tau[get_tau_tagger("jet")] >= wp_config.tau_vs_jet.tight

    return base_mask, trigger_specific_mask, iso_mask, noid_mask


@tau_selection.init
def tau_selection_init(self: Selector) -> None:
    # register tec shifts
    self.shifts |= {
        shift_inst.name
        for shift_inst in self.config_inst.shifts
        if shift_inst.has_tag("tec")
    }
    # Add columns for the right tau tagger (id prefix for DeepTau, raw prefix for PNet)
    col_prefix = getattr(self.config_inst.x, "tau_tagger_column_prefix", "id")
    tagger = self.config_inst.x.tau_tagger
    self.uses |= {
        f"Tau.{col_prefix}{tagger}VS{tag}"
        for tag in ("e", "mu", "jet")
    }


@selector(
    uses={"{Tau,TrigObj}.{pt,eta,phi}"},
    # shifts are declared dynamically below in tau_selection_init
    exposed=False,
)
def tau_trigger_matching(
    self: Selector,
    events: ak.Array,
    trigger: Trigger,
    trigger_fired: ak.Array,
    leg_masks: dict[str, ak.Array],
    **kwargs,
) -> tuple[ak.Array]:
    """
    Tau trigger matching.
    """
    if ak.all(ak.num(events.Tau) == 0):
        logger.info("no taus found in event chunk")
        return full_like(events.Tau.pt, False, dtype=bool)

    is_cross_e = trigger.has_tag("cross_e_tau")
    is_cross_mu = trigger.has_tag("cross_mu_tau")
    is_cross_tau = trigger.has_tag("cross_tau_tau")
    is_cross_tau_vbf = trigger.has_tag("cross_tau_tau_vbf")
    is_cross_tau_jet = trigger.has_tag("cross_tau_tau_jet")
    is_any_cross_tau = is_cross_tau or is_cross_tau_vbf or is_cross_tau_jet
    assert is_cross_e or is_cross_mu or is_any_cross_tau

    # start per-tau mask with trigger object matching per leg
    if is_cross_e or is_cross_mu:
        assert trigger.n_legs == len(leg_masks) == 2
        assert abs(trigger.legs["tau"].pdg_id) == 15
        # match leg 1
        return trigger_object_matching(
            events.Tau,
            events.TrigObj[leg_masks["tau"]],
            event_mask=trigger_fired,
        )

    # is_any_cross_tau
    assert trigger.n_legs == len(leg_masks) >= 2
    assert abs(trigger.legs["tau1"].pdg_id) == 15
    assert abs(trigger.legs["tau2"].pdg_id) == 15

    # match both legs
    matches_leg0 = trigger_object_matching(
        events.Tau,
        events.TrigObj[leg_masks["tau1"]],
        event_mask=trigger_fired,
    )
    matches_leg1 = trigger_object_matching(
        events.Tau,
        events.TrigObj[leg_masks["tau2"]],
        event_mask=trigger_fired,
    )

    # taus need to be matched to at least one leg, but as a side condition
    # each leg has to have at least one match to a tau
    matches = (
        (matches_leg0 | matches_leg1) &
        ak.any(matches_leg0, axis=1) &
        ak.any(matches_leg1, axis=1)
    )
    return matches


@selector(
    uses={
        electron_selection, electron_trigger_matching, muon_selection, muon_trigger_matching,
        tau_selection, tau_trigger_matching,
        "event", "{Electron,Muon,Tau}.{charge,mass}",
        # jets are needed for the ttbarMR region
        "Jet.{pt,eta,phi}",
        IF_NOT_NANO_V15("Jet.jetId"),
        IF_NANO_V12("Jet.btagPNetB"),
        IF_NANO_V14("Jet.btagPNetB"),
        IF_NANO_V15("Jet.{btagPNetB,btagUParTAK4B}"),
    },
    produces={
        electron_selection, electron_trigger_matching, muon_selection, muon_trigger_matching,
        tau_selection, tau_trigger_matching,
        # new columns
        "channel_id", "leptons_os", "tau2_isolated",
        "single_triggered", "cross_triggered",
        "trig_match", "trig_match_bdt", "matched_trigger_ids",
        "tight_sel", "tight_sel_bdt",
        "ok_bdt_eormu",
        "TauIso", "TauNoID",
        "MuonLoose", "MuonTight", "Muon.cone_pt", "Muon.muonLeptoMVA_hh",
        "ElectronLoose", "ElectronTight", "Electron.cone_pt", "Electron.electronLeptoMVA_hh",
    },
    # when True, evaluate the measurement regions instead of the physics channels
    ffmr=False,
    mr_channels={"cttbarMR", "cwzMR", "cdyMR"},
)
def lepton_selection(
    self: Selector,
    events: ak.Array,
    trigger_results: SelectionResult,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """
    Combined lepton selection.
    """
    wp_config = self.config_inst.x.tau_id_working_points
    disable_triggers = getattr(self.config_inst.x, "disable_triggers", False)
    tagger = self.config_inst.x.tau_tagger
    col_prefix = getattr(self.config_inst.x, "tau_tagger_column_prefix", "id")
    get_tau_tagger = lambda tag: f"{col_prefix}{tagger}VS{tag}"

    # get channels from the config
    print(self.config_inst)
    channels = {
        name: self.config_inst.get_channel(name)
        for name in self.config_inst.x.channel_names
    }

    # Compute and add custom muon MVA scores as output column
    try:
        muon_mva_scores = compute_muon_mva_score(events)
        events = set_ak_column(events, ("Muon", "muonLeptoMVA_hh"), muon_mva_scores)

    except Exception as e:
        print(f"Failed to compute custom muon MVA ({e}), creating dummy column with zeros")
        events = set_ak_column(events, ("Muon", "muonLeptoMVA_hh"), ak.zeros_like(events.Muon.pt))

    # Compute and add custom electron MVA scores as output column
    try:
        electron_mva_scores = compute_electron_mva_score(events)
        events = set_ak_column(events, ("Electron", "electronLeptoMVA_hh"), electron_mva_scores)

    except Exception as e:
        print(f"Failed to compute custom electron MVA ({e}), creating dummy column with zeros")
        events = set_ak_column(events, ("Electron", "electronLeptoMVA_hh"), ak.zeros_like(events.Electron.pt))

    # prepare vectors for output vectors
    false_mask = (abs(events.event) < 0)
    channel_id = np.uint32(1) * false_mask
    ok_bdt_eormu = false_mask
    tau2_isolated = false_mask
    leptons_os = false_mask
    single_triggered = false_mask
    cross_triggered = false_mask
    tight_sel = false_mask
    trig_match = false_mask
    tight_sel_bdt = false_mask
    trig_match_bdt = false_mask
    sel_electron_mask = full_like(events.Electron.pt, False, dtype=bool)
    sel_looseelectron_mask = full_like(events.Electron.pt, False, dtype=bool)
    sel_tightelectron_mask = full_like(events.Electron.pt, False, dtype=bool)
    electron_cone_pt = full_like(events.Electron.pt, np.float32(-999.0), dtype=np.float32)
    sel_muon_mask = full_like(events.Muon.pt, False, dtype=bool)
    muon_cone_pt = full_like(events.Muon.pt, np.float32(-999.0), dtype=np.float32)
    sel_loosemuon_mask = full_like(events.Muon.pt, False, dtype=bool)
    sel_tightmuon_mask = full_like(events.Muon.pt, False, dtype=bool)
    sel_tau_mask = full_like(events.Tau.pt, False, dtype=bool)
    sel_isotau_mask = full_like(events.Tau.pt, False, dtype=bool)
    sel_noid_tau_mask = full_like(events.Tau.pt, False, dtype=bool)
    leading_taus = events.Tau[:, :0]
    matched_trigger_ids = []
    lepton_part_trigger_ids = []

    # indices for sorting taus first by isolation, then by pt
    # for this, combine iso and pt values, e.g. iso 255 and pt 32.3 -> 2550032.3
    f = 1
    if len(ak.flatten(events.Tau.pt)) > 0:
        f = 10**(np.ceil(np.log10(ak.max(events.Tau.pt))) + 2)
    tau_sorting_key = events.Tau[f"raw{self.config_inst.x.tau_tagger}VSjet"] * f + events.Tau.pt
    # tau_sorting_indices = ak.argsort(tau_sorting_key, axis=-1, ascending=False)
    # perform each lepton election step separately per trigger, avoid caching
    # sel_kwargs = {**kwargs, "call_force": True}

    # ────────────────────────────────────────────────────────────────
    # 1 FIRST LOOP – build and cache masks once per fired trigger
    # ────────────────────────────────────────────────────────────────

    _trig_cache = {}
    _tid_tags = {}
    e_trig_any = full_like(events.event, False, dtype=bool)  # we OR all fired flags for single_e here
    mu_trig_any = full_like(events.event, False, dtype=bool)  # we OR all fired flags for single_mu here
    tau_trig_any = full_like(events.event, False, dtype=bool)
    e_match_any = full_like(events.Electron.pt, False, dtype=bool)
    mu_match_any = full_like(events.Muon.pt, False, dtype=bool)

    for trigger, fired, leg_masks in trigger_results.x.trigger_data:

        if not ak.any(fired):
            continue

        e_mask, e_ctrl, e_veto, e_cone_pt = self[electron_selection](events, trigger, **kwargs)
        mu_mask, mu_ctrl, mu_veto, mu_cone_pt = self[muon_selection](events, trigger, **kwargs)
        e_mask_bdt, e_ctrl_bdt, e_veto_bdt, e_cone_pt_bdt = self[electron_selection](events, trigger,
            ch_key="eormu", **kwargs)
        mu_mask_bdt, mu_ctrl_bdt, mu_veto_bdt, mu_cone_pt_bdt = self[muon_selection](events, trigger,
            ch_key="eormu", **kwargs)
        tau_mask, tau_trigger_specific_mask, tau_iso_mask, noid_tau_mask = self[tau_selection](events,
            trigger, e_veto, mu_veto, **kwargs)

        electron_cone_pt = ak.where(
            e_ctrl,
            e_cone_pt,
            electron_cone_pt,
        )
        muon_cone_pt = ak.where(
            mu_ctrl,
            mu_cone_pt,
            muon_cone_pt,
        )
        # early study tagger independendt taus
        # sel_noid_tau_mask = noid_tau_mask
        sel_noid_tau_mask = sel_noid_tau_mask | noid_tau_mask
        if trigger.has_tag({"single_e"}):
            e_match = self[electron_trigger_matching](events, trigger, fired, leg_masks, **kwargs)
            e_trig_any = e_trig_any | fired  # “any single_e fired in this event?”
            e_match_any = e_match_any | e_match  # OR electron matching across all single_e tids
        else:
            # same jagged shape as events.Electron.pt; all False means "no e matched this trigger"
            e_match = full_like(events.Electron.pt, False, dtype=bool)

        # muon matching: only for triggers with a muon leg
        if trigger.has_tag({"single_mu"}):
            mu_match = self[muon_trigger_matching](events, trigger, fired, leg_masks, **kwargs)
            mu_trig_any = mu_trig_any | fired      # “any single_mu fired in this event?”
            mu_match_any = mu_match_any | mu_match
        else:
            mu_match = full_like(events.Muon.pt, False, dtype=bool)

        if (trigger.has_tag({"cross_tau_tau"}) or trigger.has_tag({"cross_tau_tau_vbf"}) or
                trigger.has_tag({"cross_tau_tau_jet"}) or trigger.has_tag({"cross_e_tau"}) or
                trigger.has_tag({"cross_mu_tau"})):
            tau_match = self[tau_trigger_matching](events, trigger, fired, leg_masks, **kwargs)
            tau_trig_any = tau_trig_any | fired
        else:
            tau_match = full_like(events.Tau.pt, False, dtype=bool)

        tid = trigger.id  # caching information particular to any trigger id
        _tid_tags[tid] = set(trigger.tags)
        _trig_cache.update({
            (tid, "e"): e_mask, (tid, "e_ctrl"): e_ctrl, (tid, "e_veto"): e_veto,
            (tid, "mu"): mu_mask, (tid, "mu_ctrl"): mu_ctrl, (tid, "mu_veto"): mu_veto,
            (tid, "e_match"): e_match, (tid, "mu_match"): mu_match,
            (tid, "tau_mask"): tau_mask,
            (tid, "noid_tau_mask"): noid_tau_mask,
            (tid, "tau_match"): tau_match,
            (tid, "tau_iso_mask"): tau_iso_mask,
            (tid, "e_ctrl_bdt"): e_ctrl_bdt, (tid, "e_mask_bdt"): e_mask_bdt, (tid, "e_veto_bdt"): e_veto_bdt,
            (tid, "mu_ctrl_bdt"): mu_ctrl_bdt, (tid, "mu_mask_bdt"): mu_mask_bdt, (tid, "mu_veto_bdt"): mu_veto_bdt,
            (tid, "fired"): fired,
        })

    # Now it is useful to define orthogonal masks: events trigger only on single electrons or single muons
    e_only = e_trig_any & ~mu_trig_any  # only single_e fired
    mu_only = mu_trig_any & ~e_trig_any  # only single_mu fired
    both_families = e_trig_any & mu_trig_any  # both fired
    # Addapted logic for channels with all flavours
    e_only_emutau = e_trig_any & ~mu_trig_any & ~tau_trig_any
    mu_only_emutau = mu_trig_any & ~e_trig_any & ~tau_trig_any
    cross_e_tau_only = tau_trig_any & ~e_trig_any & ~mu_trig_any

    # all_tids = list(_tid_tags.keys())
    tids = TIDGroups(_tid_tags)
    _trig_cache.update({
        # set of events that have triggered at least one single_e trigger
        ("fam", "e_trig_any"): e_trig_any,
        # set of events that have triggered at least one single_mu trigger
        ("fam", "mu_trig_any"): mu_trig_any,
        ("fam", "tau_trig_any"): tau_trig_any,
        # set of events that have triggered at least one single_e trigger and no one single_mu trigger
        ("fam", "e_only"): e_only,
        # set of events that have triggered at least one single_mu trigger and no one single_e trigger
        ("fam", "mu_only"): mu_only,
        # set of events that have triggered at least one singe_e and single_mu trigger
        ("fam", "both_families"): both_families,
        ("fam", "e_only_emutau"): e_only_emutau,
        ("fam", "mu_only_emutau"): mu_only_emutau,
        ("fam", "cross_e_tau_only"): cross_e_tau_only,
        # Electrons that have matched a single_e trigger object
        ("fam", "e_match_any"): e_match_any,
        ("fam", "mu_match_any"): mu_match_any,
    })

    data_stream = self.dataset_inst.name.split("_")[1] if self.dataset_inst.is_data else None

    # ensuring the v15 jetID fix required for the ttbar mr jet selection
    if self.ffmr and self.config_inst.x.jet_id_has_multiplicity:
        events = self[jet_id](events, **kwargs)

    # ────────────────────────────────────────────────────────────────
    # 2 SECOND LOOP – evaluate every physics channel once
    # ────────────────────────────────────────────────────────────────
    for ch_key, spec in channels.items():

        # the measurement regions overlap several physics channels, so the two cannot run in the
        # same pass, they would collide in channel_id and in the leptons_os / tight_sel columns
        if (ch_key in self.mr_channels) != self.ffmr:
            continue

        # eormu : set trig_ids to any particular trigger, so that eormu does not run over all triggers
        if ch_key in {"ceormu"}:
            if self.dataset_inst.is_mc:
                trig_ids = tids.single_e
            else:
                continue

        # 3l0th + 3l1th + 4l: single, double, and triple lepton triggers
        elif ch_key in {"c3e", "c4e"}:
            if self.dataset_inst.is_mc or data_stream == "e":
                trig_ids = tids.single_e + tids.double_e + tids.triple_e
            else:
                continue

        elif ch_key in {"c3etau"}:
            if self.dataset_inst.is_mc or data_stream == "e":
                trig_ids = tids.single_e + tids.double_e + tids.triple_e + tids.cross_e_tau
            else:
                continue

        elif ch_key in {"c3mu", "c4mu"}:
            if self.dataset_inst.is_mc or data_stream == "mu":
                trig_ids = tids.single_mu + tids.double_mu + tids.triple_mu
            else:
                continue

        elif ch_key in {"c3mutau"}:
            if self.dataset_inst.is_mc or data_stream == "mu":
                trig_ids = tids.single_mu + tids.double_mu + tids.triple_mu + tids.cross_mu_tau
            else:
                continue

        elif ch_key in {"c2e2mu"}:
            if self.dataset_inst.is_mc:
                trig_ids = (tids.single_e + tids.single_mu + tids.double_e + tids.double_mu +
                            tids.double_emu + tids.triple_eemu + tids.triple_emumu)
            elif data_stream == "muoneg":
                trig_ids = tids.double_emu + tids.triple_emumu + tids.triple_eemu
            elif data_stream == "mu":
                trig_ids = tids.single_mu + tids.double_mu
            elif data_stream == "e":
                trig_ids = tids.single_e + tids.double_e
            else:
                continue

        elif ch_key in {"c3emu"}:
            if self.dataset_inst.is_mc:
                trig_ids = (tids.single_e + tids.single_mu + tids.double_e +
                            tids.double_emu + tids.triple_e + tids.triple_eemu)
            elif data_stream == "muoneg":
                trig_ids = tids.double_emu + tids.triple_eemu
            elif data_stream == "e":
                trig_ids = tids.single_e + tids.double_e + tids.triple_e
            elif data_stream == "mu":
                trig_ids = tids.single_mu
            else:
                continue

        elif ch_key in {"ce3mu"}:
            if self.dataset_inst.is_mc:
                trig_ids = (tids.single_e + tids.single_mu + tids.double_mu +
                            tids.double_emu + tids.triple_mu + tids.triple_emumu)
            elif data_stream == "muoneg":
                trig_ids = tids.double_emu + tids.triple_emumu
            elif data_stream == "mu":
                trig_ids = tids.single_mu + tids.double_mu + tids.triple_mu
            elif data_stream == "e":
                trig_ids = tids.single_e
            else:
                continue

        elif ch_key in {"c2emu"}:
            if self.dataset_inst.is_mc:
                trig_ids = tids.single_e + tids.single_mu + tids.double_e + tids.double_emu + tids.triple_eemu
            elif data_stream == "muoneg":
                trig_ids = tids.double_emu + tids.triple_eemu
            elif data_stream == "e":
                trig_ids = tids.single_e + tids.double_e
            elif data_stream == "mu":
                trig_ids = tids.single_mu
            else:
                continue

        elif ch_key in {"c2emutau"}:
            if self.dataset_inst.is_mc:
                trig_ids = (tids.single_e + tids.single_mu + tids.double_e +
                            tids.double_emu + tids.triple_eemu + tids.cross_e_tau + tids.cross_mu_tau)
            elif data_stream == "muoneg":
                trig_ids = tids.double_emu + tids.triple_eemu
            elif data_stream == "e":
                trig_ids = tids.single_e + tids.double_e + tids.cross_e_tau
            elif data_stream == "mu":
                trig_ids = tids.single_mu + tids.cross_mu_tau
            else:
                continue

        elif ch_key in {"ce2mu"}:
            if self.dataset_inst.is_mc:
                trig_ids = tids.single_e + tids.single_mu + tids.double_mu + tids.double_emu + tids.triple_emumu
            elif data_stream == "muoneg":
                trig_ids = tids.double_emu + tids.triple_emumu
            elif data_stream == "mu":
                trig_ids = tids.single_mu + tids.double_mu
            elif data_stream == "e":
                trig_ids = tids.single_e
            else:
                continue

        elif ch_key in {"ce2mutau"}:
            if self.dataset_inst.is_mc:
                trig_ids = (tids.single_e + tids.single_mu + tids.double_mu +
                            tids.double_emu + tids.triple_emumu + tids.cross_e_tau + tids.cross_mu_tau)
            elif data_stream == "muoneg":
                trig_ids = tids.double_emu + tids.triple_emumu
            elif data_stream == "mu":
                trig_ids = tids.single_mu + tids.double_mu + tids.cross_mu_tau
            elif data_stream == "e":
                trig_ids = tids.single_e + tids.cross_e_tau
            else:
                continue

        # 2l0or1tau + 2l2th: single, double mixed lepton triggers
        elif ch_key in {"c2eSS"}:
            if self.dataset_inst.is_mc or data_stream == "e":
                trig_ids = tids.single_e + tids.double_e
            else:
                continue

        elif ch_key in {"c2eSS1tau"}:
            if self.dataset_inst.is_mc or data_stream == "e":
                trig_ids = tids.single_e + tids.double_e + tids.cross_e_tau
            else:
                continue

        elif ch_key in {"c2e2tau"}:
            if self.dataset_inst.is_mc:
                trig_ids = tids.single_e + tids.double_e + tids.cross_e_tau + tids.cross_tau_tau_any
            elif data_stream == "e":
                trig_ids = tids.single_e + tids.double_e + tids.cross_e_tau
            elif data_stream == "tau":
                trig_ids = tids.cross_tau_tau_any
            else:
                continue

        elif ch_key in {"c2muSS"}:
            if self.dataset_inst.is_mc or data_stream == "mu":
                trig_ids = tids.single_mu + tids.double_mu
            else:
                continue

        elif ch_key in {"c2muSS1tau"}:
            if self.dataset_inst.is_mc or data_stream == "mu":
                trig_ids = tids.single_mu + tids.double_mu + tids.cross_mu_tau
            else:
                continue

        elif ch_key in {"c2mu2tau"}:
            if self.dataset_inst.is_mc:
                trig_ids = tids.single_mu + tids.double_mu + tids.cross_mu_tau + tids.cross_tau_tau_any
            elif data_stream == "mu":
                trig_ids = tids.single_mu + tids.double_mu + tids.cross_mu_tau
            elif data_stream == "tau":
                trig_ids = tids.cross_tau_tau_any
            else:
                continue

        elif ch_key in {"cemu2tau"}:
            if self.dataset_inst.is_mc:
                trig_ids = (tids.single_e + tids.single_mu + tids.double_emu +
                            tids.cross_e_tau + tids.cross_mu_tau + tids.cross_tau_tau_any)
            elif data_stream == "muoneg":
                trig_ids = tids.double_emu
            elif data_stream == "e":
                trig_ids = tids.single_e + tids.cross_e_tau
            elif data_stream == "mu":
                trig_ids = tids.single_mu + tids.cross_mu_tau
            elif data_stream == "tau":
                trig_ids = tids.cross_tau_tau_any
            else:
                continue

        elif ch_key in {"cemuSS"}:
            if self.dataset_inst.is_mc:
                trig_ids = tids.single_e + tids.single_mu + tids.double_emu
            elif data_stream == "muoneg":
                trig_ids = tids.double_emu
            elif data_stream == "e":
                trig_ids = tids.single_e
            elif data_stream == "mu":
                trig_ids = tids.single_mu
            else:
                continue

        elif ch_key in {"cemuSS1tau"}:
            if self.dataset_inst.is_mc:
                trig_ids = (tids.single_e + tids.single_mu + tids.double_emu +
                            tids.cross_e_tau + tids.cross_mu_tau)
            elif data_stream == "muoneg":
                trig_ids = tids.double_emu
            elif data_stream == "e":
                trig_ids = tids.single_e + tids.cross_e_tau
            elif data_stream == "mu":
                trig_ids = tids.single_mu + tids.cross_mu_tau
            else:
                continue

        # 1l3th
        elif ch_key in {"ce3tau"}:
            if self.dataset_inst.is_mc:
                trig_ids = tids.single_e + tids.cross_e_tau + tids.cross_tau_tau_any
            elif data_stream == "tau":
                trig_ids = tids.cross_tau_tau_any
            elif data_stream == "e":
                trig_ids = tids.single_e + tids.cross_e_tau
            else:
                continue

        elif ch_key in {"cmu3tau"}:
            if self.dataset_inst.is_mc:
                trig_ids = tids.single_mu + tids.cross_mu_tau + tids.cross_tau_tau_any
            elif data_stream == "tau":
                trig_ids = tids.cross_tau_tau_any
            elif data_stream == "mu":
                trig_ids = tids.single_mu + tids.cross_mu_tau
            else:
                continue

        # 1l2th, 4tauh
        elif ch_key in {"c4tau"}:
            if self.dataset_inst.is_mc or data_stream == "tau":
                trig_ids = tids.cross_tau_tau_any
            else:
                continue

        elif ch_key in {"ce2tau"}:
            if self.dataset_inst.is_mc:
                trig_ids = tids.single_e + tids.cross_e_tau + tids.cross_tau_tau_any
            elif data_stream == "e":
                trig_ids = tids.single_e + tids.cross_e_tau
            elif data_stream == "tau":
                trig_ids = tids.cross_tau_tau_any
            else:
                continue

        elif ch_key in {"cmu2tau"}:
            if self.dataset_inst.is_mc:
                trig_ids = tids.single_mu + tids.cross_mu_tau + tids.cross_tau_tau_any
            elif data_stream == "mu":
                trig_ids = tids.single_mu + tids.cross_mu_tau
            elif data_stream == "tau":
                trig_ids = tids.cross_tau_tau_any
            else:
                continue

        # ttbar measurement region
        elif ch_key in {"cttbarMR"}:
            if self.dataset_inst.is_mc:
                trig_ids = (tids.single_e + tids.single_mu +
                            tids.double_e + tids.double_mu + tids.double_emu +
                            tids.cross_e_tau + tids.cross_mu_tau)
            elif data_stream == "muoneg":
                trig_ids = tids.double_emu
            elif data_stream == "e":
                trig_ids = tids.single_e + tids.double_e + tids.cross_e_tau
            elif data_stream == "mu":
                trig_ids = tids.single_mu + tids.double_mu + tids.cross_mu_tau
            else:
                continue

        # WZ measurement region
        elif ch_key in {"cwzMR"}:
            if self.dataset_inst.is_mc:
                trig_ids = (tids.single_e + tids.single_mu +
                            tids.double_e + tids.double_mu + tids.double_emu +
                            tids.triple_e + tids.triple_mu + tids.triple_eemu + tids.triple_emumu +
                            tids.cross_e_tau + tids.cross_mu_tau)
            elif data_stream == "muoneg":
                trig_ids = tids.double_emu + tids.triple_eemu + tids.triple_emumu
            elif data_stream == "e":
                trig_ids = tids.single_e + tids.double_e + tids.triple_e + tids.cross_e_tau
            elif data_stream == "mu":
                trig_ids = tids.single_mu + tids.double_mu + tids.triple_mu + tids.cross_mu_tau
            else:
                continue

        # Drell-Yan + jets measurement region
        elif ch_key in {"cdyMR"}:
            if self.dataset_inst.is_mc:
                trig_ids = tids.single_mu + tids.double_mu + tids.cross_mu_tau
            elif data_stream == "mu":
                trig_ids = tids.single_mu + tids.double_mu + tids.cross_mu_tau
            else:
                continue

        else:
            continue

        good_evt = ak.zeros_like(events.event, dtype=bool)

        for tid in trig_ids:
            e_mask = _trig_cache[(tid, "e")]
            e_ctrl = _trig_cache[(tid, "e_ctrl")]
            e_veto = _trig_cache[(tid, "e_veto")]
            e_match = _trig_cache[(tid, "e_match")]
            e_ctrl_bdt = _trig_cache[(tid, "e_ctrl_bdt")]
            e_veto_bdt = _trig_cache[(tid, "e_veto_bdt")]
            e_mask_bdt = _trig_cache[(tid, "e_mask_bdt")]

            mu_mask = _trig_cache[(tid, "mu")]
            mu_ctrl = _trig_cache[(tid, "mu_ctrl")]
            mu_veto = _trig_cache[(tid, "mu_veto")]
            mu_match = _trig_cache[(tid, "mu_match")]
            mu_ctrl_bdt = _trig_cache[(tid, "mu_ctrl_bdt")]
            mu_veto_bdt = _trig_cache[(tid, "mu_veto_bdt")]
            mu_mask_bdt = _trig_cache[(tid, "mu_mask_bdt")]

            tau_mask = _trig_cache[(tid, "tau_mask")]
            noid_tau_mask = _trig_cache[(tid, "noid_tau_mask")]
            tau_iso_mask = _trig_cache[(tid, "tau_iso_mask")]

            fired = _trig_cache[(tid, "fired")]

            # channel independent tau ID cuts vs e and mu (PNet: unified VLoose vs_e + Tight vs_mu)
            ch_tau_mask = (
                tau_mask &
                (events.Tau[get_tau_tagger("e")] >= wp_config.tau_vs_e.vloose) &
                (events.Tau[get_tau_tagger("mu")] >= wp_config.tau_vs_mu.tight)
            )

            ok = ak.ones_like(events.event, dtype=bool)

            if ch_key == "ceormu":

                e_base = (
                    (ak.sum(e_veto_bdt, axis=1) >= 1) &
                    (ak.sum(ch_tau_mask, axis=1) >= 0)
                )
                mu_base = (
                    (ak.sum(mu_veto_bdt, axis=1) >= 1) &
                    (ak.sum(ch_tau_mask, axis=1) >= 0)
                )

                base_ok = e_base | mu_base

                ok_bdt_eormu = ok_bdt_eormu | base_ok

                sel_electron_mask = sel_electron_mask | (e_base & e_ctrl_bdt)
                sel_looseelectron_mask = sel_looseelectron_mask | (e_base & e_veto_bdt)
                sel_tightelectron_mask = sel_tightelectron_mask | (e_base & e_mask_bdt)
                sel_muon_mask = sel_muon_mask | (mu_base & mu_ctrl_bdt)
                sel_loosemuon_mask = sel_loosemuon_mask | (mu_base & mu_veto_bdt)
                sel_tightmuon_mask = sel_tightmuon_mask | (mu_base & mu_mask_bdt)

                # leptons_os = ak.where(ok_bdt_eormu, False, leptons_os)
                tight_ok = (e_base & (ak.sum(e_mask_bdt,  axis=1) >= 1)) | (mu_base & (ak.sum(mu_mask_bdt,  axis=1) >= 1))  # noqa E501
                tight_sel_bdt = tight_sel_bdt | tight_ok
                # if tid in single_e_tids:
                #     trig_match_ok = base_ok & (ak.sum(e_match & e_ctrl_bdt, axis=1) >= 1)
                # elif tid in tids.single_mu:
                #     trig_match_ok = base_ok & (ak.sum(mu_match & mu_ctrl_bdt, axis=1) >= 1)

                trig_match_ok = base_ok
                trig_match_bdt = trig_match_bdt | trig_match_ok
                single_triggered = ak.where(trig_match_ok, True, single_triggered)

                ids = ak.where(trig_match_ok, np.float32(tid), np.float32(np.nan))
                matched_trigger_ids.append(ak.singletons(ak.nan_to_none(ids)))

                continue

            elif ch_key == "c3e":

                base_ok = (
                    (ak.sum(e_ctrl, axis=1) == 3) &
                    (ak.sum(e_veto, axis=1) == 3) &
                    (ak.sum(mu_veto, axis=1) == 0) &
                    (ak.sum(ch_tau_mask, axis=1) == 0)
                )

                if not disable_triggers:
                    base_ok = base_ok & fired

                ok = ak.where(base_ok, ok, False)

                sel_electron_mask = sel_electron_mask | (ok & e_ctrl)
                sel_looseelectron_mask = sel_looseelectron_mask | (ok & e_veto)
                sel_tightelectron_mask = sel_tightelectron_mask | (ok & e_mask)

                e_charge = events.Electron.charge[e_ctrl]
                chargeok = (np.abs(ak.sum(e_charge, axis=1)) == 1)
                leptons_os = ak.where(ok, chargeok, leptons_os)

                tight_ok = ok & (ak.sum(e_mask, axis=1) == 3)
                tight_sel = tight_sel | tight_ok

                trig_match_ok = base_ok & (ak.sum(e_match & e_ctrl, axis=1) >= 1)
                trig_match = trig_match | trig_match_ok

                single_triggered = ak.where(trig_match_ok, True, single_triggered)
                ids = ak.where(trig_match_ok, np.float32(tid), np.float32(np.nan))
                matched_trigger_ids.append(ak.singletons(ak.nan_to_none(ids)))

            elif ch_key == "c3mu":
                base_ok = (
                    (ak.sum(mu_ctrl, axis=1) == 3) &
                    (ak.sum(mu_veto, axis=1) == 3) &
                    (ak.sum(e_veto, axis=1) == 0) &
                    (ak.sum(ch_tau_mask, axis=1) == 0)
                )
                if not disable_triggers:
                    base_ok = base_ok & fired

                ok = ak.where(base_ok, ok, False)

                sel_muon_mask = sel_muon_mask | (ok & mu_ctrl)
                sel_loosemuon_mask = sel_loosemuon_mask | (ok & mu_veto)
                sel_tightmuon_mask = sel_tightmuon_mask | (ok & mu_mask)

                mu_charge = events.Muon.charge[mu_ctrl]
                chargeok = (np.abs(ak.sum(mu_charge, axis=1)) == 1)
                leptons_os = ak.where(ok, chargeok, leptons_os)

                tight_ok = ok & (ak.sum(mu_mask, axis=1) == 3)
                tight_sel = tight_sel | tight_ok

                trig_match_ok = base_ok & (ak.sum(mu_match & mu_ctrl, axis=1) >= 1)
                trig_match = trig_match | trig_match_ok

                single_triggered = ak.where(trig_match_ok, True, single_triggered)
                ids = ak.where(trig_match_ok, np.float32(tid), np.float32(np.nan))
                matched_trigger_ids.append(ak.singletons(ak.nan_to_none(ids)))

            elif ch_key == "c2emu":
                base_ok = (
                    (ak.sum(e_ctrl, axis=1) == 2) &
                    (ak.sum(e_veto, axis=1) == 2) &
                    (ak.sum(mu_ctrl, axis=1) == 1) &
                    (ak.sum(mu_veto, axis=1) == 1) &
                    (ak.sum(ch_tau_mask, axis=1) == 0)
                )
                if not disable_triggers:
                    base_ok = base_ok & fired

                ok = ak.where(base_ok, ok, False)

                sel_electron_mask = sel_electron_mask | (ok & e_ctrl)
                sel_looseelectron_mask = sel_looseelectron_mask | (ok & e_veto)
                sel_tightelectron_mask = sel_tightelectron_mask | (ok & e_mask)
                sel_muon_mask = sel_muon_mask | (ok & mu_ctrl)
                sel_loosemuon_mask = sel_loosemuon_mask | (ok & mu_veto)
                sel_tightmuon_mask = sel_tightmuon_mask | (ok & mu_mask)

                e_charge = events.Electron.charge[e_ctrl]
                mu_charge = events.Muon.charge[mu_ctrl]
                chargeok = (np.abs((ak.sum(e_charge, axis=1) + ak.sum(mu_charge, axis=1))) == 1)
                leptons_os = ak.where(ok, chargeok, leptons_os)

                tight_ok = ok & ((ak.sum(e_mask, axis=1) == 2) & (ak.sum(mu_mask, axis=1) == 1))
                tight_sel = tight_sel | tight_ok
                trig_match_ok = base_ok
                if tid in tids.single_e:
                    # emu_from_e — accept ONLY events with e_only (anti-overlap)
                    trig_match_ok = trig_match_ok & e_only & (ak.sum(e_match & e_ctrl, axis=1) >= 1)

                elif tid in tids.single_mu:
                    # emu_from_mu — allow both_families; the matching/logic below handles e-side
                    trig_match_ok = trig_match_ok & (ak.sum(mu_match & mu_ctrl, axis=1) >= 1)
                    # for events with both triggers firing:
                    if_e_fired = base_ok & e_trig_any & (ak.sum(e_match_any & e_ctrl, axis=1) >= 1)
                    trig_match_ok = ak.where(e_trig_any, trig_match_ok & if_e_fired, trig_match_ok)

                trig_match = trig_match | trig_match_ok

                single_triggered = ak.where(trig_match_ok, True, single_triggered)
                ids = ak.where(trig_match_ok, np.float32(tid), np.float32(np.nan))
                matched_trigger_ids.append(ak.singletons(ak.nan_to_none(ids)))

            elif ch_key == "ce2mu":
                base_ok = (
                    (ak.sum(e_ctrl, axis=1) == 1) &
                    (ak.sum(e_veto, axis=1) == 1) &
                    (ak.sum(mu_ctrl, axis=1) == 2) &
                    (ak.sum(mu_veto, axis=1) == 2) &
                    (ak.sum(ch_tau_mask, axis=1) == 0)
                )
                if not disable_triggers:
                    base_ok = base_ok & fired

                ok = ak.where(base_ok, ok, False)

                sel_electron_mask = sel_electron_mask | (ok & e_ctrl)
                sel_looseelectron_mask = sel_looseelectron_mask | (ok & e_veto)
                sel_tightelectron_mask = sel_tightelectron_mask | (ok & e_mask)
                sel_muon_mask = sel_muon_mask | (ok & mu_ctrl)
                sel_loosemuon_mask = sel_loosemuon_mask | (ok & mu_veto)
                sel_tightmuon_mask = sel_tightmuon_mask | (ok & mu_mask)

                e_charge = events.Electron.charge[e_ctrl]
                mu_charge = events.Muon.charge[mu_ctrl]
                chargeok = (np.abs((ak.sum(e_charge, axis=1) + ak.sum(mu_charge, axis=1))) == 1)
                leptons_os = ak.where(ok, chargeok, leptons_os)

                tight_ok = ok & ((ak.sum(e_mask, axis=1) == 1) & (ak.sum(mu_mask, axis=1) == 2))
                tight_sel = tight_sel | tight_ok
                trig_match_ok = base_ok
                if tid in tids.single_e:
                    # emu_from_e — accept ONLY events with e_only (anti-overlap)
                    trig_match_ok = trig_match_ok & e_only & (ak.sum(e_match & e_ctrl, axis=1) >= 1)

                elif tid in tids.single_mu:
                    # emu_from_mu — allow both_families; the matching/logic below handles e-side
                    trig_match_ok = trig_match_ok & (ak.sum(mu_match & mu_ctrl, axis=1) >= 1)
                    # for events with both triggers firing:
                    if_e_fired = base_ok & e_trig_any & (ak.sum(e_match_any & e_ctrl, axis=1) >= 1)
                    trig_match_ok = ak.where(e_trig_any, trig_match_ok & if_e_fired, trig_match_ok)

                trig_match = trig_match | trig_match_ok

                single_triggered = ak.where(trig_match_ok, True, single_triggered)
                ids = ak.where(trig_match_ok, np.float32(tid), np.float32(np.nan))
                matched_trigger_ids.append(ak.singletons(ak.nan_to_none(ids)))

            elif ch_key == "c4e":
                base_ok = (
                    (ak.sum(e_ctrl, axis=1) == 4) &
                    (ak.sum(e_veto, axis=1) == 4) &
                    (ak.sum(mu_veto, axis=1) == 0) &
                    (ak.sum(ch_tau_mask, axis=1) == 0)
                )
                if not disable_triggers:
                    base_ok = base_ok & fired

                ok = ak.where(base_ok, ok, False)

                sel_electron_mask = sel_electron_mask | (ok & e_ctrl)
                sel_looseelectron_mask = sel_looseelectron_mask | (ok & e_veto)
                sel_tightelectron_mask = sel_tightelectron_mask | (ok & e_mask)

                e_charge = events.Electron.charge[e_ctrl]
                chargeok = (np.abs(ak.sum(e_charge, axis=1)) == 0)
                leptons_os = ak.where(ok, chargeok, leptons_os)

                tight_ok = ok & (ak.sum(e_mask, axis=1) == 4)
                tight_sel = tight_sel | tight_ok

                trig_match_ok = base_ok & (ak.sum(e_match & e_ctrl, axis=1) >= 1)
                trig_match = trig_match | trig_match_ok

                single_triggered = ak.where(trig_match_ok, True, single_triggered)
                ids = ak.where(trig_match_ok, np.float32(tid), np.float32(np.nan))
                matched_trigger_ids.append(ak.singletons(ak.nan_to_none(ids)))

            elif ch_key == "c4mu":
                base_ok = (
                    (ak.sum(mu_ctrl, axis=1) == 4) &
                    (ak.sum(mu_veto, axis=1) == 4) &
                    (ak.sum(e_veto, axis=1) == 0) &
                    (ak.sum(ch_tau_mask, axis=1) == 0)
                )
                if not disable_triggers:
                    base_ok = base_ok & fired

                ok = ak.where(base_ok, ok, False)

                sel_muon_mask = sel_muon_mask | (ok & mu_ctrl)
                sel_loosemuon_mask = sel_loosemuon_mask | (ok & mu_veto)
                sel_tightmuon_mask = sel_tightmuon_mask | (ok & mu_mask)

                mu_charge = events.Muon.charge[mu_ctrl]
                chargeok = (np.abs(ak.sum(mu_charge, axis=1)) == 0)
                leptons_os = ak.where(ok, chargeok, leptons_os)

                tight_ok = ok & (ak.sum(mu_mask, axis=1) == 4)
                tight_sel = tight_sel | tight_ok

                trig_match_ok = base_ok & (ak.sum(mu_match & mu_ctrl, axis=1) >= 1)
                trig_match = trig_match | trig_match_ok

                single_triggered = ak.where(trig_match_ok, True, single_triggered)
                ids = ak.where(trig_match_ok, np.float32(tid), np.float32(np.nan))
                matched_trigger_ids.append(ak.singletons(ak.nan_to_none(ids)))

            elif ch_key == "c3emu":
                base_ok = (
                    (ak.sum(e_ctrl, axis=1) == 3) &
                    (ak.sum(e_veto, axis=1) == 3) &
                    (ak.sum(mu_ctrl, axis=1) == 1) &
                    (ak.sum(mu_veto, axis=1) == 1) &
                    (ak.sum(ch_tau_mask, axis=1) == 0)
                )
                if not disable_triggers:
                    base_ok = base_ok & fired

                ok = ak.where(base_ok, ok, False)

                sel_electron_mask = sel_electron_mask | (ok & e_ctrl)
                sel_looseelectron_mask = sel_looseelectron_mask | (ok & e_veto)
                sel_tightelectron_mask = sel_tightelectron_mask | (ok & e_mask)
                sel_muon_mask = sel_muon_mask | (ok & mu_ctrl)
                sel_loosemuon_mask = sel_loosemuon_mask | (ok & mu_veto)
                sel_tightmuon_mask = sel_tightmuon_mask | (ok & mu_mask)

                e_charge = events.Electron.charge[e_ctrl]
                mu_charge = events.Muon.charge[mu_ctrl]
                chargeok = (np.abs((ak.sum(e_charge, axis=1) + ak.sum(mu_charge, axis=1))) == 0)
                leptons_os = ak.where(ok, chargeok, leptons_os)

                tight_ok = ok & ((ak.sum(e_mask, axis=1) == 3) & (ak.sum(mu_mask, axis=1) == 1))
                tight_sel = tight_sel | tight_ok

                trig_match_ok = base_ok
                if tid in tids.single_e:
                    # emu_from_e — accept ONLY events with e_only (anti-overlap)
                    trig_match_ok = trig_match_ok & e_only & (ak.sum(e_match & e_ctrl, axis=1) >= 1)

                elif tid in tids.single_mu:
                    # emu_from_mu — allow both_families; the matching/logic below handles e-side
                    trig_match_ok = trig_match_ok & (ak.sum(mu_match & mu_ctrl, axis=1) >= 1)
                    # for events with both triggers firing:
                    if_e_fired = base_ok & e_trig_any & (ak.sum(e_match_any & e_ctrl, axis=1) >= 1)
                    trig_match_ok = ak.where(e_trig_any, trig_match_ok & if_e_fired, trig_match_ok)

                trig_match = trig_match | trig_match_ok

                single_triggered = ak.where(trig_match_ok, True, single_triggered)
                ids = ak.where(trig_match_ok, np.float32(tid), np.float32(np.nan))
                matched_trigger_ids.append(ak.singletons(ak.nan_to_none(ids)))

            elif ch_key == "c2e2mu":
                base_ok = (
                    (ak.sum(e_ctrl, axis=1) == 2) &
                    (ak.sum(e_veto, axis=1) == 2) &
                    (ak.sum(mu_ctrl, axis=1) == 2) &
                    (ak.sum(mu_veto, axis=1) == 2) &
                    (ak.sum(ch_tau_mask, axis=1) == 0)
                )
                if not disable_triggers:
                    base_ok = base_ok & fired

                ok = ak.where(base_ok, ok, False)

                sel_electron_mask = sel_electron_mask | (ok & e_ctrl)
                sel_looseelectron_mask = sel_looseelectron_mask | (ok & e_veto)
                sel_tightelectron_mask = sel_tightelectron_mask | (ok & e_mask)
                sel_muon_mask = sel_muon_mask | (ok & mu_ctrl)
                sel_loosemuon_mask = sel_loosemuon_mask | (ok & mu_veto)
                sel_tightmuon_mask = sel_tightmuon_mask | (ok & mu_mask)

                e_charge = events.Electron.charge[e_ctrl]
                mu_charge = events.Muon.charge[mu_ctrl]
                chargeok = (np.abs((ak.sum(e_charge, axis=1) + ak.sum(mu_charge, axis=1))) == 0)
                leptons_os = ak.where(ok, chargeok, leptons_os)

                tight_ok = ok & ((ak.sum(e_mask, axis=1) == 2) & (ak.sum(mu_mask, axis=1) == 2))
                tight_sel = tight_sel | tight_ok

                trig_match_ok = base_ok
                if tid in tids.single_e:
                    # emu_from_e — accept ONLY events with e_only (anti-overlap)
                    trig_match_ok = trig_match_ok & e_only & (ak.sum(e_match & e_ctrl, axis=1) >= 1)

                elif tid in tids.single_mu:
                    # emu_from_mu — allow both_families; the matching/logic below handles e-side
                    trig_match_ok = trig_match_ok & (ak.sum(mu_match & mu_ctrl, axis=1) >= 1)
                    # for events with both triggers firing:
                    if_e_fired = base_ok & e_trig_any & (ak.sum(e_match_any & e_ctrl, axis=1) >= 1)
                    trig_match_ok = ak.where(e_trig_any, trig_match_ok & if_e_fired, trig_match_ok)

                trig_match = trig_match | trig_match_ok

                single_triggered = ak.where(trig_match_ok, True, single_triggered)
                ids = ak.where(trig_match_ok, np.float32(tid), np.float32(np.nan))
                matched_trigger_ids.append(ak.singletons(ak.nan_to_none(ids)))

            elif ch_key == "ce3mu":
                base_ok = (
                    (ak.sum(e_ctrl, axis=1) == 1) &
                    (ak.sum(e_veto, axis=1) == 1) &
                    (ak.sum(mu_ctrl, axis=1) == 3) &
                    (ak.sum(mu_veto, axis=1) == 3) &
                    (ak.sum(ch_tau_mask, axis=1) == 0)
                )
                if not disable_triggers:
                    base_ok = base_ok & fired

                ok = ak.where(base_ok, ok, False)

                sel_electron_mask = sel_electron_mask | (ok & e_ctrl)
                sel_looseelectron_mask = sel_looseelectron_mask | (ok & e_veto)
                sel_tightelectron_mask = sel_tightelectron_mask | (ok & e_mask)
                sel_muon_mask = sel_muon_mask | (ok & mu_ctrl)
                sel_loosemuon_mask = sel_loosemuon_mask | (ok & mu_veto)
                sel_tightmuon_mask = sel_tightmuon_mask | (ok & mu_mask)

                e_charge = events.Electron.charge[e_ctrl]
                mu_charge = events.Muon.charge[mu_ctrl]
                chargeok = (np.abs((ak.sum(e_charge, axis=1) + ak.sum(mu_charge, axis=1))) == 0)
                leptons_os = ak.where(ok, chargeok, leptons_os)

                tight_ok = ok & ((ak.sum(e_mask, axis=1) == 1) & (ak.sum(mu_mask, axis=1) == 3))
                tight_sel = tight_sel | tight_ok

                trig_match_ok = base_ok
                if tid in tids.single_e:
                    # emu_from_e — accept ONLY events with e_only (anti-overlap)
                    trig_match_ok = trig_match_ok & e_only & (ak.sum(e_match & e_ctrl, axis=1) >= 1)

                elif tid in tids.single_mu:
                    # emu_from_mu — allow both_families; the matching/logic below handles e-side
                    trig_match_ok = trig_match_ok & (ak.sum(mu_match & mu_ctrl, axis=1) >= 1)
                    # for events with both triggers firing:
                    if_e_fired = base_ok & e_trig_any & (ak.sum(e_match_any & e_ctrl, axis=1) >= 1)
                    trig_match_ok = ak.where(e_trig_any, trig_match_ok & if_e_fired, trig_match_ok)

                trig_match = trig_match | trig_match_ok

                single_triggered = ak.where(trig_match_ok, True, single_triggered)
                ids = ak.where(trig_match_ok, np.float32(tid), np.float32(np.nan))
                matched_trigger_ids.append(ak.singletons(ak.nan_to_none(ids)))

            elif ch_key == "c3etau":
                base_ok = (
                    (ak.sum(e_ctrl, axis=1) == 3) &
                    (ak.sum(e_veto, axis=1) == 3) &
                    (ak.sum(mu_veto, axis=1) == 0) &
                    (ak.sum(ch_tau_mask, axis=1) == 1)
                )
                if not disable_triggers:
                    base_ok = base_ok & fired

                ok = ak.where(base_ok, ok, False)

                sel_electron_mask = sel_electron_mask | (ok & e_ctrl)
                sel_looseelectron_mask = sel_looseelectron_mask | (ok & e_veto)
                sel_tightelectron_mask = sel_tightelectron_mask | (ok & e_mask)
                sel_tau_mask = sel_tau_mask | (ok & ch_tau_mask)
                sel_isotau_mask = sel_isotau_mask | (ok & (ch_tau_mask & tau_iso_mask))

                e_charge = events.Electron.charge[e_ctrl]
                tau_charge = events.Tau.charge[ch_tau_mask]
                chargeok = ((np.abs((ak.sum(tau_charge, axis=1)) +
                    (ak.sum(e_charge, axis=1))) == 0) & (np.abs(ak.sum(e_charge, axis=1)) == 1))
                leptons_os = ak.where(ok, chargeok, leptons_os)

                ch_tau_mask = ch_tau_mask & (events.Tau[get_tau_tagger("e")] >= wp_config.tau_vs_e.vloose)
                tight_ok = ok & ((ak.sum((ch_tau_mask & tau_iso_mask), axis=1) == 1) & (ak.sum(e_mask, axis=1) == 3))
                tight_sel = tight_sel | tight_ok

                trig_match_ok = base_ok
                if tid in tids.single_e:
                    trig_match_ok = trig_match_ok & e_only_emutau & (ak.sum(e_match & e_ctrl, axis=1) >= 1)
                elif tid in tids.cross_e_tau:
                    trig_match_ok = trig_match_ok & (
                        (ak.sum(tau_match & ch_tau_mask, axis=1) >= 1) &
                        (ak.sum(e_match & e_ctrl, axis=1) >= 1)
                    )

                trig_match = trig_match | trig_match_ok

                single_triggered = ak.where(trig_match_ok, True, single_triggered)
                ids = ak.where(trig_match_ok, np.float32(tid), np.float32(np.nan))
                matched_trigger_ids.append(ak.singletons(ak.nan_to_none(ids)))

            elif ch_key == "c2e2tau":
                base_ok = (
                    (ak.sum(e_ctrl, axis=1) == 2) &
                    (ak.sum(e_veto, axis=1) == 2) &
                    (ak.sum(mu_veto, axis=1) == 0) &
                    (ak.sum(ch_tau_mask, axis=1) == 2)
                )
                if not disable_triggers:
                    base_ok = base_ok & fired

                ok = ak.where(base_ok, ok, False)

                sel_electron_mask = sel_electron_mask | (ok & e_ctrl)
                sel_looseelectron_mask = sel_looseelectron_mask | (ok & e_veto)
                sel_tightelectron_mask = sel_tightelectron_mask | (ok & e_mask)
                sel_tau_mask = sel_tau_mask | (ok & ch_tau_mask)
                sel_isotau_mask = sel_isotau_mask | (ok & (ch_tau_mask & tau_iso_mask))

                e_charge = events.Electron.charge[e_ctrl]
                tau_charge = events.Tau.charge[ch_tau_mask]
                chargeok = ((np.abs((ak.sum(tau_charge, axis=1)) +
                    (ak.sum(e_charge, axis=1))) == 0))
                leptons_os = ak.where(ok, chargeok, leptons_os)

                tight_ok = ok & ((ak.sum((ch_tau_mask & tau_iso_mask), axis=1) == 2) & (ak.sum(e_mask, axis=1) == 2))
                tight_sel = tight_sel | tight_ok

                trig_match_ok = base_ok
                if tid in tids.single_e:
                    trig_match_ok = trig_match_ok & e_only_emutau & (ak.sum(e_match & e_ctrl, axis=1) >= 1)
                elif tid in tids.cross_e_tau:
                    trig_match_ok = trig_match_ok & (
                        (ak.sum(tau_match & ch_tau_mask, axis=1) >= 1) &
                        (ak.sum(e_match & e_ctrl, axis=1) >= 1)
                    )
                elif tid in tids.cross_tau_tau_any:
                    trig_match_ok = trig_match_ok & (ak.sum(tau_match & ch_tau_mask, axis=1) >= 1)

                trig_match = trig_match | trig_match_ok

                single_triggered = ak.where(trig_match_ok, True, single_triggered)
                ids = ak.where(trig_match_ok, np.float32(tid), np.float32(np.nan))
                matched_trigger_ids.append(ak.singletons(ak.nan_to_none(ids)))

            elif ch_key == "ce3tau":
                base_ok = (
                    (ak.sum(e_ctrl, axis=1) == 1) &
                    (ak.sum(e_veto, axis=1) == 1) &
                    (ak.sum(mu_veto, axis=1) == 0) &
                    (ak.sum(ch_tau_mask, axis=1) == 3)
                )
                if not disable_triggers:
                    base_ok = base_ok & fired

                ok = ak.where(base_ok, ok, False)

                sel_electron_mask = sel_electron_mask | (ok & e_ctrl)
                sel_looseelectron_mask = sel_looseelectron_mask | (ok & e_veto)
                sel_tightelectron_mask = sel_tightelectron_mask | (ok & e_mask)
                sel_tau_mask = sel_tau_mask | (ok & ch_tau_mask)
                sel_isotau_mask = sel_isotau_mask | (ok & (ch_tau_mask & tau_iso_mask))

                e_charge = events.Electron.charge[e_ctrl]
                tau_charge = events.Tau.charge[ch_tau_mask]
                chargeok = (np.abs((ak.sum(tau_charge, axis=1)) +
                    (ak.sum(e_charge, axis=1))) == 0)
                leptons_os = ak.where(ok, chargeok, leptons_os)

                ch_tau_mask = ch_tau_mask & (events.Tau[get_tau_tagger("e")] >= wp_config.tau_vs_e.vloose)
                tight_ok = ok & ((ak.sum((ch_tau_mask & tau_iso_mask), axis=1) == 3) & (ak.sum(e_mask, axis=1) == 1))
                tight_sel = tight_sel | tight_ok

                trig_match_ok = base_ok
                if tid in tids.single_e:
                    trig_match_ok = trig_match_ok & e_only_emutau & (ak.sum(e_match & e_ctrl, axis=1) >= 1)
                elif tid in tids.cross_e_tau:
                    trig_match_ok = trig_match_ok & (
                        (ak.sum(tau_match & ch_tau_mask, axis=1) >= 1) &
                        (ak.sum(e_match & e_ctrl, axis=1) >= 1)
                    )

                trig_match = trig_match | trig_match_ok

                single_triggered = ak.where(trig_match_ok, True, single_triggered)
                ids = ak.where(trig_match_ok, np.float32(tid), np.float32(np.nan))
                matched_trigger_ids.append(ak.singletons(ak.nan_to_none(ids)))

            elif ch_key == "c3mutau":
                base_ok = (
                    (ak.sum(mu_ctrl, axis=1) == 3) &
                    (ak.sum(mu_veto, axis=1) == 3) &
                    (ak.sum(e_veto, axis=1) == 0) &
                    (ak.sum(ch_tau_mask, axis=1) == 1)
                )
                if not disable_triggers:
                    base_ok = base_ok & fired

                ok = ak.where(base_ok, ok, False)

                sel_muon_mask = sel_muon_mask | (ok & mu_ctrl)
                sel_loosemuon_mask = sel_loosemuon_mask | (ok & mu_veto)
                sel_tightmuon_mask = sel_tightmuon_mask | (ok & mu_mask)
                sel_tau_mask = sel_tau_mask | (ok & ch_tau_mask)
                sel_isotau_mask = sel_isotau_mask | (ok & (ch_tau_mask & tau_iso_mask))

                mu_charge = events.Muon.charge[mu_ctrl]
                tau_charge = events.Tau.charge[ch_tau_mask]
                chargeok = ((np.abs((ak.sum(tau_charge, axis=1)) +
                    (ak.sum(mu_charge, axis=1))) == 0) & (np.abs((ak.sum(mu_charge, axis=1))) == 1))
                leptons_os = ak.where(ok, chargeok, leptons_os)

                ch_tau_mask = ch_tau_mask & (events.Tau[get_tau_tagger("e")] >= wp_config.tau_vs_e.vloose)
                tight_ok = ok & ((ak.sum((ch_tau_mask & tau_iso_mask), axis=1) == 1) &
                    (ak.sum(mu_mask, axis=1) == 3))
                tight_sel = tight_sel | tight_ok

                trig_match_ok = base_ok
                if tid in tids.single_mu:
                    trig_match_ok = trig_match_ok & mu_only_emutau & (ak.sum(mu_match & mu_ctrl, axis=1) >= 1)
                elif tid in tids.cross_mu_tau:
                    trig_match_ok = trig_match_ok & (
                        (ak.sum(tau_match & ch_tau_mask, axis=1) >= 1) &
                        (ak.sum(mu_match & mu_ctrl, axis=1) >= 1)
                    )

                trig_match = trig_match | trig_match_ok

                single_triggered = ak.where(trig_match_ok, True, single_triggered)
                ids = ak.where(trig_match_ok, np.float32(tid), np.float32(np.nan))
                matched_trigger_ids.append(ak.singletons(ak.nan_to_none(ids)))

            elif ch_key == "c2mu2tau":
                base_ok = (
                    (ak.sum(mu_ctrl, axis=1) == 2) &
                    (ak.sum(mu_veto, axis=1) == 2) &
                    (ak.sum(e_veto, axis=1) == 0) &
                    (ak.sum(ch_tau_mask, axis=1) == 2)
                )
                if not disable_triggers:
                    base_ok = base_ok & fired

                ok = ak.where(base_ok, ok, False)

                sel_muon_mask = sel_muon_mask | (ok & mu_ctrl)
                sel_loosemuon_mask = sel_loosemuon_mask | (ok & mu_veto)
                sel_tightmuon_mask = sel_tightmuon_mask | (ok & mu_mask)
                sel_tau_mask = sel_tau_mask | (ok & ch_tau_mask)
                sel_isotau_mask = sel_isotau_mask | (ok & (ch_tau_mask & tau_iso_mask))

                mu_charge = events.Muon.charge[mu_ctrl]
                tau_charge = events.Tau.charge[ch_tau_mask]
                chargeok = ((np.abs((ak.sum(tau_charge, axis=1)) +
                    (ak.sum(mu_charge, axis=1))) == 0))
                leptons_os = ak.where(ok, chargeok, leptons_os)

                tight_ok = ok & ((ak.sum((ch_tau_mask & tau_iso_mask), axis=1) == 2) & (ak.sum(mu_mask, axis=1) == 2))
                tight_sel = tight_sel | tight_ok

                trig_match_ok = base_ok
                if tid in tids.single_mu:
                    trig_match_ok = trig_match_ok & mu_only_emutau & (ak.sum(mu_match & mu_ctrl, axis=1) >= 1)
                elif tid in tids.cross_mu_tau:
                    trig_match_ok = trig_match_ok & (
                        (ak.sum(tau_match & ch_tau_mask, axis=1) >= 1) &
                        (ak.sum(mu_match & mu_ctrl, axis=1) >= 1)
                    )
                elif tid in tids.cross_tau_tau_any:
                    trig_match_ok = trig_match_ok & (ak.sum(tau_match & ch_tau_mask, axis=1) >= 1)

                trig_match = trig_match | trig_match_ok

                single_triggered = ak.where(trig_match_ok, True, single_triggered)
                ids = ak.where(trig_match_ok, np.float32(tid), np.float32(np.nan))
                matched_trigger_ids.append(ak.singletons(ak.nan_to_none(ids)))

            elif ch_key == "cmu3tau":
                base_ok = (
                    (ak.sum(mu_ctrl, axis=1) == 1) &
                    (ak.sum(mu_veto, axis=1) == 1) &
                    (ak.sum(e_veto, axis=1) == 0) &
                    (ak.sum(ch_tau_mask, axis=1) == 3)
                )
                if not disable_triggers:
                    base_ok = base_ok & fired

                ok = ak.where(base_ok, ok, False)

                sel_muon_mask = sel_muon_mask | (ok & mu_ctrl)
                sel_loosemuon_mask = sel_loosemuon_mask | (ok & mu_veto)
                sel_tightmuon_mask = sel_tightmuon_mask | (ok & mu_mask)
                sel_tau_mask = sel_tau_mask | (ok & ch_tau_mask)
                sel_isotau_mask = sel_isotau_mask | (ok & (ch_tau_mask & tau_iso_mask))

                mu_charge = events.Muon.charge[mu_ctrl]
                tau_charge = events.Tau.charge[ch_tau_mask]
                chargeok = (np.abs((ak.sum(tau_charge, axis=1)) +
                    (ak.sum(mu_charge, axis=1))) == 0)
                leptons_os = ak.where(ok, chargeok, leptons_os)

                ch_tau_mask = ch_tau_mask & (events.Tau[get_tau_tagger("e")] >= wp_config.tau_vs_e.vloose)
                tight_ok = ok & ((ak.sum((ch_tau_mask & tau_iso_mask), axis=1) == 3) & (ak.sum(mu_mask, axis=1) == 1))
                tight_sel = tight_sel | tight_ok

                trig_match_ok = base_ok
                if tid in tids.single_mu:
                    trig_match_ok = trig_match_ok & mu_only_emutau & (ak.sum(mu_match & mu_ctrl, axis=1) >= 1)
                elif tid in tids.cross_mu_tau:
                    trig_match_ok = trig_match_ok & (
                        (ak.sum(tau_match & ch_tau_mask, axis=1) >= 1) &
                        (ak.sum(mu_match & mu_ctrl, axis=1) >= 1)
                    )

                trig_match = trig_match | trig_match_ok

                single_triggered = ak.where(trig_match_ok, True, single_triggered)
                ids = ak.where(trig_match_ok, np.float32(tid), np.float32(np.nan))
                matched_trigger_ids.append(ak.singletons(ak.nan_to_none(ids)))

            elif ch_key == "c2emutau":
                base_ok = (
                    (ak.sum(e_ctrl, axis=1) == 2) &
                    (ak.sum(e_veto, axis=1) == 2) &
                    (ak.sum(mu_ctrl, axis=1) == 1) &
                    (ak.sum(mu_veto, axis=1) == 1) &
                    (ak.sum(ch_tau_mask, axis=1) == 1)
                )
                if not disable_triggers:
                    base_ok = base_ok & fired

                ok = ak.where(base_ok, ok, False)

                sel_electron_mask = sel_electron_mask | (ok & e_ctrl)
                sel_looseelectron_mask = sel_looseelectron_mask | (ok & e_veto)
                sel_tightelectron_mask = sel_tightelectron_mask | (ok & e_mask)
                sel_muon_mask = sel_muon_mask | (ok & mu_ctrl)
                sel_loosemuon_mask = sel_loosemuon_mask | (ok & mu_veto)
                sel_tightmuon_mask = sel_tightmuon_mask | (ok & mu_mask)
                sel_tau_mask = sel_tau_mask | (ok & ch_tau_mask)
                sel_isotau_mask = sel_isotau_mask | (ok & (ch_tau_mask & tau_iso_mask))

                e_charge = events.Electron.charge[e_ctrl]
                mu_charge = events.Muon.charge[mu_ctrl]
                tau_charge = events.Tau.charge[ch_tau_mask]
                chargeok = ((np.abs((ak.sum(tau_charge, axis=1)) +
                    (ak.sum(e_charge, axis=1)) + (ak.sum(mu_charge, axis=1))) == 0) &
                    (np.abs((ak.sum(e_charge, axis=1)) + (ak.sum(mu_charge, axis=1))) == 1))
                leptons_os = ak.where(ok, chargeok, leptons_os)

                ch_tau_mask = ch_tau_mask & (events.Tau[get_tau_tagger("e")] >= wp_config.tau_vs_e.vloose)
                tight_ok = ok & ((ak.sum((ch_tau_mask & tau_iso_mask), axis=1) == 1) &
                    (ak.sum(e_mask, axis=1) == 2) & (ak.sum(mu_mask, axis=1) == 1))
                tight_sel = tight_sel | tight_ok

                trig_match_ok = base_ok
                if tid in tids.single_e:
                    trig_match_ok = trig_match_ok & (ak.sum(e_match & e_ctrl, axis=1) >= 1)
                    if_mu_fired = base_ok & mu_trig_any & (ak.sum(mu_match_any & mu_ctrl, axis=1) >= 1)
                    trig_match_ok = ak.where(mu_trig_any, trig_match_ok & if_mu_fired, trig_match_ok)
                elif tid in tids.single_mu:
                    trig_match_ok = trig_match_ok & (ak.sum(mu_match & mu_ctrl, axis=1) >= 1)
                    if_e_fired = base_ok & e_trig_any & (ak.sum(e_match_any & e_ctrl, axis=1) >= 1)
                    trig_match_ok = ak.where(e_trig_any, trig_match_ok & if_e_fired, trig_match_ok)
                elif tid in tids.cross_e_tau:
                    trig_match_ok = trig_match_ok & (
                        (ak.sum(tau_match & ch_tau_mask, axis=1) >= 1) &
                        (ak.sum(e_match & e_ctrl, axis=1) >= 1)
                    )
                    if_mu_fired = base_ok & mu_trig_any & (ak.sum(mu_match_any & mu_ctrl, axis=1) >= 1)
                    trig_match_ok = ak.where(mu_trig_any, trig_match_ok & if_mu_fired, trig_match_ok)
                elif tid in tids.cross_mu_tau:
                    trig_match_ok = trig_match_ok & (
                        (ak.sum(tau_match & ch_tau_mask, axis=1) >= 1) &
                        (ak.sum(mu_match & mu_ctrl, axis=1) >= 1)
                    )
                    if_e_fired = base_ok & e_trig_any & (ak.sum(e_match_any & e_ctrl, axis=1) >= 1)
                    trig_match_ok = ak.where(e_trig_any, trig_match_ok & if_e_fired, trig_match_ok)

                trig_match = trig_match | trig_match_ok

                single_triggered = ak.where(trig_match_ok, True, single_triggered)
                ids = ak.where(trig_match_ok, np.float32(tid), np.float32(np.nan))
                matched_trigger_ids.append(ak.singletons(ak.nan_to_none(ids)))

            elif ch_key == "ce2mutau":
                base_ok = (
                    (ak.sum(e_ctrl, axis=1) == 1) &
                    (ak.sum(e_veto, axis=1) == 1) &
                    (ak.sum(mu_ctrl, axis=1) == 2) &
                    (ak.sum(mu_veto, axis=1) == 2) &
                    (ak.sum(ch_tau_mask, axis=1) == 1)
                )
                if not disable_triggers:
                    base_ok = base_ok & fired

                ok = ak.where(base_ok, ok, False)

                sel_electron_mask = sel_electron_mask | (ok & e_ctrl)
                sel_looseelectron_mask = sel_looseelectron_mask | (ok & e_veto)
                sel_tightelectron_mask = sel_tightelectron_mask | (ok & e_mask)
                sel_muon_mask = sel_muon_mask | (ok & mu_ctrl)
                sel_loosemuon_mask = sel_loosemuon_mask | (ok & mu_veto)
                sel_tightmuon_mask = sel_tightmuon_mask | (ok & mu_mask)
                sel_tau_mask = sel_tau_mask | (ok & ch_tau_mask)
                sel_isotau_mask = sel_isotau_mask | (ok & (ch_tau_mask & tau_iso_mask))

                e_charge = events.Electron.charge[e_ctrl]
                mu_charge = events.Muon.charge[mu_ctrl]
                tau_charge = events.Tau.charge[ch_tau_mask]
                chargeok = ((np.abs((ak.sum(tau_charge, axis=1)) +
                    (ak.sum(e_charge, axis=1)) + (ak.sum(mu_charge, axis=1))) == 0) &
                    (np.abs((ak.sum(e_charge, axis=1)) + (ak.sum(mu_charge, axis=1))) == 1))
                leptons_os = ak.where(ok, chargeok, leptons_os)

                ch_tau_mask = ch_tau_mask & (events.Tau[get_tau_tagger("e")] >= wp_config.tau_vs_e.vloose)
                tight_ok = ok & ((ak.sum((ch_tau_mask & tau_iso_mask), axis=1) == 1) &
                    (ak.sum(e_mask, axis=1) == 1) & (ak.sum(mu_mask, axis=1) == 2))
                tight_sel = tight_sel | tight_ok

                trig_match_ok = base_ok
                if tid in tids.single_e:
                    trig_match_ok = trig_match_ok & (ak.sum(e_match & e_ctrl, axis=1) >= 1)
                    if_mu_fired = base_ok & mu_trig_any & (ak.sum(mu_match_any & mu_ctrl, axis=1) >= 1)
                    trig_match_ok = ak.where(mu_trig_any, trig_match_ok & if_mu_fired, trig_match_ok)
                elif tid in tids.single_mu:
                    trig_match_ok = trig_match_ok & (ak.sum(mu_match & mu_ctrl, axis=1) >= 1)
                    if_e_fired = base_ok & e_trig_any & (ak.sum(e_match_any & e_ctrl, axis=1) >= 1)
                    trig_match_ok = ak.where(e_trig_any, trig_match_ok & if_e_fired, trig_match_ok)
                elif tid in tids.cross_e_tau:
                    trig_match_ok = trig_match_ok & (
                        (ak.sum(tau_match & ch_tau_mask, axis=1) >= 1) &
                        (ak.sum(e_match & e_ctrl, axis=1) >= 1)
                    )
                    if_mu_fired = base_ok & mu_trig_any & (ak.sum(mu_match_any & mu_ctrl, axis=1) >= 1)
                    trig_match_ok = ak.where(mu_trig_any, trig_match_ok & if_mu_fired, trig_match_ok)
                elif tid in tids.cross_mu_tau:
                    trig_match_ok = trig_match_ok & (
                        (ak.sum(tau_match & ch_tau_mask, axis=1) >= 1) &
                        (ak.sum(mu_match & mu_ctrl, axis=1) >= 1)
                    )
                    if_e_fired = base_ok & e_trig_any & (ak.sum(e_match_any & e_ctrl, axis=1) >= 1)
                    trig_match_ok = ak.where(e_trig_any, trig_match_ok & if_e_fired, trig_match_ok)

                trig_match = trig_match | trig_match_ok

                single_triggered = ak.where(trig_match_ok, True, single_triggered)
                ids = ak.where(trig_match_ok, np.float32(tid), np.float32(np.nan))
                matched_trigger_ids.append(ak.singletons(ak.nan_to_none(ids)))

            elif ch_key == "cemu2tau":
                base_ok = (
                    (ak.sum(e_ctrl, axis=1) == 1) &
                    (ak.sum(e_veto, axis=1) == 1) &
                    (ak.sum(mu_ctrl, axis=1) == 1) &
                    (ak.sum(mu_veto, axis=1) == 1) &
                    (ak.sum(ch_tau_mask, axis=1) == 2)
                )
                if not disable_triggers:
                    base_ok = base_ok & fired

                ok = ak.where(base_ok, ok, False)

                sel_electron_mask = sel_electron_mask | (ok & e_ctrl)
                sel_looseelectron_mask = sel_looseelectron_mask | (ok & e_veto)
                sel_tightelectron_mask = sel_tightelectron_mask | (ok & e_mask)
                sel_muon_mask = sel_muon_mask | (ok & mu_ctrl)
                sel_loosemuon_mask = sel_loosemuon_mask | (ok & mu_veto)
                sel_tightmuon_mask = sel_tightmuon_mask | (ok & mu_mask)
                sel_tau_mask = sel_tau_mask | (ok & ch_tau_mask)
                sel_isotau_mask = sel_isotau_mask | (ok & (ch_tau_mask & tau_iso_mask))

                e_charge = events.Electron.charge[e_ctrl]
                mu_charge = events.Muon.charge[mu_ctrl]
                tau_charge = events.Tau.charge[ch_tau_mask]
                chargeok = ((np.abs((ak.sum(tau_charge, axis=1)) +
                    (ak.sum(e_charge, axis=1)) + (ak.sum(mu_charge, axis=1))) == 0))
                leptons_os = ak.where(ok, chargeok, leptons_os)

                tight_ok = ok & ((ak.sum((ch_tau_mask & tau_iso_mask), axis=1) == 2) &
                    (ak.sum(e_mask, axis=1) == 1) & (ak.sum(mu_mask, axis=1) == 1))
                tight_sel = tight_sel | tight_ok

                trig_match_ok = base_ok
                if tid in tids.single_e:
                    trig_match_ok = trig_match_ok & (ak.sum(e_match & e_ctrl, axis=1) >= 1)
                    if_mu_fired = base_ok & mu_trig_any & (ak.sum(mu_match_any & mu_ctrl, axis=1) >= 1)
                    trig_match_ok = ak.where(mu_trig_any, trig_match_ok & if_mu_fired, trig_match_ok)
                elif tid in tids.single_mu:
                    trig_match_ok = trig_match_ok & (ak.sum(mu_match & mu_ctrl, axis=1) >= 1)
                    if_e_fired = base_ok & e_trig_any & (ak.sum(e_match_any & e_ctrl, axis=1) >= 1)
                    trig_match_ok = ak.where(e_trig_any, trig_match_ok & if_e_fired, trig_match_ok)
                elif tid in tids.cross_e_tau:
                    trig_match_ok = trig_match_ok & (
                        (ak.sum(tau_match & ch_tau_mask, axis=1) >= 1) &
                        (ak.sum(e_match & e_ctrl, axis=1) >= 1)
                    )
                    if_mu_fired = base_ok & mu_trig_any & (ak.sum(mu_match_any & mu_ctrl, axis=1) >= 1)
                    trig_match_ok = ak.where(mu_trig_any, trig_match_ok & if_mu_fired, trig_match_ok)
                elif tid in tids.cross_mu_tau:
                    trig_match_ok = trig_match_ok & (
                        (ak.sum(tau_match & ch_tau_mask, axis=1) >= 1) &
                        (ak.sum(mu_match & mu_ctrl, axis=1) >= 1)
                    )
                    if_e_fired = base_ok & e_trig_any & (ak.sum(e_match_any & e_ctrl, axis=1) >= 1)
                    trig_match_ok = ak.where(e_trig_any, trig_match_ok & if_e_fired, trig_match_ok)
                elif tid in tids.cross_tau_tau_any:
                    trig_match_ok = trig_match_ok & (ak.sum(tau_match & ch_tau_mask, axis=1) >= 1)

                trig_match = trig_match | trig_match_ok

                single_triggered = ak.where(trig_match_ok, True, single_triggered)
                ids = ak.where(trig_match_ok, np.float32(tid), np.float32(np.nan))
                matched_trigger_ids.append(ak.singletons(ak.nan_to_none(ids)))

            elif ch_key == "c2eSS1tau":
                base_ok = (
                    (ak.sum(e_ctrl, axis=1) == 2) &
                    (ak.sum(e_veto, axis=1) == 2) &
                    (ak.sum(mu_veto, axis=1) == 0) &
                    (ak.sum(ch_tau_mask, axis=1) == 1)
                )
                if not disable_triggers:
                    base_ok = base_ok & fired

                ok = ak.where(base_ok, ok, False)

                sel_electron_mask = sel_electron_mask | (ok & e_ctrl)
                sel_looseelectron_mask = sel_looseelectron_mask | (ok & e_veto)
                sel_tightelectron_mask = sel_tightelectron_mask | (ok & e_mask)
                sel_tau_mask = sel_tau_mask | (ok & ch_tau_mask)
                sel_isotau_mask = sel_isotau_mask | (ok & (ch_tau_mask & tau_iso_mask))

                e_charge = events.Electron.charge[e_ctrl]
                tau_charge = events.Tau.charge[ch_tau_mask]
                chargeok = ((np.abs(ak.sum(e_charge, axis=1)) == 2) &
                    (np.abs((ak.sum(e_charge, axis=1)) + (ak.sum(tau_charge, axis=1))) == 1))
                leptons_os = ak.where(ok, chargeok, leptons_os)

                tight_ok = ok & ((ak.sum((ch_tau_mask & tau_iso_mask), axis=1) == 1) &
                    (ak.sum(e_mask, axis=1) == 2))
                tight_sel = tight_sel | tight_ok

                trig_match_ok = base_ok & (ak.sum(e_match & e_ctrl, axis=1) >= 1)
                trig_match = trig_match | trig_match_ok

                single_triggered = ak.where(trig_match_ok, True, single_triggered)
                ids = ak.where(trig_match_ok, np.float32(tid), np.float32(np.nan))
                matched_trigger_ids.append(ak.singletons(ak.nan_to_none(ids)))

            elif ch_key == "c2muSS1tau":
                base_ok = (
                    (ak.sum(mu_ctrl, axis=1) == 2) &
                    (ak.sum(mu_veto, axis=1) == 2) &
                    (ak.sum(e_veto, axis=1) == 0) &
                    (ak.sum(ch_tau_mask, axis=1) == 1)
                )
                if not disable_triggers:
                    base_ok = base_ok & fired

                ok = ak.where(base_ok, ok, False)

                sel_muon_mask = sel_muon_mask | (ok & mu_ctrl)
                sel_loosemuon_mask = sel_loosemuon_mask | (ok & mu_veto)
                sel_tightmuon_mask = sel_tightmuon_mask | (ok & mu_mask)
                sel_tau_mask = sel_tau_mask | (ok & ch_tau_mask)
                sel_isotau_mask = sel_isotau_mask | (ok & (ch_tau_mask & tau_iso_mask))

                mu_charge = events.Muon.charge[mu_ctrl]
                tau_charge = events.Tau.charge[ch_tau_mask]
                chargeok = ((np.abs(ak.sum(mu_charge, axis=1)) == 2) &
                    (np.abs((ak.sum(mu_charge, axis=1)) + (ak.sum(tau_charge, axis=1))) == 1))
                leptons_os = ak.where(ok, chargeok, leptons_os)

                tight_ok = ok & ((ak.sum((ch_tau_mask & tau_iso_mask), axis=1) == 1) &
                    (ak.sum(mu_mask, axis=1) == 2))
                tight_sel = tight_sel | tight_ok

                trig_match_ok = base_ok & (ak.sum(mu_match & mu_ctrl, axis=1) >= 1)
                trig_match = trig_match | trig_match_ok

                single_triggered = ak.where(trig_match_ok, True, single_triggered)
                ids = ak.where(trig_match_ok, np.float32(tid), np.float32(np.nan))
                matched_trigger_ids.append(ak.singletons(ak.nan_to_none(ids)))

            elif ch_key == "cemuSS1tau":
                base_ok = (
                    (ak.sum(e_ctrl, axis=1) == 1) &
                    (ak.sum(e_veto, axis=1) == 1) &
                    (ak.sum(mu_ctrl, axis=1) == 1) &
                    (ak.sum(mu_veto, axis=1) == 1) &
                    (ak.sum(ch_tau_mask, axis=1) == 1)
                )
                if not disable_triggers:
                    base_ok = base_ok & fired

                ok = ak.where(base_ok, ok, False)

                sel_electron_mask = sel_electron_mask | (ok & e_ctrl)
                sel_looseelectron_mask = sel_looseelectron_mask | (ok & e_veto)
                sel_tightelectron_mask = sel_tightelectron_mask | (ok & e_mask)
                sel_muon_mask = sel_muon_mask | (ok & mu_ctrl)
                sel_loosemuon_mask = sel_loosemuon_mask | (ok & mu_veto)
                sel_tightmuon_mask = sel_tightmuon_mask | (ok & mu_mask)
                sel_tau_mask = sel_tau_mask | (ok & ch_tau_mask)
                sel_isotau_mask = sel_isotau_mask | (ok & (ch_tau_mask & tau_iso_mask))

                e_charge = events.Electron.charge[e_ctrl]
                mu_charge = events.Muon.charge[mu_ctrl]
                tau_charge = events.Tau.charge[ch_tau_mask]
                chargeok = ((np.abs(ak.sum(e_charge, axis=1) + ak.sum(mu_charge, axis=1)) == 2) &
                    (np.abs((ak.sum(e_charge, axis=1)) + ak.sum(mu_charge, axis=1) +
                    (ak.sum(tau_charge, axis=1))) == 1))
                leptons_os = ak.where(ok, chargeok, leptons_os)

                tight_ok = ok & ((ak.sum((ch_tau_mask & tau_iso_mask), axis=1) == 1) &
                    (ak.sum(e_mask, axis=1) == 1) & (ak.sum(mu_mask, axis=1) == 1))
                tight_sel = tight_sel | tight_ok

                trig_match_ok = base_ok
                if tid in tids.single_e:
                    # emu_from_e — accept ONLY events with e_only (anti-overlap)
                    trig_match_ok = trig_match_ok & e_only & (ak.sum(e_match & e_ctrl, axis=1) >= 1)

                elif tid in tids.single_mu:
                    # emu_from_mu — allow both_families; the matching/logic below handles e-side
                    trig_match_ok = trig_match_ok & (ak.sum(mu_match & mu_ctrl, axis=1) >= 1)
                    # for events with both triggers firing:
                    if_e_fired = base_ok & e_trig_any & (ak.sum(e_match_any & e_ctrl, axis=1) >= 1)
                    trig_match_ok = ak.where(e_trig_any, trig_match_ok & if_e_fired, trig_match_ok)

                trig_match = trig_match | trig_match_ok

                single_triggered = ak.where(trig_match_ok, True, single_triggered)
                ids = ak.where(trig_match_ok, np.float32(tid), np.float32(np.nan))
                matched_trigger_ids.append(ak.singletons(ak.nan_to_none(ids)))

            elif ch_key == "c2eSS":
                base_ok = (
                    (ak.sum(e_ctrl, axis=1) == 2) &
                    (ak.sum(e_veto, axis=1) == 2) &
                    (ak.sum(mu_veto, axis=1) == 0) &
                    (ak.sum(ch_tau_mask, axis=1) == 0)
                )
                if not disable_triggers:
                    base_ok = base_ok & fired

                ok = ak.where(base_ok, ok, False)

                sel_electron_mask = sel_electron_mask | (ok & e_ctrl)
                sel_looseelectron_mask = sel_looseelectron_mask | (ok & e_veto)
                sel_tightelectron_mask = sel_tightelectron_mask | (ok & e_mask)

                e_charge = events.Electron.charge[e_ctrl]
                chargeok = (np.abs(ak.sum(e_charge, axis=1)) == 0)
                leptons_os = ak.where(ok, chargeok, leptons_os)

                tight_ok = ok & (ak.sum(e_mask, axis=1) == 2)
                tight_sel = tight_sel | tight_ok

                trig_match_ok = base_ok & (ak.sum(e_match & e_ctrl, axis=1) >= 1)
                trig_match = trig_match | trig_match_ok

                single_triggered = ak.where(trig_match_ok, True, single_triggered)
                ids = ak.where(trig_match_ok, np.float32(tid), np.float32(np.nan))
                matched_trigger_ids.append(ak.singletons(ak.nan_to_none(ids)))

            elif ch_key == "c2muSS":
                base_ok = (
                    (ak.sum(mu_ctrl, axis=1) == 2) &
                    (ak.sum(mu_veto, axis=1) == 2) &
                    (ak.sum(e_veto, axis=1) == 0) &
                    (ak.sum(ch_tau_mask, axis=1) == 0)
                )
                if not disable_triggers:
                    base_ok = base_ok & fired

                ok = ak.where(base_ok, ok, False)

                sel_muon_mask = sel_muon_mask | (ok & mu_ctrl)
                sel_loosemuon_mask = sel_loosemuon_mask | (ok & mu_veto)
                sel_tightmuon_mask = sel_tightmuon_mask | (ok & mu_mask)

                mu_charge = events.Muon.charge[mu_ctrl]
                chargeok = (np.abs(ak.sum(mu_charge, axis=1)) == 0)
                leptons_os = ak.where(ok, chargeok, leptons_os)

                tight_ok = ok & (ak.sum(mu_mask, axis=1) == 2)
                tight_sel = tight_sel | tight_ok

                trig_match_ok = base_ok & (ak.sum(mu_match & mu_ctrl, axis=1) >= 1)
                trig_match = trig_match | trig_match_ok

                single_triggered = ak.where(trig_match_ok, True, single_triggered)
                ids = ak.where(trig_match_ok, np.float32(tid), np.float32(np.nan))
                matched_trigger_ids.append(ak.singletons(ak.nan_to_none(ids)))

            elif ch_key == "cemuSS":
                base_ok = (
                    (ak.sum(e_ctrl, axis=1) == 1) &
                    (ak.sum(e_veto, axis=1) == 1) &
                    (ak.sum(mu_ctrl, axis=1) == 1) &
                    (ak.sum(mu_veto, axis=1) == 1) &
                    (ak.sum(ch_tau_mask, axis=1) == 0)
                )
                if not disable_triggers:
                    base_ok = base_ok & fired

                ok = ak.where(base_ok, ok, False)

                sel_electron_mask = sel_electron_mask | (ok & e_ctrl)
                sel_looseelectron_mask = sel_looseelectron_mask | (ok & e_veto)
                sel_tightelectron_mask = sel_tightelectron_mask | (ok & e_mask)
                sel_muon_mask = sel_muon_mask | (ok & mu_ctrl)
                sel_loosemuon_mask = sel_loosemuon_mask | (ok & mu_veto)
                sel_tightmuon_mask = sel_tightmuon_mask | (ok & mu_mask)

                e_charge = events.Electron.charge[e_ctrl]
                mu_charge = events.Muon.charge[mu_ctrl]
                chargeok = (np.abs((ak.sum(e_charge, axis=1) + ak.sum(mu_charge, axis=1))) == 0)
                leptons_os = ak.where(ok, chargeok, leptons_os)

                tight_ok = ok & ((ak.sum(e_mask, axis=1) == 1) & (ak.sum(mu_mask, axis=1) == 1))
                tight_sel = tight_sel | tight_ok

                trig_match_ok = base_ok
                if tid in tids.single_e:
                    # emu_from_e — accept ONLY events with e_only (anti-overlap)
                    trig_match_ok = trig_match_ok & e_only & (ak.sum(e_match & e_ctrl, axis=1) >= 1)

                elif tid in tids.single_mu:
                    # emu_from_mu — allow both_families; the matching/logic below handles e-side
                    trig_match_ok = trig_match_ok & (ak.sum(mu_match & mu_ctrl, axis=1) >= 1)
                    # for events with both triggers firing:
                    if_e_fired = base_ok & e_trig_any & (ak.sum(e_match_any & e_ctrl, axis=1) >= 1)
                    trig_match_ok = ak.where(e_trig_any, trig_match_ok & if_e_fired, trig_match_ok)

                trig_match = trig_match | trig_match_ok

                single_triggered = ak.where(trig_match_ok, True, single_triggered)
                ids = ak.where(trig_match_ok, np.float32(tid), np.float32(np.nan))
                matched_trigger_ids.append(ak.singletons(ak.nan_to_none(ids)))

            elif ch_key == "ce2tau":
                base_ok = (
                    (ak.sum(e_ctrl, axis=1) == 1) &
                    (ak.sum(e_veto, axis=1) == 1) &
                    (ak.sum(mu_veto, axis=1) == 0) &
                    (ak.sum(ch_tau_mask, axis=1) == 2)
                )
                if not disable_triggers:
                    base_ok = base_ok & fired

                ok = ak.where(base_ok, ok, False)

                sel_electron_mask = sel_electron_mask | (ok & e_ctrl)
                sel_looseelectron_mask = sel_looseelectron_mask | (ok & e_veto)
                sel_tightelectron_mask = sel_tightelectron_mask | (ok & e_mask)
                sel_tau_mask = sel_tau_mask | (ok & ch_tau_mask)
                sel_isotau_mask = sel_isotau_mask | (ok & (ch_tau_mask & tau_iso_mask))

                e_charge = events.Electron.charge[e_ctrl]
                tau_charge = events.Tau.charge[ch_tau_mask]
                chargeok = ((np.abs((ak.sum(e_charge, axis=1)) + (ak.sum(tau_charge, axis=1))) == 1))
                leptons_os = ak.where(ok, chargeok, leptons_os)

                tight_ok = ok & ((ak.sum((ch_tau_mask & tau_iso_mask), axis=1) == 2) &
                    (ak.sum(e_mask, axis=1) == 1))
                tight_sel = tight_sel | tight_ok

                trig_match_ok = base_ok & (ak.sum(e_match & e_ctrl, axis=1) >= 1)
                trig_match = trig_match | trig_match_ok

                single_triggered = ak.where(trig_match_ok, True, single_triggered)
                ids = ak.where(trig_match_ok, np.float32(tid), np.float32(np.nan))
                matched_trigger_ids.append(ak.singletons(ak.nan_to_none(ids)))

            elif ch_key == "cmu2tau":
                base_ok = (
                    (ak.sum(mu_ctrl, axis=1) == 1) &
                    (ak.sum(mu_veto, axis=1) == 1) &
                    (ak.sum(e_veto, axis=1) == 0) &
                    (ak.sum(ch_tau_mask, axis=1) == 2)
                )
                if not disable_triggers:
                    base_ok = base_ok & fired

                ok = ak.where(base_ok, ok, False)

                sel_muon_mask = sel_muon_mask | (ok & mu_ctrl)
                sel_loosemuon_mask = sel_loosemuon_mask | (ok & mu_veto)
                sel_tightmuon_mask = sel_tightmuon_mask | (ok & mu_mask)
                sel_tau_mask = sel_tau_mask | (ok & ch_tau_mask)
                sel_isotau_mask = sel_isotau_mask | (ok & (ch_tau_mask & tau_iso_mask))

                mu_charge = events.Muon.charge[mu_ctrl]
                tau_charge = events.Tau.charge[ch_tau_mask]
                chargeok = ((np.abs((ak.sum(mu_charge, axis=1)) + (ak.sum(tau_charge, axis=1))) == 1))
                leptons_os = ak.where(ok, chargeok, leptons_os)

                tight_ok = ok & ((ak.sum((ch_tau_mask & tau_iso_mask), axis=1) == 2) &
                    (ak.sum(mu_mask, axis=1) == 1))
                tight_sel = tight_sel | tight_ok

                trig_match_ok = base_ok & (ak.sum(mu_match & mu_ctrl, axis=1) >= 1)
                trig_match = trig_match | trig_match_ok

                single_triggered = ak.where(trig_match_ok, True, single_triggered)
                ids = ak.where(trig_match_ok, np.float32(tid), np.float32(np.nan))
                matched_trigger_ids.append(ak.singletons(ak.nan_to_none(ids)))

            elif ch_key == "c4tau":
                base_ok = (
                    (ak.sum(mu_veto, axis=1) == 0) &
                    (ak.sum(e_veto, axis=1) == 0) &
                    (ak.sum(ch_tau_mask, axis=1) == 4)
                )
                if not disable_triggers:
                    base_ok = base_ok & fired

                ok = ak.where(base_ok, ok, False)

                sel_tau_mask = sel_tau_mask | (ok & ch_tau_mask)
                sel_isotau_mask = sel_isotau_mask | (ok & (ch_tau_mask & tau_iso_mask))

                tau_charge = events.Tau.charge[ch_tau_mask]
                chargeok = (np.abs((ak.sum(tau_charge, axis=1))) == 0)
                leptons_os = ak.where(ok, chargeok, leptons_os)

                tight_ok = ok & (ak.sum((ch_tau_mask & tau_iso_mask), axis=1) == 4)
                tight_sel = tight_sel | tight_ok

                trig_match_ok = base_ok
                if tid in tids.cross_tau_tau_any:
                    trig_match_ok = trig_match_ok & (ak.sum(tau_match & ch_tau_mask, axis=1) >= 1)

                trig_match = trig_match | trig_match_ok

                single_triggered = ak.where(trig_match_ok, True, single_triggered)
                ids = ak.where(trig_match_ok, np.float32(tid), np.float32(np.nan))
                matched_trigger_ids.append(ak.singletons(ak.nan_to_none(ids)))

            elif ch_key == "cttbarMR":

                mr_z_mass = 91.18
                mr_z_window = 10.0

                if self.config_inst.campaign.x.year in {2024, 2025, 2026}:
                    mr_btag_tagger = "UParTAK4"
                    mr_btag_discriminator = "btagUParTAK4B"
                else:
                    mr_btag_tagger = "particleNet"
                    mr_btag_discriminator = "btagPNetB"
                mr_btagcut_tight = self.config_inst.x.btag_working_points[mr_btag_tagger]["tight"]
                mr_btagcut_medium = self.config_inst.x.btag_working_points[mr_btag_tagger]["medium"]
                mr_btagcut_loose = self.config_inst.x.btag_working_points[mr_btag_tagger]["loose"]

                mr_jet_mask = (
                    (events.Jet.jetId == 6) &
                    (events.Jet.pt > 20.0) &
                    (abs(events.Jet.eta) < 2.5)
                )
                mr_jet_mask = mr_jet_mask & ak.all(
                    events.Jet.metric_table(events.Electron[e_ctrl]) > 0.5, axis=2,
                )
                mr_jet_mask = mr_jet_mask & ak.all(
                    events.Jet.metric_table(events.Muon[mu_ctrl]) > 0.5, axis=2,
                )
                mr_jet_notau = mr_jet_mask & ak.all(
                    events.Jet.metric_table(events.Tau[noid_tau_mask]) > 0.5, axis=2,
                )
                mr_jet_btag = events.Jet[mr_btag_discriminator]
                mr_jet_ok = (
                    (ak.sum(mr_jet_mask, axis=1) >= 2) &
                    (ak.sum(mr_jet_notau & (mr_jet_btag > mr_btagcut_medium), axis=1) >= 1) &
                    (ak.sum(mr_jet_mask & (mr_jet_btag > mr_btagcut_loose), axis=1) >= 2) &
                    (ak.sum(mr_jet_notau & (mr_jet_btag > mr_btagcut_tight), axis=1) < 1)
                )

                mr_n_e = ak.sum(e_mask, axis=1)
                mr_n_mu = ak.sum(mu_mask, axis=1)
                mr_es = ak.pad_none(events.Electron[e_mask], 2, axis=1)
                mr_mus = ak.pad_none(events.Muon[mu_mask], 2, axis=1)

                mr_mll = ak.where(
                    mr_n_e == 2,
                    (mr_es[:, 0] * 1 + mr_es[:, 1] * 1).mass,
                    ak.where(
                        mr_n_mu == 2,
                        (mr_mus[:, 0] * 1 + mr_mus[:, 1] * 1).mass,
                        (mr_es[:, 0] * 1 + mr_mus[:, 0] * 1).mass,
                    ),
                )
                mr_mll = ak.fill_none(mr_mll, 0.0)

                mr_charge_prod = ak.where(
                    mr_n_e == 2,
                    mr_es[:, 0].charge * mr_es[:, 1].charge,
                    ak.where(
                        mr_n_mu == 2,
                        mr_mus[:, 0].charge * mr_mus[:, 1].charge,
                        mr_es[:, 0].charge * mr_mus[:, 0].charge,
                    ),
                )
                mr_os_ok = ak.fill_none(mr_charge_prod < 0, False)
                mr_same_flavour = (mr_n_e == 2) | (mr_n_mu == 2)

                base_ok = (
                    (mr_n_e + mr_n_mu == 2) &
                    (ak.sum(e_veto, axis=1) + ak.sum(mu_veto, axis=1) == 2) &
                    mr_os_ok &
                    (mr_mll > 12.0) &
                    (~mr_same_flavour | (np.abs(mr_mll - mr_z_mass) > mr_z_window)) &
                    (ak.sum(ch_tau_mask, axis=1) >= 1) &
                    mr_jet_ok
                )

                mr_total_charge = (
                    ak.sum(events.Electron.charge[e_ctrl], axis=1) +
                    ak.sum(events.Muon.charge[mu_ctrl], axis=1) +
                    ak.sum(events.Tau.charge[ch_tau_mask], axis=1)
                )
                base_ok = base_ok & (
                    (ak.sum(ch_tau_mask, axis=1) != 2) | (np.abs(mr_total_charge) != 0)
                )

                if not disable_triggers:
                    base_ok = base_ok & fired

                ok = ak.where(base_ok, ok, False)

                sel_electron_mask = sel_electron_mask | (ok & e_ctrl)
                sel_looseelectron_mask = sel_looseelectron_mask | (ok & e_veto)
                sel_tightelectron_mask = sel_tightelectron_mask | (ok & e_mask)
                sel_muon_mask = sel_muon_mask | (ok & mu_ctrl)
                sel_loosemuon_mask = sel_loosemuon_mask | (ok & mu_veto)
                sel_tightmuon_mask = sel_tightmuon_mask | (ok & mu_mask)
                sel_tau_mask = sel_tau_mask | (ok & ch_tau_mask)
                sel_isotau_mask = sel_isotau_mask | (ok & (ch_tau_mask & tau_iso_mask))

                chargeok = mr_os_ok
                leptons_os = ak.where(ok, chargeok, leptons_os)

                tight_ok = ok & (ak.sum((ch_tau_mask & tau_iso_mask), axis=1) >= 1)
                tight_sel = tight_sel | tight_ok

                trig_match_ok = base_ok
                if tid in tids.single_e:
                    trig_match_ok = trig_match_ok & (ak.sum(e_match & e_ctrl, axis=1) >= 1)
                    if_mu_fired = base_ok & mu_trig_any & (ak.sum(mu_match_any & mu_ctrl, axis=1) >= 1)
                    trig_match_ok = ak.where(mu_trig_any, trig_match_ok & if_mu_fired, trig_match_ok)
                elif tid in tids.single_mu:
                    trig_match_ok = trig_match_ok & (ak.sum(mu_match & mu_ctrl, axis=1) >= 1)
                    if_e_fired = base_ok & e_trig_any & (ak.sum(e_match_any & e_ctrl, axis=1) >= 1)
                    trig_match_ok = ak.where(e_trig_any, trig_match_ok & if_e_fired, trig_match_ok)
                elif tid in tids.double_e:
                    trig_match_ok = trig_match_ok & (ak.sum(e_match & e_ctrl, axis=1) >= 2)
                elif tid in tids.double_mu:
                    trig_match_ok = trig_match_ok & (ak.sum(mu_match & mu_ctrl, axis=1) >= 2)
                elif tid in tids.double_emu:
                    trig_match_ok = trig_match_ok & (
                        (ak.sum(e_match & e_ctrl, axis=1) >= 1) &
                        (ak.sum(mu_match & mu_ctrl, axis=1) >= 1)
                    )
                elif tid in tids.cross_e_tau:
                    trig_match_ok = trig_match_ok & (
                        (ak.sum(tau_match & ch_tau_mask, axis=1) >= 1) &
                        (ak.sum(e_match & e_ctrl, axis=1) >= 1)
                    )
                elif tid in tids.cross_mu_tau:
                    trig_match_ok = trig_match_ok & (
                        (ak.sum(tau_match & ch_tau_mask, axis=1) >= 1) &
                        (ak.sum(mu_match & mu_ctrl, axis=1) >= 1)
                    )

                trig_match = trig_match | trig_match_ok

                single_triggered = ak.where(trig_match_ok, True, single_triggered)
                ids = ak.where(trig_match_ok, np.float32(tid), np.float32(np.nan))
                matched_trigger_ids.append(ak.singletons(ak.nan_to_none(ids)))

            elif ch_key == "cwzMR":

                mr_z_mass = 91.18
                mr_z_window = 10.0

                def mr_has_z_pair(objs):
                    pairs = ak.combinations(objs, 2, axis=1, fields=["l1", "l2"])
                    os_pairs = (pairs.l1.charge * pairs.l2.charge) < 0
                    masses = (pairs.l1 * 1 + pairs.l2 * 1).mass
                    return ak.any(os_pairs & (abs(masses - mr_z_mass) < mr_z_window), axis=1)

                mr_z_from_e = mr_has_z_pair(events.Electron[e_mask])
                mr_z_from_mu = mr_has_z_pair(events.Muon[mu_mask])

                e_charge = events.Electron.charge[e_mask]
                mu_charge = events.Muon.charge[mu_mask]
                # tau_charge = events.Tau.charge[ch_tau_mask]

                # for the 3 same-flavour leptons, the Z pair is the OS one closest to mZ
                charge_3mu = mr_z_from_mu & (np.abs(ak.sum(mu_charge, axis=1)) == 1)
                charge_3e = mr_z_from_e & (np.abs(ak.sum(e_charge, axis=1)) == 1)
                charge_2mue = mr_z_from_mu & (ak.sum(mu_charge, axis=1) == 0)
                charge_mu2e = mr_z_from_e & (ak.sum(e_charge, axis=1) == 0)

                base_ok_3mu = (
                    (ak.sum(e_veto, axis=1) == 0) &
                    (ak.sum(mu_ctrl, axis=1) == 3) &
                    (ak.sum(mu_veto, axis=1) == 3) &
                    (ak.sum(mu_mask, axis=1) == 3) &
                    (ak.sum(ch_tau_mask, axis=1) >= 1) &
                    charge_3mu
                )
                base_ok_2mue = (
                    (ak.sum(e_ctrl, axis=1) == 1) &
                    (ak.sum(e_veto, axis=1) == 1) &
                    (ak.sum(e_mask, axis=1) == 1) &
                    (ak.sum(mu_ctrl, axis=1) == 2) &
                    (ak.sum(mu_veto, axis=1) == 2) &
                    (ak.sum(mu_mask, axis=1) == 2) &
                    (ak.sum(ch_tau_mask, axis=1) >= 1) &
                    charge_2mue
                )
                base_ok_mu2e = (
                    (ak.sum(e_ctrl, axis=1) == 2) &
                    (ak.sum(e_veto, axis=1) == 2) &
                    (ak.sum(e_mask, axis=1) == 2) &
                    (ak.sum(mu_ctrl, axis=1) == 1) &
                    (ak.sum(mu_veto, axis=1) == 1) &
                    (ak.sum(mu_mask, axis=1) == 1) &
                    (ak.sum(ch_tau_mask, axis=1) >= 1) &
                    charge_mu2e
                )
                base_ok_3e = (
                    (ak.sum(mu_veto, axis=1) == 0) &
                    (ak.sum(e_ctrl, axis=1) == 3) &
                    (ak.sum(e_veto, axis=1) == 3) &
                    (ak.sum(e_mask, axis=1) == 3) &
                    (ak.sum(ch_tau_mask, axis=1) >= 1) &
                    charge_3e
                )
                base_ok = base_ok_3mu | base_ok_2mue | base_ok_mu2e | base_ok_3e

                # we can use the charge criteria to ensure orthogonality with the 3l1th SR
                mr_all_iso = (
                    ak.sum(ch_tau_mask & tau_iso_mask, axis=1) == ak.sum(ch_tau_mask, axis=1)
                )
                mr_total_charge = (
                    ak.sum(events.Electron.charge[e_ctrl], axis=1) +
                    ak.sum(events.Muon.charge[mu_ctrl], axis=1) +
                    ak.sum(events.Tau.charge[ch_tau_mask], axis=1)
                )
                base_ok = base_ok & (
                    (ak.sum(ch_tau_mask, axis=1) != 1) | ~mr_all_iso |
                    (np.abs(mr_total_charge) != 0)
                )

                if not disable_triggers:
                    base_ok = base_ok & fired

                ok = ak.where(base_ok, ok, False)

                sel_electron_mask = sel_electron_mask | (ok & e_ctrl)
                sel_looseelectron_mask = sel_looseelectron_mask | (ok & e_veto)
                sel_tightelectron_mask = sel_tightelectron_mask | (ok & e_mask)
                sel_muon_mask = sel_muon_mask | (ok & mu_ctrl)
                sel_loosemuon_mask = sel_loosemuon_mask | (ok & mu_veto)
                sel_tightmuon_mask = sel_tightmuon_mask | (ok & mu_mask)
                sel_tau_mask = sel_tau_mask | (ok & ch_tau_mask)
                sel_isotau_mask = sel_isotau_mask | (ok & (ch_tau_mask & tau_iso_mask))

                chargeok = charge_3mu | charge_2mue | charge_mu2e | charge_3e
                leptons_os = ak.where(ok, chargeok, leptons_os)

                # ch_tau_mask = ch_tau_mask & (events.Tau[get_tau_tagger("e")] >= wp_config.tau_vs_e.vloose)
                tight_ok = ok & ((ak.sum((ch_tau_mask & tau_iso_mask), axis=1) >= 1))
                tight_sel = tight_sel | tight_ok

                trig_match_ok = base_ok
                if tid in tids.single_e:
                    trig_match_ok = trig_match_ok & (ak.sum(e_match & e_ctrl, axis=1) >= 1)
                    if_mu_fired = base_ok & mu_trig_any & (ak.sum(mu_match_any & mu_ctrl, axis=1) >= 1)
                    trig_match_ok = ak.where(mu_trig_any, trig_match_ok & if_mu_fired, trig_match_ok)
                elif tid in tids.single_mu:
                    trig_match_ok = trig_match_ok & (ak.sum(mu_match & mu_ctrl, axis=1) >= 1)
                    if_e_fired = base_ok & e_trig_any & (ak.sum(e_match_any & e_ctrl, axis=1) >= 1)
                    trig_match_ok = ak.where(e_trig_any, trig_match_ok & if_e_fired, trig_match_ok)
                elif tid in tids.cross_e_tau:
                    trig_match_ok = trig_match_ok & (
                        (ak.sum(tau_match & ch_tau_mask, axis=1) >= 1) &
                        (ak.sum(e_match & e_ctrl, axis=1) >= 1)
                    )
                    if_mu_fired = base_ok & mu_trig_any & (ak.sum(mu_match_any & mu_ctrl, axis=1) >= 1)
                    trig_match_ok = ak.where(mu_trig_any, trig_match_ok & if_mu_fired, trig_match_ok)
                elif tid in tids.cross_mu_tau:
                    trig_match_ok = trig_match_ok & (
                        (ak.sum(tau_match & ch_tau_mask, axis=1) >= 1) &
                        (ak.sum(mu_match & mu_ctrl, axis=1) >= 1)
                    )
                    if_e_fired = base_ok & e_trig_any & (ak.sum(e_match_any & e_ctrl, axis=1) >= 1)
                    trig_match_ok = ak.where(e_trig_any, trig_match_ok & if_e_fired, trig_match_ok)

                trig_match = trig_match | trig_match_ok

                single_triggered = ak.where(trig_match_ok, True, single_triggered)
                ids = ak.where(trig_match_ok, np.float32(tid), np.float32(np.nan))
                matched_trigger_ids.append(ak.singletons(ak.nan_to_none(ids)))

            elif ch_key == "cdyMR":

                mr_z_mass = 91.18
                mr_z_window = 10.0
                mr_mus = ak.pad_none(events.Muon[mu_mask], 2, axis=1)
                mr_mumu_mass = (mr_mus[:, 0] * 1 + mr_mus[:, 1] * 1).mass
                mr_z_ok = ak.fill_none(abs(mr_mumu_mass - mr_z_mass) < mr_z_window, False)

                base_ok = (
                    (ak.sum(mu_ctrl, axis=1) == 2) &
                    (ak.sum(mu_veto, axis=1) == 2) &
                    (ak.sum(mu_mask, axis=1) == 2) &
                    (ak.sum(e_veto, axis=1) == 0) &
                    (ak.sum(ch_tau_mask, axis=1) >= 1) &
                    mr_z_ok
                )

                # we can use the charge criteria to ensure orthogonality with the 2lSS1th
                # and 2l2th SRs
                mr_all_iso = (
                    ak.sum(ch_tau_mask & tau_iso_mask, axis=1) == ak.sum(ch_tau_mask, axis=1)
                )
                mr_total_charge = (
                    ak.sum(events.Muon.charge[mu_ctrl], axis=1) +
                    ak.sum(events.Tau.charge[ch_tau_mask], axis=1)
                )
                base_ok = base_ok & (
                    (np.abs(ak.sum(events.Muon.charge[mu_ctrl], axis=1)) == 0) &
                    ((ak.sum(ch_tau_mask, axis=1) != 2) | ~mr_all_iso |
                        (np.abs(mr_total_charge) != 0))
                )

                if not disable_triggers:
                    base_ok = base_ok & fired

                ok = ak.where(base_ok, ok, False)

                sel_muon_mask = sel_muon_mask | (ok & mu_ctrl)
                sel_loosemuon_mask = sel_loosemuon_mask | (ok & mu_veto)
                sel_tightmuon_mask = sel_tightmuon_mask | (ok & mu_mask)
                sel_tau_mask = sel_tau_mask | (ok & ch_tau_mask)
                sel_isotau_mask = sel_isotau_mask | (ok & (ch_tau_mask & tau_iso_mask))

                mu_charge = events.Muon.charge[mu_mask]
                # tau_charge = events.Tau.charge[ch_tau_mask]
                chargeok = (np.abs((ak.sum(mu_charge, axis=1))) == 0)
                leptons_os = ak.where(ok, chargeok, leptons_os)

                tight_ok = ok & (ak.sum((ch_tau_mask & tau_iso_mask), axis=1) >= 1)
                tight_sel = tight_sel | tight_ok

                trig_match_ok = base_ok
                if tid in tids.single_mu:
                    trig_match_ok = trig_match_ok & mu_only_emutau & (ak.sum(mu_match & mu_ctrl, axis=1) >= 1)
                elif tid in tids.double_mu:
                    trig_match_ok = trig_match_ok & (
                        (ak.sum(mu_match & mu_ctrl, axis=1) >= 1)
                    )

                trig_match = trig_match | trig_match_ok

                single_triggered = ak.where(trig_match_ok, True, single_triggered)
                ids = ak.where(trig_match_ok, np.float32(tid), np.float32(np.nan))

        # accumulate over triggers
            good_evt = ak.where(ok, True, good_evt)

        if ch_key != "ceormu":
            channel_id = update_channel_ids(events, channel_id, spec.id, good_evt)

    # some final type conversions
    channel_id = ak.values_astype(channel_id, np.uint32)
    leptons_os = ak.fill_none(leptons_os, False)
    tight_sel = ak.fill_none(tight_sel, False)
    tight_sel_bdt = ak.fill_none(tight_sel_bdt, False)
    trig_match = ak.fill_none(trig_match, False)
    trig_match_bdt = ak.fill_none(trig_match_bdt, False)
    ok_bdt_eormu = ak.fill_none(ok_bdt_eormu, False)

    # concatenate matched trigger ids
    empty_ids = ak.singletons(full_like(events.event, 0, dtype=np.int32), axis=0)[:, :0]
    merge_ids = lambda ids: ak.values_astype(ak.concatenate(ids, axis=1), np.int32) if ids else empty_ids
    matched_trigger_ids = merge_ids(matched_trigger_ids)
    lepton_part_trigger_ids = merge_ids(lepton_part_trigger_ids)

    # save new columns
    events = set_ak_column(events, "channel_id", channel_id)
    events = set_ak_column(events, "leptons_os", leptons_os)
    events = set_ak_column(events, "tau2_isolated", tau2_isolated)
    events = set_ak_column(events, "single_triggered", single_triggered)
    events = set_ak_column(events, "cross_triggered", cross_triggered)
    events = set_ak_column(events, "matched_trigger_ids", matched_trigger_ids)

    # new columns for lepton bdt
    events = set_ak_column(events, "ok_bdt_eormu", ok_bdt_eormu)
    events = set_ak_column(events, "tight_sel_bdt", tight_sel_bdt)
    events = set_ak_column(events, "trig_match_bdt", trig_match_bdt)

    # new selections for the physical channels
    events = set_ak_column(events, "tight_sel", tight_sel)
    events = set_ak_column(events, "trig_match", trig_match)

    # convert lepton masks to sorted indices (pt for e/mu, iso for tau)
    sel_electron_indices = sorted_indices_from_mask(sel_electron_mask, events.Electron.pt, ascending=False)
    sel_muon_indices = sorted_indices_from_mask(sel_muon_mask, events.Muon.pt, ascending=False)
    sel_tau_indices = sorted_indices_from_mask(sel_tau_mask, tau_sorting_key, ascending=False)
    sel_noid_tau_indicies = sorted_indices_from_mask(sel_noid_tau_mask, events.Tau.pt, ascending=False)

    sel_looseelectron_indices = sorted_indices_from_mask(sel_looseelectron_mask, events.Electron.pt, ascending=False)
    sel_loosemuon_indices = sorted_indices_from_mask(sel_loosemuon_mask, events.Muon.pt, ascending=False)

    sel_tightelectron_indices = sorted_indices_from_mask(sel_tightelectron_mask, events.Electron.pt, ascending=False)
    sel_tightmuon_indices = sorted_indices_from_mask(sel_tightmuon_mask, events.Muon.pt, ascending=False)
    sel_isotau_indices = sorted_indices_from_mask(sel_isotau_mask, tau_sorting_key, ascending=False)
    # Saving cone-pT for fakeable leptons
    events = set_ak_column(
        events,
        "Electron.cone_pt",
        electron_cone_pt,
    )
    events = set_ak_column(
        events,
        "Muon.cone_pt",
        muon_cone_pt,
    )
    # events = set_ak_column(events, "Electron", events.Electron[sel_electron_indices])
    events = set_ak_column(events, "ElectronLoose", events.Electron[sel_looseelectron_indices])
    events = set_ak_column(events, "ElectronTight", events.Electron[sel_tightelectron_indices])
    # events = set_ak_column(events, "Muon", events.Muon[sel_muon_indices])
    events = set_ak_column(events, "MuonLoose", events.Muon[sel_loosemuon_indices])
    events = set_ak_column(events, "MuonTight", events.Muon[sel_tightmuon_indices])
    # events = set_ak_column(events, "Tau", events.Tau[sel_tau_indices])
    events = set_ak_column(events, "TauIso", events.Tau[sel_isotau_indices])
    events = set_ak_column(events, "TauNoID", events.Tau[sel_noid_tau_indicies])

    return events, SelectionResult(
        steps={
            "lepton": (channel_id != 0) | ok_bdt_eormu,
        },
        objects={
            "Electron": {
                "Electron": sel_electron_indices,
                "ElectronLoose": sel_looseelectron_indices,
                "ElectronTight": sel_tightelectron_indices,
            },
            "Muon": {
                "Muon": sel_muon_indices,
                "MuonLoose": sel_loosemuon_indices,
                "MuonTight": sel_tightmuon_indices,
            },
            "Tau": {
                "Tau": sel_tau_indices,
                "TauIso": sel_isotau_indices,
                "TauNoID": sel_noid_tau_indicies,
            },
        },
        aux={
            # save the selected lepton pair for the duration of the selection
            # multiplication of a coffea particle with 1 yields the lorentz vector
            "lepton_pair": ak.concatenate(
                [
                    events.Electron[sel_electron_indices] * 1,
                    events.Muon[sel_muon_indices] * 1,
                    events.Tau[sel_tau_indices] * 1,
                ],
                axis=1,
            )[:, :2],

            # save the matched trigger ids of the trigger with jet legs for the duration of the selection
            # these will be updated in the jet selection and then stored in the matched_trigger_ids column
            "lepton_part_trigger_ids": lepton_part_trigger_ids,
            # save the leading taus for the duration of the selection
            # exactly 1 for etau/mutau and exactly 2 for tautau
            "leading_taus": leading_taus,
            "eles": sel_electron_indices,
            "mus": sel_muon_indices,
            "taus": sel_tau_indices,
            # new collections
            "eles_loose": sel_looseelectron_indices,
            "mus_loose": sel_loosemuon_indices,
            "eles_tight": sel_tightelectron_indices,
            "mus_tight": sel_tightmuon_indices,
            "taus_iso": sel_isotau_indices,
            "taus_noid": sel_noid_tau_indicies,
        },
    )


@lepton_selection.init
def lepton_selection_init(self: Selector, **kwargs) -> None:
    # add column to load the raw tau tagger score for tau sorting
    tagger = self.config_inst.x.tau_tagger
    # For sorting, always use raw scores; for PNet this is "rawPNetVSjet",
    # for DeepTau this is "rawDeepTau...VSjet"
    self.uses.add(f"Tau.raw{tagger}VSjet")

    # ensuring the v15 jetID fix required for the ttbar mr jet selection
    if self.ffmr and self.config_inst.x.jet_id_has_multiplicity:
        self.uses.add(jet_id)


# fills the measurement regions instead of the physics channels, used by the default_ffmr selector.
# the two never run together, so all output columns keep their usual names
lepton_selection_ffmr = lepton_selection.derive("lepton_selection_ffmr", cls_dict={"ffmr": True})

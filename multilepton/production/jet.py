# coding: utf-8

"""
Jet scale factor production.
"""

from __future__ import annotations

import functools

import law

from columnflow.production import Producer, producer
from columnflow.util import maybe_import, load_correction_set
from columnflow.columnar_util import set_ak_column, flat_np_view, layout_ak_array


ak = maybe_import("awkward")
np = maybe_import("numpy")
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


@producer(
    uses={
        "channel_id", "Jet.{pt,eta,phi,mass}",
    },
    mc_only=True,
    get_jet_file=(lambda self, external_files: external_files.trigger_sf.jet),
    get_jet_corrector=(lambda self: self.config_inst.x.jet_trigger_corrector),
    efficiency_name="jet_trigger_eff",
)
def jet_trigger_efficiencies(
    self: Producer,
    events: ak.Array,
    jet_mask: ak.Array | type(Ellipsis) = Ellipsis,
    **kwargs,
) -> ak.Array:
    """
    Producer for jet trigger efficiencies derived by the CCLUB group at object level. Requires an external file in the
    config under ``trigger_sf.jet``.

    *get_jet_file* can be adapted in a subclass in case it is stored differently in the external files. A correction
    set named ``"jet_trigger_corrector"`` is extracted from it.

    Resources:
    https://gitlab.cern.ch/cclubbtautau/AnalysisCore/-/tree/cclub_cmssw15010/data/TriggerScaleFactors?ref_type=heads
    """

    # flat absolute eta and pt views
    abs_eta = flat_np_view(abs(events.Jet.eta[jet_mask]), axis=1)
    pt = flat_np_view(events.Jet.pt[jet_mask], axis=1)
    variable_map = {
        "pt": pt,
        "abseta": abs_eta,
    }

    for kind in ["data", "mc"]:
        for syst, postfix in [
            ("nom", ""),
            ("up", "_up"),
            ("down", "_down"),
        ]:
            variable_map_syst = {
                **variable_map,
                "syst": syst,
                "data_or_mc": kind,
            }
            inputs = [variable_map_syst[inp.name] for inp in self.jet_trig_corrector.inputs]
            sf_flat = self.jet_trig_corrector(*inputs)
            sf = layout_ak_array(sf_flat, events.Jet.pt[jet_mask])
            events = set_ak_column(events, f"{self.efficiency_name}_{kind}{postfix}", sf, value_type=np.float32)
    return events


@jet_trigger_efficiencies.init
def jet_trigger_efficiencies_init(self: Producer, **kwargs) -> None:
    # add the product of nominal and up/down variations to produced columns
    self.produces.add(f"{self.efficiency_name}_{{data,mc}}{{,_up,_down}}")


@jet_trigger_efficiencies.requires
def jet_trigger_efficiencies_requires(self: Producer, task: law.Task, reqs: dict) -> None:
    from columnflow.tasks.external import BundleExternalFiles
    if "external_files" in reqs:
        return
    reqs["external_files"] = BundleExternalFiles.req(task)


@jet_trigger_efficiencies.setup
def jet_trigger_efficiencies_setup(
    self: Producer,
    task: law.Task,
    reqs: dict,
    inputs: dict,
    reader_targets: law.util.InsertableDict,
) -> None:
    bundle = reqs["external_files"]

    # create the trigger and id correctors
    correction_set = load_correction_set(self.get_jet_file(bundle.files))
    # print("Available keys:", list(correction_set.keys()))
    self.jet_trig_corrector = correction_set[self.get_jet_corrector()]
    # assert self.jet_trig_corrector.version in [0, 1]

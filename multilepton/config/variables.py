# coding: utf-8

"""
Definition of variables.
"""

import order as od

from functools import partial

from columnflow.columnar_util import EMPTY_FLOAT
from columnflow.util import maybe_import

ak = maybe_import("awkward")


def build_ht(events):
    objects = ak.concatenate([events.Electron * 1, events.Muon * 1, events.Tau * 1, events.Jet * 1], axis=1)[:, :]
    objects_sum = objects.sum(axis=1)
    return objects_sum.pt


# build variables for dilepton, dijet, and hh
def delta_r12(vectors):
    # delta r between first two elements
    dr = ak.firsts(vectors[:, :1], axis=1).delta_r(ak.firsts(vectors[:, 1:2], axis=1))
    return ak.fill_none(dr, EMPTY_FLOAT)


def build_dilep(events, which=None):
    leps = ak.concatenate([events.Electron * 1, events.Muon * 1, events.Tau * 1], axis=1)[:, :2]
    if which == "dr":
        return delta_r12(leps)
    dilep = leps.sum(axis=1)
    if which is None:
        return dilep * 1
    if which == "mass":
        return dilep.mass
    if which == "pt":
        return dilep.pt
    if which == "eta":
        return dilep.eta
    if which == "abs_eta":
        return abs(dilep.eta)
    if which == "phi":
        return dilep.phi
    if which == "energy":
        return dilep.energy
    raise ValueError(f"Unknown which: {which}")


def build_m4l(events):
    objects = ak.concatenate([events.Electron * 1, events.Muon * 1], axis=1)[:, :]
    objects_sum = objects.sum(axis=1)
    return objects_sum.mass


def build_nbjets(config, events, which=None, wp="medium"):
    if which == "btagPNetB":
        wp_value = config.x.btag_working_points["particleNet"][wp]
    elif which == "btagDeepFlavB":
        wp_value = config.x.btag_working_points["deepjet"][wp]
    else:
        raise ValueError(f"Unknown which: {which}")
    bjet_mask = events.Jet[which] >= wp_value
    objects = events.Jet[bjet_mask]
    objects_num = ak.num(objects, axis=1)
    return objects_num


def add_variables(config: od.Config) -> None:
    """
    Adds all variables to a *config*.
    """
    build_ht.inputs = ["{Electron,Muon,Tau,Jet}.{pt,eta,phi,mass}"]
    build_dilep.inputs = ["{Electron,Muon,Tau}.{pt,eta,phi,mass}"]
    build_m4l.inputs = ["{Electron,Muon}.{pt,eta,phi,mass}"]
    build_nbjets.inputs = ["Jet.{btagPNetB,btagDeepFlavB}"]

    add_variable(
        config,
        name="event",
        expression="event",
        binning=(1, 0.0, 1.0e9),
        x_title="Event number",
        discrete_x=True,
    )
    add_variable(
        config,
        name="run",
        expression="run",
        binning=(1, 100000.0, 500000.0),
        x_title="Run number",
        discrete_x=True,
    )
    add_variable(
        config,
        name="lumi",
        expression="luminosityBlock",
        binning=(1, 0.0, 5000.0),
        x_title="Luminosity block",
        discrete_x=True,
    )
    add_variable(
        config,
        name="ht",
        expression=partial(build_ht),
        aux={"inputs": build_ht.inputs},
        binning=[0, 80, 120, 160, 200, 240, 280, 320, 400, 500, 600, 800],
        unit="GeV",
        x_title="HT",
    )
    add_variable(
        config,
        name="jet_pt",
        expression="Jet.pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"all Jet $p_{T}$",
    )
    add_variable(
        config,
        name="jet1_pt",
        expression="Jet.pt[:,0]",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Leading jet $p_{T}$",
    )
    add_variable(
        config,
        name="jet1_eta",
        expression="Jet.eta[:,0]",
        binning=(30, -3.0, 3.0),
        x_title=r"Leading jet $\eta$",
    )
    add_variable(
        config,
        name="jet1_phi",
        expression="Jet.phi[:,0]",
        binning=(66, -3.3, 3.3),
        x_title=r"Leading jet $\phi$",
    )
    add_variable(
        config,
        name="jet2_pt",
        expression="Jet.pt[:,1]",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Subleading jet $p_{T}$",
    )
    add_variable(
        config,
        name="jet2_eta",
        expression="Jet.eta[:,1]",
        binning=(30, -3.0, 3.0),
        x_title=r"Subleading jet $\eta$",
    )
    add_variable(
        config,
        name="jet2_phi",
        expression="Jet.phi[:,1]",
        binning=(66, -3.3, 3.3),
        x_title=r"Subleading jet $\phi$",
    )
    add_variable(
        config,
        name="met_phi",
        expression="PuppiMET.phi",
        binning=(66, -3.3, 3.3),
        x_title=r"MET $\phi$",
    )
    # weights
    add_variable(
        config,
        name="mc_weight",
        expression="mc_weight",
        binning=(200, -10, 10),
        x_title="MC weight",
    )
    add_variable(
        config,
        name="pu_weight",
        expression="pu_weight",
        binning=(40, 0, 2),
        x_title="Pileup weight",
    )
    # add_variable(
    #     config,
    #     name="normalized_pu_weight",
    #     expression="normalized_pu_weight",
    #     binning=(40, 0, 2),
    #     x_title="Normalized pileup weight",
    # )
    # add_variable(
    #     config,
    #     name="btag_weight",
    #     expression="btag_weight",
    #     binning=(60, 0, 3),
    #     x_title="b-tag weight",
    # )
    # add_variable(
    #     config,
    #     name="normalized_btag_weight",
    #     expression="normalized_btag_weight",
    #     binning=(60, 0, 3),
    #     x_title="Normalized b-tag weight",
    # )
    # add_variable(
    #     config,
    #     name="normalized_njet_btag_weight",
    #     expression="normalized_njet_btag_weight",
    #     binning=(60, 0, 3),
    #     x_title="$N_{jet}$ normalized b-tag weight",
    # )
    # cutflow variables
    add_variable(
        config,
        name="cf_njet",
        expression="cutflow.n_jet",
        binning=(17, -0.5, 16.5),
        x_title="Jet multiplicity",
        discrete_x=True,
    )
    add_variable(
        config,
        name="cf_ht",
        expression="cutflow.ht",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$H_{T}$",
    )
    add_variable(
        config,
        name="cf_jet1_pt",
        expression="cutflow.jet1_pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Leading jet $p_{T}$",
    )
    add_variable(
        config,
        name="cf_jet1_eta",
        expression="cutflow.jet1_eta",
        binning=(40, -5.0, 5.0),
        x_title=r"Leading jet $\eta$",
    )
    add_variable(
        config,
        name="cf_jet1_phi",
        expression="cutflow.jet1_phi",
        binning=(66, -3.3, 3.3),
        x_title=r"Leading jet $\phi$",
    )
    add_variable(
        config,
        name="cf_jet2_pt",
        expression="cutflow.jet2_pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Subleading jet $p_{T}$",
    )
    add_variable(
        config,
        name="nbjets_deepjet_medium",
        expression=partial(build_nbjets, config, which="btagDeepFlavB"),
        aux={"inputs": build_nbjets.inputs},
        binning=(11, -0.5, 10.5),
        x_title=r"Number of b-jets (DeepJet medium)",
        discrete_x=True,
    )
    add_variable(
        config,
        name="nbjets_pnet_medium",
        expression=partial(build_nbjets, config, which="btagPNetB"),
        aux={"inputs": build_nbjets.inputs},
        binning=(11, -0.5, 10.5),
        x_title=r"Number of b-jets (PNet medium)",
        discrete_x=True,
    )
    add_variable(
        config,
        name="nbjets_deepjet_loose",
        expression=partial(build_nbjets, config, which="btagDeepFlavB", wp="loose"),
        aux={"inputs": build_nbjets.inputs},
        binning=(11, -0.5, 10.5),
        x_title=r"Number of b-jets (DeepJet loose)",
        discrete_x=True,
    )
    add_variable(
        config,
        name="nbjets_pnet_loose",
        expression=partial(build_nbjets, config, which="btagPNetB", wp="loose"),
        aux={"inputs": build_nbjets.inputs},
        binning=(11, -0.5, 10.5),
        x_title=r"Number of b-jets (PNet loose)",
        discrete_x=True,
    )
    # dilepton variables
    add_variable(
        config,
        name="dilep_energy",
        expression=partial(build_dilep, which="energy"),
        aux={"inputs": build_dilep.inputs},
        binning=(40, 40, 300),
        unit="GeV",
        x_title=r"$E_{ll}$",
    )
    add_variable(
        config,
        name="dilep_mass",
        expression=partial(build_dilep, which="mass"),
        aux={"inputs": build_dilep.inputs},
        binning=(40, 40, 120),
        unit="GeV",
        x_title=r"$m_{ll}$",
    )
    add_variable(
        config,
        name="dilep_pt",
        expression=partial(build_dilep, which="pt"),
        aux={"inputs": build_dilep.inputs},
        binning=(40, 0, 200),
        unit="GeV",
        x_title=r"$p_{T,ll}$",
    )
    add_variable(
        config,
        name="dilep_eta",
        expression=partial(build_dilep, which="eta"),
        aux={"inputs": build_dilep.inputs},
        binning=(50, -5, 5),
        unit="GeV",
        x_title=r"$\eta_{ll}$",
    )
    add_variable(
        config,
        name="dilep_phi",
        expression=partial(build_dilep, which="phi"),
        aux={"inputs": build_dilep.inputs},
        binning=(66, -3.3, 3.3),
        unit="GeV",
        x_title=r"$\phi_{ll}$",
    )
    add_variable(
        config,
        name="dilep_dr",
        expression=partial(build_dilep, which="dr"),
        aux={"inputs": build_dilep.inputs},
        binning=(30, 0, 6),
        x_title=r"$\Delta R_{ll}$",
    )
    # single lepton variables
    # single electron
    add_variable(
        config,
        name="e1_pt",
        expression="Electron.pt[:, 0]",
        binning=(30, 0, 150),
        x_title=r"Leading electron $p_{T}$",
    )
    add_variable(
        config,
        name="e2_pt",
        expression="Electron.pt[:,1]",
        binning=(30, 0, 150),
        x_title=r"Subleading electron $p_{T}$",
    )
    add_variable(
        config,
        name="e1_eta",
        expression="Electron.eta[:,0]",
        binning=(50, -2.5, 2.5),
        x_title=r"Leading electron $\eta$",
    )
    add_variable(
        config,
        name="e2_eta",
        expression="Electron.eta[:,1]",
        binning=(50, -2.5, 2.5),
        x_title=r"Subleading electron $\eta$",
    )
    add_variable(
        config,
        name="e1_phi",
        expression="Electron.phi[:,0]",
        binning=(66, -3.3, 3.3),
        x_title=r"Leading electron $\phi$",
    )
    add_variable(
        config,
        name="e2_phi",
        expression="Electron.phi[:,1]",
        binning=(66, -3.3, 3.3),
        x_title=r"Subleading electron $\phi$",
    )
    # single tau
    add_variable(
        config,
        name="tau1_pt",
        expression="Tau.pt[:, 0]",
        binning=(30, 0, 150),
        x_title=r"Leading tau p$_{T}$",
    )
    add_variable(
        config,
        name="tau2_pt",
        expression="Tau.pt[:,1]",
        binning=(30, 0, 150),
        x_title=r"Subleading tau $p_{T}$",
    )
    add_variable(
        config,
        name="tau1_eta",
        expression="Tau.eta[:,0]",
        binning=(50, -2.5, 2.5),
        x_title=r"Leading tau $\eta$",
    )
    add_variable(
        config,
        name="tau2_eta",
        expression="Tau.eta[:,1]",
        binning=(50, -2.5, 2.5),
        x_title=r"Subleading tau $\eta$",
    )
    add_variable(
        config,
        name="tau1_phi",
        expression="Tau.phi[:,0]",
        binning=(66, -3.3, 3.3),
        x_title=r"Leading tau $\phi$",
    )
    add_variable(
        config,
        name="tau2_phi",
        expression="Tau.phi[:,1]",
        binning=(66, -3.3, 3.3),
        x_title=r"Subleading tau $\phi$",
    )
    # single mu
    add_variable(
        config,
        name="mu1_pt",
        expression="Muon.pt[:,0]",
        binning=(30, 0, 150),
        x_title=r"Leading muon $p_{T}$",
    )
    add_variable(
        config,
        name="mu2_pt",
        expression="Muon.pt[:,1]",
        binning=(30, 0, 150),
        x_title=r"Subleading muon $p_{T}$",
    )
    add_variable(
        config,
        name="mu1_eta",
        expression="Muon.eta[:,0]",
        binning=(50, -2.5, 2.5),
        x_title=r"Leading muon $\eta$",
    )
    add_variable(
        config,
        name="mu2_eta",
        expression="Muon.eta[:,1]",
        binning=(50, -2.5, 2.5),
        x_title=r"Subleading muon $\eta$",
    )
    add_variable(
        config,
        name="mu1_phi",
        expression="Muon.phi[:,0]",
        binning=(66, -3.3, 3.3),
        x_title=r"Leading muon $\phi$",
    )
    add_variable(
        config,
        name="mu2_phi",
        expression="Muon.phi[:,1]",
        binning=(66, -3.3, 3.3),
        x_title=r"Subleading muon $\phi$",
    )
    add_variable(
        config,
        name="njets",
        expression=lambda events: ak.num(events.Jet["pt"], axis=1),
        aux={"inputs": {"Jet.pt"}},
        binning=(11, -0.5, 10.5),
        x_title=r"Number of jets",
    )
    add_variable(
        config,
        name="nmu",
        expression=lambda events: ak.num(events.Muon["pt"], axis=1),
        aux={"inputs": {"Muon.pt"}},
        binning=(11, -0.5, 10.5),
        x_title=r"Number of muons",
    )
    add_variable(
        config,
        name="nlep",
        expression=lambda events: ak.num((ak.concatenate([events.Electron["pt"] * 1, events.Muon["pt"] * 1], axis=1)[:, :]), axis=1),  # noqa: E501
        aux={"inputs": {"{Electron,Muon}.pt"}},
        binning=(11, -0.5, 10.5),
        x_title=r"Number of leptons",
    )
    add_variable(
        config,
        name="m4l",
        expression=partial(build_m4l),
        aux={"inputs": build_m4l.inputs},
        binning=[0, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 380, 400, 450],
        unit="GeV",
        x_title="$m_{4\ell}$",  # noqa: W605
    )
    add_variable(
        config,
        name="nele",
        expression=lambda events: ak.num(events.Electron["pt"], axis=1),
        aux={"inputs": {"Electron.pt"}},
        binning=(11, -0.5, 10.5),
        x_title=r"Number of electrons",
    )
    add_variable(
        config,
        name="ntau",
        expression=lambda events: ak.num(events.Tau["pt"], axis=1),
        aux={"inputs": {"Tau.pt"}},
        binning=(11, -0.5, 10.5),
        x_title=r"Number of electrons",
    )

    for proc in ["hh", "tt", "dy"]:
        # outputs of the resonant pDNN at SM-like mass and spin values
        add_variable(
            config,
            name=f"res_pdnn_{proc}",
            expression=f"res_pdnn_s0_m500_{proc}",
            binning=(25, 0.0, 1.0),
            x_title=rf"{proc.upper()} output node, res. pDNN$_{{m_{{HH}}=500\,GeV,s=0}}$",
        )
        # outputs of the resonant DNN trained over flat masses
        add_variable(
            config,
            name=f"res_dnn_{proc}",
            expression=f"res_dnn_{proc}",
            binning=(25, 0.0, 1.0),
            x_title=rf"{proc.upper()} output node, res. DNN",
        )
        add_variable(
            config,
            name=f"res_dnn_{proc}_fine",
            expression=f"res_dnn_{proc}",
            binning=(5000, 0.0, 1.0),
            x_title=rf"{proc.upper()} output node, res. DNN",
        )


# helper to add a variable to the config with some defaults
def add_variable(config: od.Config, *args, **kwargs) -> od.Variable:
    kwargs.setdefault("null_value", EMPTY_FLOAT)
    variable = config.add_variable(*args, **kwargs)
    if not variable.has_aux("underflow"):
        variable.x.underflow = True
    if not variable.has_aux("overflow"):
        variable.x.overflow = True
    return variable

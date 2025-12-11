# coding: utf-8

"""
HH -> multi-leptons selection methods.
"""

from columnflow.categorization import Categorizer, categorizer
from columnflow.util import maybe_import

ak = maybe_import("awkward")


@categorizer(uses={"event"})
def cat_all(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # keep all events
    return events, ak.ones_like(events.event) == 1

#
# di-lepton channels
#


@categorizer(uses={"channel_id"})
def cat_etau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.cetau.id


@categorizer(uses={"channel_id"})
def cat_mutau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.cmutau.id


@categorizer(uses={"channel_id"})
def cat_tautau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.ctautau.id


@categorizer(uses={"channel_id"})
def cat_ee(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.cee.id


@categorizer(uses={"channel_id"})
def cat_mumu(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.cmumu.id


@categorizer(uses={"channel_id"})
def cat_emu(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.cemu.id


# multilepton channels
@categorizer(uses={"channel_id"})
def cat_3e(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.c3e.id


@categorizer(uses={"channel_id"})
def cat_2emu(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.c2emu.id


@categorizer(uses={"channel_id"})
def cat_e2mu(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.ce2mu.id


@categorizer(uses={"channel_id"})
def cat_3mu(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.c3mu.id


@categorizer(uses={"channel_id"})
def cat_4e(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.c4e.id


@categorizer(uses={"channel_id"})
def cat_3emu(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.c3emu.id


@categorizer(uses={"channel_id"})
def cat_2e2mu(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.c2e2mu.id


@categorizer(uses={"channel_id"})
def cat_e3mu(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.ce3mu.id


@categorizer(uses={"channel_id"})
def cat_4mu(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.c4mu.id


@categorizer(uses={"channel_id"})
def cat_3etau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.c3etau.id


@categorizer(uses={"channel_id"})
def cat_2emutau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.c2emutau.id


@categorizer(uses={"channel_id"})
def cat_e2mutau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.ce2mutau.id


@categorizer(uses={"channel_id"})
def cat_3mutau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.c3mutau.id


@categorizer(uses={"channel_id"})
def cat_2e2tau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.c2e2tau.id


@categorizer(uses={"channel_id"})
def cat_emu2tau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.cemu2tau.id


@categorizer(uses={"channel_id"})
def cat_2mu2tau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.c2mu2tau.id


@categorizer(uses={"channel_id"})
def cat_e3tau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.ce3tau.id


@categorizer(uses={"channel_id"})
def cat_mu3tau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.cmu3tau.id


@categorizer(uses={"channel_id"})
def cat_4tau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.c4tau.id


@categorizer(uses={"channel_id"})
def cat_c2e0or1tau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.c2e0or1tau.id


@categorizer(uses={"channel_id"})
def cat_cemu0or1tau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.cemu0or1tau.id


@categorizer(uses={"channel_id"})
def cat_c2mu0or1tau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.c2mu0or1tau.id


# 3l/4l inclusive, later split into CR / SR via Z-peak
@categorizer(uses={"channel_id", "Jet.btagPNetB", "tight_sel", "Electron.charge", "Muon.charge", "leptons_os"})
def cat_3l0tau_SR(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    catmask = events.channel_id == self.config_inst.channels.n.c3e.id
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c3mu.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c2emu.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.ce2mu.id)
    wp_loose = self.config_inst.x.btag_working_points["particleNet"]["loose"]
    wp_medium = self.config_inst.x.btag_working_points["particleNet"]["medium"]
    tagged_loose = events.Jet.btagPNetB > wp_loose
    tagged_medium = events.Jet.btagPNetB > wp_medium
    bveto = (ak.sum(tagged_loose, axis=1) < 2) & (ak.sum(tagged_medium, axis=1) < 1)
    SR = events.tight_sel == 1
    chargeok = events.leptons_os == 1
    return events, (catmask & bveto & SR & chargeok)


@categorizer(uses={"channel_id", "Jet.btagPNetB", "tight_sel", "Electron.charge", "Muon.charge", "leptons_os"})
def cat_3l0tau_SB(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    catmask = events.channel_id == self.config_inst.channels.n.c3e.id
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c3mu.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c2emu.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.ce2mu.id)
    wp_loose = self.config_inst.x.btag_working_points["particleNet"]["loose"]
    wp_medium = self.config_inst.x.btag_working_points["particleNet"]["medium"]
    tagged_loose = events.Jet.btagPNetB > wp_loose
    tagged_medium = events.Jet.btagPNetB > wp_medium
    bveto = (ak.sum(tagged_loose, axis=1) < 2) & (ak.sum(tagged_medium, axis=1) < 1)
    SB = events.tight_sel == 0
    chargeok = events.leptons_os == 1
    return events, (catmask & bveto & SB & chargeok)


@categorizer(uses={"channel_id"})
def cat_4l(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    catmask = events.channel_id == self.config_inst.channels.n.c4e.id
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c3emu.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c2e2mu.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.ce3mu.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c4mu.id)
    return events, catmask


@categorizer(uses={"channel_id", "Jet.btagPNetB", "tight_sel", "Electron.charge", "Muon.charge", "leptons_os"})
def cat_4l_SR(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    catmask = events.channel_id == self.config_inst.channels.n.c4e.id
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c3emu.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c2e2mu.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.ce3mu.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c4mu.id)
    wp_loose = self.config_inst.x.btag_working_points["particleNet"]["loose"]
    wp_medium = self.config_inst.x.btag_working_points["particleNet"]["medium"]
    tagged_loose = events.Jet.btagPNetB > wp_loose
    tagged_medium = events.Jet.btagPNetB > wp_medium
    bveto = (ak.sum(tagged_loose, axis=1) < 2) & (ak.sum(tagged_medium, axis=1) < 1)
    SR = events.tight_sel == 1
    chargeok = events.leptons_os == 1
    return events, (catmask & bveto & SR & chargeok)


@categorizer(uses={"channel_id", "Jet.btagPNetB", "tight_sel", "Electron.charge", "Muon.charge", "leptons_os"})
def cat_4l_SB(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    catmask = events.channel_id == self.config_inst.channels.n.c4e.id
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c3emu.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c2e2mu.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.ce3mu.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c4mu.id)
    wp_loose = self.config_inst.x.btag_working_points["particleNet"]["loose"]
    wp_medium = self.config_inst.x.btag_working_points["particleNet"]["medium"]
    tagged_loose = events.Jet.btagPNetB > wp_loose
    tagged_medium = events.Jet.btagPNetB > wp_medium
    bveto = (ak.sum(tagged_loose, axis=1) < 2) & (ak.sum(tagged_medium, axis=1) < 1)
    SB = events.tight_sel == 0
    chargeok = events.leptons_os == 1
    return events, (catmask & bveto & SB & chargeok)


@categorizer(uses={"channel_id", "Jet.btagPNetB", "tight_sel", "Electron.charge", "Muon.charge",
    "Tau.charge", "leptons_os"})
def cat_3l1tau_SR(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    catmask = events.channel_id == self.config_inst.channels.n.c3etau.id
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c2emutau.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.ce2mutau.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c3mutau.id)
    wp_loose = self.config_inst.x.btag_working_points["particleNet"]["loose"]
    wp_medium = self.config_inst.x.btag_working_points["particleNet"]["medium"]
    tagged_loose = events.Jet.btagPNetB > wp_loose
    tagged_medium = events.Jet.btagPNetB > wp_medium
    bveto = (ak.sum(tagged_loose, axis=1) < 2) & (ak.sum(tagged_medium, axis=1) < 1)
    SR = events.tight_sel == 1
    chargeok = events.leptons_os == 1
    return events, (catmask & bveto & SR & chargeok)


@categorizer(uses={"channel_id", "Jet.btagPNetB", "tight_sel", "Electron.charge", "Muon.charge",
    "Tau.charge", "leptons_os"})
def cat_3l1tau_SB(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    catmask = events.channel_id == self.config_inst.channels.n.c3etau.id
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c2emutau.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.ce2mutau.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c3mutau.id)
    wp_loose = self.config_inst.x.btag_working_points["particleNet"]["loose"]
    wp_medium = self.config_inst.x.btag_working_points["particleNet"]["medium"]
    tagged_loose = events.Jet.btagPNetB > wp_loose
    tagged_medium = events.Jet.btagPNetB > wp_medium
    bveto = (ak.sum(tagged_loose, axis=1) < 2) & (ak.sum(tagged_medium, axis=1) < 1)
    SB = events.tight_sel == 0
    chargeok = events.leptons_os == 1
    return events, (catmask & bveto & SB & chargeok)


@categorizer(uses={"channel_id", "Jet.btagPNetB", "tight_sel", "Electron.charge", "Muon.charge",
    "Tau.charge", "leptons_os"})
def cat_2l2tau_SR(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    catmask = events.channel_id == self.config_inst.channels.n.c2e2tau.id
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.cemu2tau.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c2mu2tau.id)
    wp_loose = self.config_inst.x.btag_working_points["particleNet"]["loose"]
    wp_medium = self.config_inst.x.btag_working_points["particleNet"]["medium"]
    tagged_loose = events.Jet.btagPNetB > wp_loose
    tagged_medium = events.Jet.btagPNetB > wp_medium
    bveto = (ak.sum(tagged_loose, axis=1) < 2) & (ak.sum(tagged_medium, axis=1) < 1)
    SR = events.tight_sel == 1
    chargeok = events.leptons_os == 1
    return events, (catmask & bveto & SR & chargeok)


@categorizer(uses={"channel_id", "Jet.btagPNetB", "tight_sel", "Electron.charge", "Muon.charge",
    "Tau.charge", "leptons_os"})
def cat_2l2tau_SB(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    catmask = events.channel_id == self.config_inst.channels.n.c2e2tau.id
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.cemu2tau.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c2mu2tau.id)
    wp_loose = self.config_inst.x.btag_working_points["particleNet"]["loose"]
    wp_medium = self.config_inst.x.btag_working_points["particleNet"]["medium"]
    tagged_loose = events.Jet.btagPNetB > wp_loose
    tagged_medium = events.Jet.btagPNetB > wp_medium
    bveto = (ak.sum(tagged_loose, axis=1) < 2) & (ak.sum(tagged_medium, axis=1) < 1)
    SB = events.tight_sel == 0
    chargeok = events.leptons_os == 1
    return events, (catmask & bveto & SB & chargeok)


@categorizer(uses={"channel_id", "Jet.btagPNetB", "tight_sel", "Electron.charge", "Muon.charge",
    "Tau.charge", "leptons_os"})
def cat_1l3tau_SR(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    catmask = events.channel_id == self.config_inst.channels.n.ce3tau.id
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.cmu3tau.id)
    wp_loose = self.config_inst.x.btag_working_points["particleNet"]["loose"]
    wp_medium = self.config_inst.x.btag_working_points["particleNet"]["medium"]
    tagged_loose = events.Jet.btagPNetB > wp_loose
    tagged_medium = events.Jet.btagPNetB > wp_medium
    bveto = (ak.sum(tagged_loose, axis=1) < 2) & (ak.sum(tagged_medium, axis=1) < 1)
    SR = events.tight_sel == 1
    chargeok = events.leptons_os == 1
    return events, (catmask & bveto & SR & chargeok)


@categorizer(uses={"channel_id", "Jet.btagPNetB", "tight_sel", "Electron.charge",
    "Muon.charge", "Tau.charge", "leptons_os"})
def cat_1l3tau_SB(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    catmask = events.channel_id == self.config_inst.channels.n.ce3tau.id
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.cmu3tau.id)
    wp_loose = self.config_inst.x.btag_working_points["particleNet"]["loose"]
    wp_medium = self.config_inst.x.btag_working_points["particleNet"]["medium"]
    tagged_loose = events.Jet.btagPNetB > wp_loose
    tagged_medium = events.Jet.btagPNetB > wp_medium
    bveto = (ak.sum(tagged_loose, axis=1) < 2) & (ak.sum(tagged_medium, axis=1) < 1)
    SB = events.tight_sel == 0
    chargeok = events.leptons_os == 1
    return events, (catmask & bveto & SB & chargeok)


@categorizer(uses={"channel_id", "Jet.btagPNetB", "tight_sel", "Tau.charge", "leptons_os"})
def cat_4tau_SR(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    catmask = events.channel_id == self.config_inst.channels.n.c4tau.id
    wp_loose = self.config_inst.x.btag_working_points["particleNet"]["loose"]
    wp_medium = self.config_inst.x.btag_working_points["particleNet"]["medium"]
    tagged_loose = events.Jet.btagPNetB > wp_loose
    tagged_medium = events.Jet.btagPNetB > wp_medium
    bveto = (ak.sum(tagged_loose, axis=1) < 2) & (ak.sum(tagged_medium, axis=1) < 1)
    SR = events.tight_sel == 1
    chargeok = events.leptons_os == 1
    return events, (catmask & bveto & SR & chargeok)


@categorizer(uses={"channel_id", "Jet.btagPNetB", "tight_sel", "Tau.charge", "leptons_os"})
def cat_4tau_SB(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    catmask = events.channel_id == self.config_inst.channels.n.c4tau.id
    wp_loose = self.config_inst.x.btag_working_points["particleNet"]["loose"]
    wp_medium = self.config_inst.x.btag_working_points["particleNet"]["medium"]
    tagged_loose = events.Jet.btagPNetB > wp_loose
    tagged_medium = events.Jet.btagPNetB > wp_medium
    bveto = (ak.sum(tagged_loose, axis=1) < 2) & (ak.sum(tagged_medium, axis=1) < 1)
    SB = events.tight_sel == 0
    chargeok = events.leptons_os == 1
    return events, (catmask & bveto & SB & chargeok)


@categorizer(uses={"channel_id", "Jet.btagPNetB", "tight_sel", "Electron.charge", "Muon.charge", "leptons_os"})
def cat_2l0or1tau_SR_SS(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    catmask = events.channel_id == self.config_inst.channels.n.c2e0or1tau.id
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.cemu0or1tau.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c2mu0or1tau.id)
    wp_loose = self.config_inst.x.btag_working_points["particleNet"]["loose"]
    wp_medium = self.config_inst.x.btag_working_points["particleNet"]["medium"]
    tagged_loose = events.Jet.btagPNetB > wp_loose
    tagged_medium = events.Jet.btagPNetB > wp_medium
    bveto = (ak.sum(tagged_loose, axis=1) < 2) & (ak.sum(tagged_medium, axis=1) < 1)
    SR = events.tight_sel == 1
    SS = events.leptons_os == 0
    return events, (catmask & bveto & SR & SS)


@categorizer(uses={"channel_id", "Jet.btagPNetB", "tight_sel", "Electron.charge", "Muon.charge", "leptons_os"})
def cat_2l0or1tau_SR_OS(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    catmask = events.channel_id == self.config_inst.channels.n.c2e0or1tau.id
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.cemu0or1tau.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c2mu0or1tau.id)
    wp_loose = self.config_inst.x.btag_working_points["particleNet"]["loose"]
    wp_medium = self.config_inst.x.btag_working_points["particleNet"]["medium"]
    tagged_loose = events.Jet.btagPNetB > wp_loose
    tagged_medium = events.Jet.btagPNetB > wp_medium
    bveto = (ak.sum(tagged_loose, axis=1) < 2) & (ak.sum(tagged_medium, axis=1) < 1)
    SR = events.tight_sel == 1
    OS = events.leptons_os == 1
    return events, (catmask & bveto & SR & OS)


@categorizer(uses={"channel_id", "Jet.btagPNetB", "tight_sel", "Electron.charge", "Muon.charge", "leptons_os"})
def cat_2l0or1tau_SB_SS(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    catmask = events.channel_id == self.config_inst.channels.n.c2e0or1tau.id
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.cemu0or1tau.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c2mu0or1tau.id)
    wp_loose = self.config_inst.x.btag_working_points["particleNet"]["loose"]
    wp_medium = self.config_inst.x.btag_working_points["particleNet"]["medium"]
    tagged_loose = events.Jet.btagPNetB > wp_loose
    tagged_medium = events.Jet.btagPNetB > wp_medium
    bveto = (ak.sum(tagged_loose, axis=1) < 2) & (ak.sum(tagged_medium, axis=1) < 1)
    SB = events.tight_sel == 0
    SS = events.leptons_os == 0
    return events, (catmask & bveto & SB & SS)


@categorizer(uses={"channel_id", "Jet.btagPNetB", "tight_sel", "Electron.charge", "Muon.charge", "leptons_os"})
def cat_2l0or1tau_SB_OS(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    catmask = events.channel_id == self.config_inst.channels.n.c2e0or1tau.id
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.cemu0or1tau.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c2mu0or1tau.id)
    wp_loose = self.config_inst.x.btag_working_points["particleNet"]["loose"]
    wp_medium = self.config_inst.x.btag_working_points["particleNet"]["medium"]
    tagged_loose = events.Jet.btagPNetB > wp_loose
    tagged_medium = events.Jet.btagPNetB > wp_medium
    bveto = (ak.sum(tagged_loose, axis=1) < 2) & (ak.sum(tagged_medium, axis=1) < 1)
    SB = events.tight_sel == 0
    OS = events.leptons_os == 1
    return events, (catmask & bveto & SB & OS)


# bveto
@categorizer(uses={"Jet.btagPNetB"})
def cat_bveto_on(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    wp_loose = self.config_inst.x.btag_working_points["particleNet"]["loose"]
    wp_medium = self.config_inst.x.btag_working_points["particleNet"]["medium"]
    tagged_loose = events.Jet.btagPNetB > wp_loose
    tagged_medium = events.Jet.btagPNetB > wp_medium
    veto = (ak.sum(tagged_loose, axis=1) < 2) & (ak.sum(tagged_medium, axis=1) < 1)
    return events, veto


@categorizer(uses={"Jet.btagPNetB"})
def cat_bveto_off(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    wp_loose = self.config_inst.x.btag_working_points["particleNet"]["loose"]
    wp_medium = self.config_inst.x.btag_working_points["particleNet"]["medium"]
    tagged_loose = events.Jet.btagPNetB > wp_loose
    tagged_medium = events.Jet.btagPNetB > wp_medium
    nonveto = (ak.sum(tagged_loose, axis=1) >= 2) | (ak.sum(tagged_medium, axis=1) >= 1)
    return events, nonveto


# The BDT category overlaps with our channels, so we need tight/trigger-matched flags individual for this cat
@categorizer(uses={"ok_bdt_eormu"})
def cat_e_or_mu(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.ok_bdt_eormu == 1


@categorizer(uses={"ok_bdt_eormu_bveto", "Jet.btagPNetB"})
def cat_e_or_mu_bveto(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    wp_loose = self.config_inst.x.btag_working_points["particleNet"]["loose"]
    wp_medium = self.config_inst.x.btag_working_points["particleNet"]["medium"]
    tagged_loose = events.Jet.btagPNetB > wp_loose
    tagged_medium = events.Jet.btagPNetB > wp_medium
    veto = (ak.sum(tagged_loose, axis=1) < 2) & (ak.sum(tagged_medium, axis=1) < 1)
    return events, ((events.ok_bdt_eormu_bveto == 1) & veto)


@categorizer(uses={"tight_sel_bdt"})
def cat_tight_bdt(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # tight true
    return events, events.tight_sel_bdt == 1


@categorizer(uses={"tight_sel_bdt"})
def cat_nontight_bdt(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # tight false
    return events, events.tight_sel_bdt == 0


@categorizer(uses={"trig_match_bdt"})
def cat_trigmatch_bdt(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # trig match
    return events, events.trig_match_bdt == 1


@categorizer(uses={"trig_match_bdt"})
def cat_nontrigmatch_bdt(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # trig match false
    return events, events.trig_match_bdt == 0


# Tight and trigger matching flags for the physical channels
@categorizer(uses={"tight_sel"})
def cat_tight(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # tight true
    return events, events.tight_sel == 1


@categorizer(uses={"tight_sel"})
def cat_nontight(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # tight false
    return events, events.tight_sel == 0


@categorizer(uses={"trig_match"})
def cat_trigmatch(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # trig match
    return events, events.trig_match == 1


@categorizer(uses={"trig_match"})
def cat_nontrigmatch(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # trig match false
    return events, events.trig_match == 0


# QCD regions
@categorizer(uses={"leptons_os"})
def cat_os(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # oppositive sign leptons
    return events, events.leptons_os == 1


@categorizer(uses={"leptons_os"})
def cat_ss(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # same sign leptons
    return events, events.leptons_os == 0


@categorizer(uses={"tau2_isolated"})
def cat_iso(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # isolated tau2
    return events, events.tau2_isolated == 1


@categorizer(uses={"tau2_isolated"})
def cat_noniso(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # noon-isolated tau2
    return events, events.tau2_isolated == 0


# kinematic regions
@categorizer(uses={"event"})
def cat_incl(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # fully inclusive selection
    return events, ak.ones_like(events.event) == 1


@categorizer(uses={"Jet.{pt,phi}"})
def cat_2j(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # two or more jets
    return events, ak.num(events.Jet.pt, axis=1) >= 2


@categorizer(uses={"Jet.btagPNetB"})
def cat_res1b(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # exactly pnet b-tags
    wp = self.config_inst.x.btag_working_points["particleNet"]["medium"]
    tagged = events.Jet.btagPNetB > wp
    return events, ak.sum(tagged, axis=1) == 1


@categorizer(uses={"Jet.btagPNetB"})
def cat_res2b(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # at least two medium pnet b-tags
    wp = self.config_inst.x.btag_working_points["particleNet"]["medium"]
    tagged = events.Jet.btagPNetB > wp
    return events, ak.sum(tagged, axis=1) >= 2


@categorizer(uses={cat_res1b, cat_res2b, "FatJet.{pt,phi}"})
def cat_boosted(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # not res1b or res2b, and exactly one selected fat jet that should also pass a tighter pt cut
    # note: this is just a draft
    mask = (
        (ak.num(events.FatJet, axis=1) == 1) &
        (ak.sum(events.FatJet.pt > 350, axis=1) == 1) &
        ~self[cat_res1b](events, **kwargs)[1] &
        ~self[cat_res2b](events, **kwargs)[1]
    )
    return events, mask


@categorizer(uses={"{Electron,Muon,Tau}.{pt,eta,phi,mass}"})
def cat_dy(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # e/mu driven DY region: mll > 40 and met < 30 (to supress tau decays into e/mu)
    leps = ak.concatenate([events.Electron * 1, events.Muon * 1, events.Tau * 1], axis=1)[:, :2]
    mask = (
        (leps.sum(axis=1).mass > 40) &
        (events[self.config_inst.x.met_name].pt < 30)
    )
    return events, mask


@cat_dy.init
def cat_dy_init(self: Categorizer) -> None:
    self.uses.add(f"{self.config_inst.x.met_name}.{{pt,phi}}")


@categorizer(uses={"{Electron,Muon,Tau}.{pt,eta,phi,mass}"})
def cat_tt(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # tt region: met > 30 (due to neutrino presence in leptonic w decays)
    mask = events[self.config_inst.x.met_name].pt > 30
    return events, mask


@cat_tt.init
def cat_tt_init(self: Categorizer) -> None:
    self.uses.add(f"{self.config_inst.x.met_name}.{{pt,phi}}")

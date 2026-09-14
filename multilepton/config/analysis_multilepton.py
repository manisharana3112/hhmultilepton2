# coding: utf-8
"""
Configuration of the HH → multileptons analysis.
"""

from __future__ import annotations

import importlib
import order as od

from columnflow.util import DotDict

from multilepton.hist_hooks.blinding import add_hooks as add_blinding_hooks
from multilepton.hist_hooks.binning import add_hooks as add_binning_hooks
from multilepton.config.configs_multilepton import add_config


# =======================================
# Analysis Definition
# =======================================
analysis_multilepton = od.Analysis(name="analysis_multilepton", id=1)

# Use lookup from law.cfg
analysis_multilepton.x.versions = {}

# Bash sandboxes required by remote tasks
analysis_multilepton.x.bash_sandboxes = [
    "$CF_BASE/sandboxes/cf.sh",
    "$MULTILEPTON_BASE/sandboxes/venv_multilepton.sh",
]

# CMSSW sandboxes (optional)
analysis_multilepton.x.cmssw_sandboxes = [
    # "$CF_BASE/sandboxes/cmssw_default.sh",
]

# =======================================
# Analysis-wide Groups and Defaults
# =======================================
analysis_multilepton.x.config_groups = {}
analysis_multilepton.x.store_parts_modifiers = {}

# =======================================
# Histogram Hooks
# =======================================
analysis_multilepton.x.hist_hooks = DotDict()
add_blinding_hooks(analysis_multilepton)
add_binning_hooks(analysis_multilepton)


# =======================================
# Lazy Config Factory Helper
# =======================================
def add_lazy_config(
    *,
    campaign_module: str,
    campaign_attr: str,
    config_name: str,
    config_id: int,
    **kwargs,
) -> None:

    """
    Register a lazily-created configuration into the multilepton analysis.

    File selection (a file count cap, or a specific file range) is not handled here: use the
    task-level ``--limit-dataset-files`` and workflow ``--branches`` parameters on the plain
    config instead, so every variant shares the same cached LFN listing and branch numbers map
    1:1 to real file indices.
    """

    def create_factory(config_id, config_name_postfix=""):
        def factory(configs):
            mod = importlib.import_module(campaign_module)
            campaign = getattr(mod, campaign_attr)
            return add_config(
                analysis_multilepton,
                campaign.copy(),
                config_name=config_name + config_name_postfix,
                config_id=config_id,
                **kwargs,
            )
        return factory

    analysis_multilepton.configs.add_lazy_factory(config_name, create_factory(config_id))


# =======================================
# Dataset Configurations
# =======================================
datasets = [
    # cid = 32024115  => (run)3(year)2024(part)1(nano_version)15
    # --- Private UHH NanoAOD datasets ---
    ("cmsdb.campaigns.run3_2022_preEE_nano_uhh_v14", "22preEE_v14_private", 320221114),
    ("cmsdb.campaigns.run3_2022_postEE_nano_uhh_v14", "22postEE_v14_private", 32022214),
    ("cmsdb.campaigns.run3_2023_preBPix_nano_uhh_v14", "23preBPix_v14_private", 32023114),
    ("cmsdb.campaigns.run3_2023_postBPix_nano_uhh_v14", "23postBPix_v14_private", 32023214),

    # --- Central NanoAOD datasets ---
    ("cmsdb.campaigns.run3_2022_preEE_nano_v12", "22preEE_v12_central", 320221112),
    ("cmsdb.campaigns.run3_2022_postEE_nano_v12", "22postEE_v12_central", 32022212),
    ("cmsdb.campaigns.run3_2023_preBPix_nano_v12", "23preBPix_v12_central", 32023112),
    ("cmsdb.campaigns.run3_2023_postBPix_nano_v12", "23postBPix_v12_central", 32023212),
    ("cmsdb.campaigns.run3_2024_nano_v15", "24_v15_central", 32024115),
]

for module, name, cid in datasets:
    add_lazy_config(
        campaign_module=module,
        campaign_attr=f"campaign_{module.split('.')[-1]}",
        config_name=name,
        config_id=cid,
    )
    # dedicated gen-matching-studies variant of the same campaign: the extra columns/categories
    # are only computed when this config is explicitly selected (--config <name>_genmatch),
    # not on every default run of the plain config above
    add_lazy_config(
        campaign_module=module,
        campaign_attr=f"campaign_{module.split('.')[-1]}",
        config_name=f"{name}_genmatch",
        config_id=cid + 1,
        enable_gen_matching_studies=True,
    )

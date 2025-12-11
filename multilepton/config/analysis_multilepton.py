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
    add_limited: bool = True,
    limit_dataset_files: int | None = None,
    **kwargs,
) -> None:
    """Register a lazily-created configuration into the multilepton analysis."""

    def create_factory(
        config_id: int,
        config_name_postfix: str = "",
        limit_dataset_files_factory: int | None = None,
        # limit_dataset_files: int | None = None,
    ):
        def factory(configs: od.UniqueObjectIndex):
            mod = importlib.import_module(campaign_module)
            campaign = getattr(mod, campaign_attr)
            # limit_dataset_files: int | None = None,
            limit_files = limit_dataset_files_factory or limit_dataset_files
            return add_config(
                analysis_multilepton,
                campaign.copy(),
                config_name=config_name + config_name_postfix,
                config_id=config_id,
                # limit_dataset_files=limit_dataset_files,
                limit_dataset_files=limit_files,
                **kwargs,
            )
        return factory

    # Add full configuration
    analysis_multilepton.configs.add_lazy_factory(config_name, create_factory(config_id))

    # Optionally add a "_limited" version
    if add_limited:
        limited_name = f"{config_name}_limited"
        if limited_name in analysis_multilepton.configs:
            raise ValueError(f"Duplicate config name detected: {limited_name}")
        analysis_multilepton.configs.add_lazy_factory(
            limited_name,
            create_factory(config_id + 200, "_limited", 1),
        )


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
        add_limited=False,
    )

# coding: utf-8

"""
Plot style definitions.
"""

from __future__ import annotations

import law
import order as od

from copy import deepcopy
from collections import defaultdict
from columnflow.util import DotMap, try_int


logger = law.logger.get_logger(__name__)


def resolve_inheritance(styles: dict) -> dict:
    """
    Resolve 'inherit' relationships in the style dictionary.
    Later definitions override parent ones.
    """
    resolved = {}
    for name, cfg in styles.items():
        if "inherit" in cfg:
            parent = cfg.pop("inherit")
            base = deepcopy(resolved.get(parent, styles.get(parent, {})))
            merged = deepcopy(base)
            merged.update(cfg)
            resolved[name] = merged
        else:
            resolved[name] = deepcopy(cfg)
    return resolved


def setup_plot_styles(config: od.Config, yaml_data) -> None:
    """
    Setup plot styles from a YAML configuration file.
    Fallback to hardcoded defaults if YAML is missing.
    """
    # General settings
    general = yaml_data.get("general", {})
    config.x.default_general_settings = {
        "cms_label": general.get("cms_label", "Work-in-progress"),
        "whitespace_fraction": general.get("whitespace_fraction", 0.31),
    }

    # Global defaults
    config.x.default_custom_style_config = yaml_data.get("default_style", "wide_legend")
    config.x.default_blinding_threshold = yaml_data.get("blinding_threshold", 0)

    # ─────────────────────────────
    # Build custom style groups
    # ─────────────────────────────
    style_defs = yaml_data.get("styles", {})
    resolved_styles = resolve_inheritance(style_defs)

    # Optional: add computed or callable parameters
    for name, style in resolved_styles.items():
        legend_cfg = style.get("legend_cfg", {})
        if legend_cfg.get("cf_entries_per_column", "auto") == "auto":
            legend_cfg["cf_entries_per_column"] = legend_entries_per_column
        style["legend_cfg"] = legend_cfg

    config.x.custom_style_config_groups = resolved_styles
    logger.info(f"Loaded {len(resolved_styles)} style configurations from analysis.yaml")
    return


def apply_process_styles(config, process_key, process_data, group=None):
    """Apply individual style info to a process if it exists.
    """
    color = process_data.get("color")
    label = process_data.get("label")
    cmsdb_list = process_data.get("cmsdb", [])

    for dataset in cmsdb_list:
        if (p := config.get_process(dataset, default=None)):
            if color:
                p.color1 = color
            if label:
                p.label = label

            # only build label automatically if none is provided AND group is signal
            if not label and group == "signal":
                name = p.name if hasattr(p, "name") else dataset  # fallback
                if "htt_hvv" in name:
                    decay = r"\tau\tau VV"
                elif "htt_htt" in name:
                    decay = r"\tau\tau\tau\tau"
                elif "hvv_vv" in name:
                    decay = "VVVV"

                # --- GGF signal ---
                if name.startswith("hh_ggf"):
                    kl = name.split("_")[-2].replace("kl", "")
                    kappa_label = create_kappa_label(**{r"\lambda": kl, "t": "1"})
                    p.label = rf"$HH_{{ggf}} \rightarrow {decay}$ __SCALE____SHORT____BREAK__({kappa_label})"

                # --- VBF signal ---
                elif name.startswith("hh_vbf"):
                    parts = {x[:2]: x[2:] for x in name.split("_") if x.startswith(("kv", "k2v", "kl"))}
                    kv = parts.get("kv", "1")
                    k2v = parts.get("k2v", "1")
                    kl = parts.get("kl", "1")
                    kappa_label = create_kappa_label(**{"2V": k2v, r"\lambda": kl, "V": kv})
                    p.label = rf"$HH_{{vbf}} \rightarrow {decay}$ __SCALE____SHORT____BREAK__({kappa_label})"
    return


def stylize_processes(config: od.Config, datasets_cfg: DotMap) -> None:
    """
    Applies style and metadata (colors, labels, etc.)
    to each process from the datasets YAML.
    Recommended cms colors see: https://cms-analysis.docs.cern.ch/guidelines/plotting/colors
    """
    # Loop through signal/background groups
    for group_name, group in datasets_cfg.items():
        if group_name not in ["signal", "background"]:
            continue

        for sub_group_name, sub_group in group.items():
            # Nested levels (e.g. 'nonresonant', 'ggf', 'vbf')
            if isinstance(sub_group, dict) and "cmsdb" not in sub_group:
                for process_name, process in sub_group.items():
                    apply_process_styles(config, process_name, process, group_name)
            else:
                apply_process_styles(config, sub_group_name, sub_group, group_name)
    return


def legend_entries_per_column(ax, handles: list, labels: list, n_cols: int) -> list[int]:
    """
    Control  number of entries such that backgrounds are in the first n - 1 columns, and everything
    else in the last one.
    """
    # get number of background and remaining entries
    n_backgrounds = sum(1 for handle in handles if handle.__class__.__name__ == "StepPatch")
    n_other = len(handles) - n_backgrounds
    # fill number of entries per column
    entries_per_col = n_cols * [0]
    n_bkg_cols = n_cols
    # set last column if non-backgrounds are present
    if n_other:
        entries_per_col[-1] = n_other
        n_bkg_cols -= 1
    # fill background columns
    for i in range(n_bkg_cols):
        entries_per_col[i] = n_backgrounds // n_bkg_cols + (n_backgrounds % n_bkg_cols > i)
    return entries_per_col


def kappa_str_to_num(value: str) -> int | float:
    """
    Converts a string-encoded kappa value to an actual number. An integer is returned if possible,
    and a float otherwise. Examples:
    .. code-block:: python
        kappa_str_to_num("1")     # 1
        kappa_str_to_num("2.45")  # 2.45
        kappa_str_to_num("m1p7")  # -1.7
    """
    value = value.replace("p", ".").replace("m", "-")
    return int(value) if try_int(value) else float(value)


def group_kappas(**kappas: dict[str, str]) -> dict[int | float, list[str]]:
    """
    Groups kappa values by their coupling strength. Examples:
    .. code-block:: python
        group_kappas(kl="1", kt="1")           # {1: ["kl", "kt"]}
        group_kappas(kl="2p45", kt="1")        # {2.45: ["kl"], 1: ["kt"]}
        group_kappas(k2v="0", kv="1", kl="1")  # {0: ["k2v"], 1: ["kv", "kl"]}
    """
    str_groups = defaultdict(list)
    for k, v in kappas.items():
        str_groups[v].append(k)
    # convert keys to numbers
    return {kappa_str_to_num(k): v for k, v in str_groups.items()}


def create_kappa_label(*, sep: str = ",", **kappas: dict[str, str]) -> str:
    parts = []
    for v, _kappas in group_kappas(**kappas).items():
        k_str = "=".join(rf"\kappa_{{{k}}}"for k in _kappas)
        parts.append(f"{k_str}={v}")
    return "$" + sep.join(parts) + "$"

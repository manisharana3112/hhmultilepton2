# coding: utf-8

"""
Definition of categories.
"""

import functools

import order as od

from columnflow.config_util import add_category, create_category_combinations


multileptons_categories = {
    # 3l/4l inclusive
    "cat3l0tau_SR": {"id": 1001, "selection": "cat_3l0tau_SR", "label": r"$3\ell 0\tau_{h}$ SR"},
    "cat3l0tau_SB": {"id": 1002, "selection": "cat_3l0tau_SB", "label": r"$3\ell 0\tau_{h}$ SB"},
    "cat4l_SR": {"id": 1003, "selection": "cat_4l_SR", "label": r"$4\ell$ SR"},
    "cat4l_SB": {"id": 1004, "selection": "cat_4l_SB", "label": r"$4\ell$ SB"},
    "cat3l1tau_SR": {"id": 1005, "selection": "cat_3l1tau_SR", "label": r"$3\ell 1\tau_{h}$ SR"},
    "cat3l1tau_SB": {"id": 1006, "selection": "cat_3l1tau_SB", "label": r"$3\ell 1\tau_{h}$ SB"},
    "cat2l2tau_SR": {"id": 1007, "selection": "cat_2l2tau_SR", "label": r"$2\ell 2\tau_{h}$ SR"},
    "cat2l2tau_SB": {"id": 1008, "selection": "cat_2l2tau_SB", "label": r"$2\ell 2\tau_{h}$ SB"},
    "cat1l3tau_SR": {"id": 1009, "selection": "cat_1l3tau_SR", "label": r"$1\ell 3\tau_{h}$ SR"},
    "cat1l3tau_SB": {"id": 1010, "selection": "cat_1l3tau_SB", "label": r"$1\ell 3\tau_{h}$ SB"},
    "cat4tau_SR": {"id": 1011, "selection": "cat_4tau_SR", "label": r"$4\tau_{h}$ SR"},
    "cat4tau_SB": {"id": 1012, "selection": "cat_4tau_SB", "label": r"$4\tau_{h}$ SB"},
    "cat2lSS1tauOS_SR": {"id": 1013, "selection": "cat_2lSS1tauOS_SR", "label": r"$2\ell SS\,  1\tau_{h}$ SR"},
    "cat2lOS1tauSS_SR": {"id": 1014, "selection": "cat_2lOS1tauSS_SR", "label": r"$2\ell OS\,  1\tau_{h}$ SR"},
    "cat2lSS1tauOS_SB": {"id": 1015, "selection": "cat_2lSS1tauOS_SB", "label": r"$2\ell SS\,  1\tau_{h}$ SB"},
    "cat2lOS1tauSS_SB": {"id": 1016, "selection": "cat_2lOS1tauSS_SB", "label": r"$2\ell OS\,  1\tau_{h}$ SB"},
    "cat2lSS0tauOS_SR": {"id": 1017, "selection": "cat_2lSS_SR", "label": r"$2\ell SS\,  0\tau_{h}$ SR"},
    "cat2lOS0tauSS_SR": {"id": 1018, "selection": "cat_2lOS_SR", "label": r"$2\ell OS\,  0\tau_{h}$ SR"},
    "cat2lSS0tauOS_SB": {"id": 1019, "selection": "cat_2lSS_SB", "label": r"$2\ell SS\,  0\tau_{h}$ SB"},
    "cat2lOS0tauSS_SB": {"id": 1020, "selection": "cat_2lOS_SB", "label": r"$2\ell OS\,  0\tau_{h}$ SB"},
    "cat1l2tau_SR": {"id": 1021, "selection": "cat_1l2tau_SR", "label": r"$1\ell 2\tau_{h}$ SR"},
    "cat1l2tau_SB": {"id": 1022, "selection": "cat_1l2tau_SB", "label": r"$1\ell 2\tau_{h}$ SB"},
    # fake factor measurement regions, only filled by the default_ffmr selector
    "catttbarMR": {"id": 1023, "selection": "cat_ttbarMR", "label": r"$t\bar{t}$ MR"},
    "catwzMR": {"id": 1024, "selection": "cat_wzMR", "label": r"$WZ$ MR"},
    "catdyMR": {"id": 1025, "selection": "cat_dyMR", "label": r"$DY$ MR"},
    # Loose category for BDT trainning + tight + trigmatch
    "ceormu_bveto": {"id": 15000, "selection": "cat_eormu_bveto", "label": r"e or $\mu$ bveto on", "tags": {"ceormu_bveto"}},  # noqa: E501
    # bveto
    "bveto_on": {"id": 30001, "selection": "cat_bveto_on", "label": "bveto on"},
    # gen-matching categories (fakes/nonfakes/conversions/flips)
    "gen_nonfakes": {"id": 20001, "selection": "cat_nonfakes", "label": "non-fakes", "tags": {"gen_nonfakes"}},
    "gen_fakes": {"id": 20002, "selection": "cat_fakes", "label": "fakes", "tags": {"gen_fakes"}},
    "gen_conversions": {"id": 20003, "selection": "cat_conversions", "label": "conversions", "tags": {"gen_conversions"}},  # noqa: E501
    "gen_flips": {"id": 20004, "selection": "cat_flips", "label": "flips", "tags": {"gen_flips"}},
    # tight/nontight
    "tight_bdt": {"id": 11000, "selection": "cat_tight_bdt", "label": "tight", "tags": {"tight_bdt"}},
    "trigmatch_bdt": {"id": 13000, "selection": "cat_trigmatch_bdt", "label": "trigger matched", "tags": {"trigmatch_bdt"}},  # noqa: E501
    "tight": {"id": 10001, "selection": "cat_tight", "label": "tight", "tags": {"tight"}},
    "trigmatch": {"id": 10003, "selection": "cat_trigmatch", "label": "trigger matched", "tags": {"trigmatch"}},
    "os": {"id": 10, "selection": "cat_os", "label": "OS", "tags": {"os"}},
    "ss": {"id": 11, "selection": "cat_ss", "label": "SS", "tags": {"ss"}},
    "iso": {"id": 12, "selection": "cat_iso", "label": r"iso", "tags": {"iso"}},
    "noniso": {"id": 13, "selection": "cat_noniso", "label": r"non-iso", "tags": {"noniso"}},
    "incl": {"id": 100, "selection": "cat_incl", "label": "inclusive"},
    "tt": {"id": 220, "selection": "cat_tt", "label": r"$t\bar{t}$ enriched"},
    "res2b": {"id": 301, "selection": "cat_res2b", "label": "res2b"},
}


def add_categories(config: od.Config) -> None:
    """
    Adds all categories to a *config*.
    """
    # root category (-1 has special meaning in cutflow)
    root_cat = add_category(config, name="all", id=-1, selection="cat_all", label="")
    _add_category = functools.partial(add_category, parent=root_cat)

    # one category per existing channel
    for ch in config.channels:
        _add_category(config, name=ch.name, id=ch.id, selection=f"cat_{ch.name[1:]}", label=ch.label, tags=ch.name)

    # opt-in only: the gen-matching classification and its categories are only needed for
    # dedicated gen-matching/fake studies (see enable_gen_matching_studies in configs_multilepton.py)
    gen_matching_enabled = config.x("enable_gen_matching_studies", False)
    gen_match_category_names = {"gen_nonfakes", "gen_fakes", "gen_conversions", "gen_flips"}

    # analysis-specific multilepton categories
    for name, cat in multileptons_categories.items():
        if name in gen_match_category_names and not gen_matching_enabled:
            continue
        _add_category(
            config,
            name=name,
            id=cat["id"],
            selection=cat["selection"],
            label=cat["label"],
            tags=cat.get("tags"),
        )

    if not gen_matching_enabled:
        return

    # ------------------------------------------------------------------
    # combine SR regions with gen-match categories
    # ------------------------------------------------------------------
    regions = [
        "cat2lSS0tauOS_SR", "cat2lOS0tauSS_SR", "cat2lSS1tauOS_SR", "cat2lOS1tauSS_SR",
        "cat1l2tau_SR", "cat3l0tau_SR", "cat4l_SR", "cat3l1tau_SR",
        "cat2l2tau_SR", "cat1l3tau_SR", "cat4tau_SR",
    ]
    gen_matches = ["gen_nonfakes", "gen_fakes", "gen_conversions", "gen_flips"]

    region_base_id = {
        "cat2lSS0tauOS_SR": 21000,
        "cat2lOS0tauSS_SR": 22000,
        "cat2lSS1tauOS_SR": 23000,
        "cat2lOS1tauSS_SR": 24000,
        "cat1l2tau_SR": 25000,
        "cat3l0tau_SR": 26000,
        "cat4l_SR": 27000,
        "cat3l1tau_SR": 28000,
        "cat2l2tau_SR": 29000,
        "cat1l3tau_SR": 31000,
        "cat4tau_SR": 32000,
    }
    gen_match_offset = {
        "gen_nonfakes": 1,
        "gen_fakes": 2,
        "gen_conversions": 3,
        "gen_flips": 4,
    }

    def name_fn(root_cats):
        return "_".join(cat.name for cat in root_cats.values())

    def kwargs_fn(root_cats):
        region_name = root_cats["region"].name
        gen_match_name = root_cats["gen_match"].name
        cat_id = region_base_id[region_name] + gen_match_offset[gen_match_name]
        return {
            "id": cat_id,
            "label": ", ".join(cat.label for cat in root_cats.values()),
        }

    create_category_combinations(
        config,
        {
            "region": [config.get_category(n) for n in regions],
            "gen_match": [config.get_category(n) for n in gen_matches],
        },
        name_fn=name_fn,
        kwargs_fn=kwargs_fn,
    )

    config.x.category_groups = {
        **{region: [f"{region}_{gm}" for gm in gen_matches] for region in regions},
        **{gm: [f"{region}_{gm}" for region in regions] for gm in gen_matches},
    }

# def add_categories(config: od.Config) -> None:
#     """
#     Adds all categories to a *config*.
#     """
#     # root category (-1 has special meaning in cutflow)
#     root_cat = add_category(config, name="all", id=-1, selection="cat_all", label="")
#     _add_category = functools.partial(add_category, parent=root_cat)

#     # One category per existing channel
#     for ch in config.channels:
#         _add_category(config, name=ch.name, id=ch.id, selection=f"cat_{ch.name[1:]}", label=ch.label, tags=ch.name)
#     # Analysis-specific multilepton categories
#     for name, cat in multileptons_categories.items():
#         _add_category(
#             config,
#             name=name,
#             id=cat["id"],
#             selection=cat["selection"],
#             label=cat["label"],
#             tags=cat.get("tags"),
#         )

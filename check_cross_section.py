# coding: utf-8

"""
Search cmsdb.processes for process objects by name and print their cross sections.

Why this exists: guessing the exact import path (e.g. `from cmsdb.processes.hh import
hh_ggf_htt_hvv_kl1_kt1_powheg`) is error-prone since module/variable names don't always
match dataset names exactly. This script walks all submodules under cmsdb.processes,
collects every `order.Process` instance it finds, and lets you search by substring --
so you don't need to know which file or exact variable name a process lives under.

Usage:
    # list all processes whose name contains "hh_ggf" or "dy_m50"
    python cmsdb_xsec_lookup.py --search hh_ggf dy_m50 tt_dl wz --ecm 13.6

    # just list everything under a keyword, without printing xsecs (to eyeball naming)
    python cmsdb_xsec_lookup.py --search hh --list-only

    # get exact match only (no substring) if you already know the exact name
    python cmsdb_xsec_lookup.py --search hh_ggf_htt_hvv_kl1_kt1_powheg --exact --ecm 13.6
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys


def discover_processes():
    """
    Walk every submodule under cmsdb.processes and collect (module_name, attr_name, proc)
    for every attribute that looks like an order.Process instance.
    """
    try:
        import cmsdb.processes as cmsdb_processes
        import order as od
    except ImportError as e:
        sys.exit(f"import failed: {e}\n(activate the columnflow/cmsdb env first)")

    found = {}  # name -> (module_name, attr_name, proc)

    for _, module_name, _ in pkgutil.iter_modules(cmsdb_processes.__path__, prefix="cmsdb.processes."):
        try:
            mod = importlib.import_module(module_name)
        except Exception as e:
            print(f"  [warn] could not import {module_name}: {e}", file=sys.stderr)
            continue

        for attr_name in dir(mod):
            if attr_name.startswith("_"):
                continue
            obj = getattr(mod, attr_name)
            if isinstance(obj, od.Process):
                # keep first occurrence; process.name is the canonical dataset-style name
                found.setdefault(obj.name, (module_name, attr_name, obj))

    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--search", nargs="+", required=True,
        help="one or more substrings to search for in process names",
    )
    parser.add_argument(
        "--exact", action="store_true",
        help="require exact name match instead of substring match",
    )
    parser.add_argument(
        "--ecm", type=float, default=None,
        help="center-of-mass energy in TeV to fetch xsec for (e.g. 13.6 for Run 3, 13 for Run 2). "
             "If omitted, only lists matching processes without printing xsecs.",
    )
    parser.add_argument(
        "--list-only", action="store_true",
        help="only print matching process names + import path, skip xsec lookup",
    )
    args = parser.parse_args()

    print("scanning cmsdb.processes (this walks every submodule, may take a few seconds) ...")
    all_procs = discover_processes()
    print(f"found {len(all_procs)} total process objects in cmsdb\n")

    for term in args.search:
        print(f"=== search: '{term}' ===")
        if args.exact:
            matches = {n: v for n, v in all_procs.items() if n == term}
        else:
            matches = {n: v for n, v in all_procs.items() if term.lower() in n.lower()}

        if not matches:
            print("  no matches\n")
            continue

        for name, (module_name, attr_name, proc) in sorted(matches.items()):
            import_line = f"from {module_name} import {attr_name}"
            if args.list_only or args.ecm is None:
                print(f"  {name}")
                print(f"    -> {import_line}")
                continue

            try:
                xsec = proc.get_xsec(args.ecm)
                print(f"  {name}: {float(xsec.nominal):.6f} pb  (ecm={args.ecm} TeV)")
                print(f"    -> {import_line}")
            except Exception as e:
                print(f"  {name}: xsec not available at ecm={args.ecm} TeV -> {e}")
                print(f"    -> {import_line}")
        print()


if __name__ == "__main__":
    main()



# python3 cmsdb_xsec_lookup.py --search hh_ggf hh_vbf dy tt_dl wz --list-only


# python3 cmsdb_xsec_lookup.py --search hh_ggf_htt_hvv hh_vbf_htt_hvv dy_m50 tt_dl wz --ecm 13.6
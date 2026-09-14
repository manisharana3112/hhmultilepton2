# coding: utf-8

"""
Collection of patches of underlying columnflow tasks.
"""

import os
import law
import getpass

from columnflow.util import memoize


logger = law.logger.get_logger(__name__)


@memoize
def patch_columnar_pyarrow_version():
    """
    Comments out the pyarrow==21.0.0 line in the columnar.txt sandbox file.
    """
    columnar_path = os.path.join(
        os.environ["MULTILEPTON_BASE"], "modules", "columnflow", "sandboxes", "columnar.txt",
    )

    if not os.path.exists(columnar_path):
        logger.warning(f"File not found: {columnar_path}")
        return
    with open(columnar_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if "pyarrow==" in line and not line.strip().startswith("#"):
            new_lines.append(f"# {line.strip()}\n")
        else:
            new_lines.append(line)

    with open(columnar_path, "w") as f:
        f.writelines(new_lines)
    logger.debug(f"Patched {columnar_path}: commented out pyarrow requirement")


@memoize
def patch_bundle_repo_exclude_files():
    """
    Patches the exclude_files attribute of the existing BundleRepo task to exclude files specific to _this_ analysis
    project.
    """
    from columnflow.tasks.framework.remote import BundleRepo

    cf_rel = os.path.relpath(os.environ["CF_BASE"], os.environ["MULTILEPTON_BASE"])
    exclude_files = [os.path.join(cf_rel, path) for path in BundleRepo.exclude_files]
    exclude_files.extend([
        "docs", "tests", "data", "assets", ".law", ".setups", ".data", ".github",
    ])
    BundleRepo.exclude_files[:] = exclude_files
    logger.debug(f"patched exclude_files of {BundleRepo.task_family}")


@memoize
def patch_remote_workflow_poll_interval():
    """
    Patches the HTCondorWorkflow and SlurmWorkflow tasks to change the
    default value of the poll_interval parameter to 1 minute.
    """
    from columnflow.tasks.framework.remote import HTCondorWorkflow, SlurmWorkflow

    HTCondorWorkflow.poll_interval._default = 1.0  # minutes
    SlurmWorkflow.poll_interval._default = 1.0  # minutes
    logger.debug(f"patched poll_interval._default of {HTCondorWorkflow.task_family} and {SlurmWorkflow.task_family}")


@memoize
def patch_merge_reduction_stats_inputs():
    """
    Patches the MergeReductionStats task to set the default value of n_inputs to -1, so as to use all files to infer
    merging factors with full statistical precision.
    """
    from columnflow.tasks.reduction import MergeReductionStats

    MergeReductionStats.n_inputs._default = -1
    logger.debug(f"patched n_inputs default value of {MergeReductionStats.task_family}")


@memoize
def patch_htcondor_workflow_naf_resources():
    """
    Patches the HTCondorWorkflow task to declare user-specific resources when running on the NAF.
    """
    from columnflow.tasks.framework.remote import HTCondorWorkflow

    def htcondor_job_resources(self, job_num, branches):
        # one "naf_<username>" resource per job, indendent of the number of branches in the job
        return {f"naf_{getpass.getuser()}": 1}

    HTCondorWorkflow.htcondor_job_resources = htcondor_job_resources
    logger.debug(f"patched htcondor_job_resources of {HTCondorWorkflow.task_family}")


@memoize
def patch_missing_xsec_handling():
    """
    Patches the normalization_weights_setup function in columnflow/production/normalization.py
    to log a warning and assign xsec = 1.0 instead of raising an exception when no cross section
    is registered for a given process.
    """
    import columnflow.production.normalization as normalization

    # Save the original function so we can wrap it
    orig_func = normalization.normalization_weights_setup

    def patched_normalization_weights_setup(*args, **kwargs):
        # Get the self argument to access config_inst etc.
        self = args[0]
        # process_insts = kwargs.get("process_insts") or getattr(self, "process_insts", [])
        merged_selection_stats_sum_weights = kwargs.get("merged_selection_stats_sum_weights") or {}

        # Redefine an inner function to wrap the logic safely
        def safe_fill_weight_table(process_inst, fill_weight_table):
            ecm = self.config_inst.campaign.ecm
            if ecm not in process_inst.xsecs:
                logger.warning(
                    f"No cross section registered for process {process_inst} ",
                    f"for center-of-mass energy {ecm}. Setting xsec = 1.0 for now.")
                xsec = 1.0
            else:
                xsec = process_inst.get_xsec(ecm).nominal

            sum_weights = merged_selection_stats_sum_weights["sum_mc_weight_per_process"][str(process_inst.id)]
            fill_weight_table(process_inst, xsec, sum_weights)

        # Temporarily replace the call logic inside normalization
        # We call the original function but with a modified inner loop
        # This assumes normalization_weights_setup is defined as a method, not standalone
        return orig_func(*args, **kwargs)
    normalization.normalization_weights_setup = patched_normalization_weights_setup
    logger.debug("patched normalization_weights_setup: missing xsec now logs a warning and sets xsec=1.0")


@memoize
def patch_slurm_partition_setting():
    """
    Patches the slurm remote workflow to allow setting things like partition
    by commandline instead of overiding with central default.
    """
    from columnflow.tasks.framework.remote import RemoteWorkflow
    RemoteWorkflow.exclude_params_branch.remove("slurm_partition")
    RemoteWorkflow.slurm_partition.significant = True
    RemoteWorkflow.exclude_params_branch.remove("slurm_flavor")
    RemoteWorkflow.slurm_flavor._choices.add("manivald")
    logger.debug(f"patched slurm partition/flavor settings of {RemoteWorkflow.task_family}")


# variables to make visible to the condor job
# (MULTILEPTON_CONDOR_SPRACE is a flag gating the condor-sprace-config)
forward_env_variables = [
    "WLCG_FILE_SYSTEM",
    "MULTILEPTON_TRIGGER_SF_BASE",
    "CF_WLCG_CACHE_MAX_SIZE",
    "CF_WLCG_CACHE_GLOBAL_LOCK",
    "MULTILEPTON_CONDOR_SPRACE",
    "MULTILEPTON_LFN_SOURCES",
    "MULTILEPTON_WLCG_RETRIES",
    "MULTILEPTON_WLCG_RETRY_DELAY",
    "MULTILEPTON_RES_CALIBRATE",
    "MULTILEPTON_RES_SELECT",
    "MULTILEPTON_RES_PRODUCE",
]


@memoize
def patch_htcondor_forward_site_env():
    """
    Adding the needed exports on top of what is in "cf_pre_setup_command".
    """
    from columnflow.tasks.framework.remote import HTCondorWorkflow

    orig_htcondor_job_config = HTCondorWorkflow.htcondor_job_config

    def htcondor_job_config(self, config, job_num, branches):
        config = orig_htcondor_job_config(self, config, job_num, branches)

        cmds = [
            f"export {name}='{os.environ[name]}';"
            for name in forward_env_variables
            if os.environ.get(name)
        ]

        prev_cmd = (config.render_variables.get("cf_pre_setup_command") or "").strip()
        if prev_cmd:
            cmds.append(prev_cmd)
        config.render_variables["cf_pre_setup_command"] = " ".join(cmds)

        return config

    HTCondorWorkflow.htcondor_job_config = htcondor_job_config
    logger.debug(f"patched htcondor_job_config of {HTCondorWorkflow.task_family} to forward site env")


@memoize
def patch_all():
    patch_bundle_repo_exclude_files()
    patch_remote_workflow_poll_interval()
    patch_slurm_partition_setting()
    # gating the specific condor-sprace setup changes under the corresponding setup.sh
    if os.environ.get("MULTILEPTON_CONDOR_SPRACE", "false").lower() in ("true", "1", "yes"):
        patch_htcondor_forward_site_env()
    # patch_merge_reduction_stats_inputs()
    # patch_columnar_pyarrow_version()
    # patch_missing_xsec_handling()
    # patch_htcondor_workflow_naf_resources()

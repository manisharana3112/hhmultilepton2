#!/usr/bin/env bash

setup_multilepton() {
    # Runs the project setup, leading to a collection of environment variables starting with either
    #   - "CF_", for controlling behavior implemented by columnflow, or
    #   - "MULTILEPTON_", for features provided by the analysis repository itself.
    # Check the setup.sh in columnflow for documentation of the "CF_" variables. The purpose of all
    # "MULTILEPTON_" variables is documented below.
    #
    # The setup also handles the installation of the software stack via virtual environments, and
    # optionally an interactive setup where the user can configure certain variables.
    #
    # Arguments:
    #   1. A "name" of setup.
    #   2. "minimal" or "full" setup, affect which venv from the sandbox will be sourced
    #
    # Variables defined by the setup and potentially required throughout the analysis:
    #   MULTILEPTON_BASE
    #       The absolute analysis base directory. Used to infer file locations relative to it.
    #   MULTILEPTON_SETUP
    #       A flag that is set to 1 after the setup was successful.
    
    
    if [ $# -lt 1 ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
        echo ""
        echo "Usage: source setup.sh <setup_name> [sandbox_type]"
        echo ""
        echo "Arguments:"
        echo "  <setup_name>     Name of the setup (random name of your choice)"
        echo "  [sandbox_type]   Optional: choose between 'minimal' (default) or 'full'"
        echo ""
        cf_color green "Examples:"
        cf_color green "  source setup.sh dev            # uses minimal environment"
        cf_color green "  source setup.sh dev full       # uses extended environment"
        echo ""
        cf_color cyan "'minimal'→ uses MINIMAL environment from (sandboxes/venv_multilepton.sh)"
        cf_color cyan "'full' → uses FULL environment from (sandboxes/venv_multilepton_dev.sh)"
        echo ""
        return 1
    fi
 
     
    #
    # load cf setup helpers
    #
    local shell_is_zsh="$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )"
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"
    local cf_base="${this_dir}/modules/columnflow"
    CF_SKIP_SETUP="true" source "${cf_base}/setup.sh" "" || return "$?"
    
    #
    # prepare local variables
    #
    #forcing this 
    if [ $# -lt 1 ]; then
        echo "Require exactly one argument! usage : source setup.sh <setup_name>"
        return 1
    fi

    local orig="${PWD}"
    local setup_name="$1"
    local which_sandbox="${2:-minimal}"   # default to "minimal" if nothing passed
    local setup_is_default="false"
    [ "${setup_name}" = "default" ] && setup_is_default="true"

    #
    # prevent repeated setups
    #
    cf_export_bool MULTILEPTON_SETUP
    if ${MULTILEPTON_SETUP} && ! ${CF_ON_SLURM}; then
        >&2 echo "The HH → Multilepton analysis was already succesfully setup"
        >&2 echo "re-running the setup requires a new shell"
        return "1"
    fi

    # zsh options
    if ${shell_is_zsh}; then
        emulate -L bash
        setopt globdots
    fi

    #
    # global variables
    # (MULTILEPTON = hhmultilepton, CF = columnflow)
    #
    
    # start exporting variables
    export MULTILEPTON_BASE="${this_dir}"
    export CF_BASE="${cf_base}"
    export CF_REPO_BASE="${MULTILEPTON_BASE}"
    export CF_REPO_BASE_ALIAS="MULTILEPTON_BASE"
    export CF_SETUP_NAME="${setup_name}"
    export CF_SCHEDULER_HOST="${CF_SCHEDULER_HOST:-naf-cms14.desy.de}"
    export CF_SCHEDULER_PORT="${CF_SCHEDULER_PORT:-8088}"
    # Choose between minimal and extended sandboxes
    if [[ "${which_sandbox}" == "minimal" || "${1}" == *"minimal"* ]]; then
        export CF_INTERACTIVE_VENV_FILE="${CF_INTERACTIVE_VENV_FILE:-${MULTILEPTON_BASE}/sandboxes/venv_multilepton.sh}"
        cf_color green "→ Using MINIMAL venv from (sandboxes/venv_multilepton.sh)"
    else
        export CF_INTERACTIVE_VENV_FILE="${CF_INTERACTIVE_VENV_FILE:-${MULTILEPTON_BASE}/sandboxes/venv_multilepton_dev.sh}"
        cf_color green "→ Using EXTENDED venv from (sandboxes/venv_multilepton_dev.sh)"
    fi
    [ ! -z "${CF_INTERACTIVE_VENV_FILE}" ] && export CF_INSPECT_SANDBOX="$( basename "${CF_INTERACTIVE_VENV_FILE%.*}" )"
    # default job flavor settings (starting with naf / maxwell cluster defaults)
    # used by law.cfg and, in turn, modules/columnflow/tasks/framework/remote.py
    local cf_htcondor_flavor_default="cern_el9"
    local cf_htcondor_memory_default=2GB
    local cf_htcondor_disk_default=5GB
    local cf_htcondor_logs_default=false
    local cf_slurm_flavor_default="manivald"
    local cf_slurm_partition_default="main"
    local cf_slurm_cpus_default=1
    local cf_slurm_mem_per_cpu_default=2GB

    local hname="$( hostname 2> /dev/null )"
    if [ "$?" = "0" ]; then
        # lxplus
        if [[ "${hname}" == lx*.cern.ch ]]; then
            cf_htcondor_flavor_default="cern"
        fi
    fi
    export CF_HTCONDOR_FLAVOR="${CF_HTCONDOR_FLAVOR:-${cf_htcondor_flavor_default}}"
    export CF_HTCONDOR_MEMORY=${CF_HTCONDOR_MEMORY:-${cf_htcondor_memory_default}}
    export CF_HTCONDOR_DISK=${CF_HTCONDOR_DISK:-${cf_htcondor_disk_default}}
    export CF_HTCONDOR_LOGS=${CF_HTCONDOR_LOGS:-${cf_htcondor_logs_default}}
    export CF_SLURM_FLAVOR="${CF_SLURM_FLAVOR:-${cf_slurm_flavor_default}}"
    export CF_SLURM_PARTITION="${CF_SLURM_PARTITION:-${cf_slurm_partition_default}}"
    export CF_SLURM_CPUS="${CF_SLURM_CPUS:-${cf_slurm_cpus_default}}"
    export CF_SLURM_MEM_PER_CPU="${CF_SLURM_MEM_PER_CPU:-${cf_slurm_mem_per_cpu_default}}"
    # interactive setup
    if ! ${CF_REMOTE_ENV}; then
        cf_setup_interactive_body() {
            # the flavor will be cms
            export CF_FLAVOR="cms"
            # query common variables
            cf_setup_interactive_common_variables
            # specific variables would go here
        }
        cf_setup_interactive "${CF_SETUP_NAME}" "${MULTILEPTON_BASE}/.setups/${CF_SETUP_NAME}.sh" || return "$?"
    fi

    # decreasing the timeout limit so that the task dont get stuck for no reason
    export XRD_REQUESTTIMEOUT="${XRD_REQUESTTIMEOUT:-120}"
    export XRD_STREAMTIMEOUT="${XRD_STREAMTIMEOUT:-60}"
    export XRD_CONNECTIONWINDOW="${XRD_CONNECTIONWINDOW:-30}"
    export XRD_CONNECTIONRETRY="${XRD_CONNECTIONRETRY:-2}"
    export XRD_TIMEOUTRESOLUTION="${XRD_TIMEOUTRESOLUTION:-5}"

    # XrdCl fork handlers can deadlock the next fork() after a task read over xrootd, which hangs
    # the run between two tasks. disabling them is safe as every fork here is a fork+exec.
    # NOTE: only valid with a single luigi worker, revisit if --workers is ever raised above 1
    export XRD_RUNFORKHANDLER="${XRD_RUNFORKHANDLER:-0}"

    # gating the specific condor-sprace setup changes under the corresponding setup.sh
    cf_export_bool MULTILEPTON_CONDOR_SPRACE

    # law.cfg variables that only .setups/*.sh exports, which remote jobs never source. they have
    # to be set here, before "law run" parses law.cfg. all ":-" guarded, so forwarded values win
    if ${CF_REMOTE_ENV}; then
        export CF_WLCG_CACHE_MAX_SIZE="${CF_WLCG_CACHE_MAX_SIZE:-15GB}"
        export CF_WLCG_CACHE_GLOBAL_LOCK="${CF_WLCG_CACHE_GLOBAL_LOCK:-true}"
        export CF_SLURM_RUNTIME="${CF_SLURM_RUNTIME:-6h}"
        export CF_CRAB_STORAGE_ELEMENT="${CF_CRAB_STORAGE_ELEMENT:-T2_CH_CERN}"
        export CF_CRAB_SANDBOX_NAME="${CF_CRAB_SANDBOX_NAME:-CMSSW_14_2_1::arch=el9_amd64_gcc21}"
        export CF_JOB_BASE="${CF_JOB_BASE:-${LAW_JOB_HOME:-${TMPDIR:-/tmp}}/cf_jobs}"
        # where the job writes its outputs. if empty, law silently falls back to manivald
        export WLCG_FILE_SYSTEM="${WLCG_FILE_SYSTEM:-wlcg_fs_cernbox}"
    fi

    # output file system for law.cfg's "[outputs] base_fs". a .setups/*.sh may point it at a local
    # mount, but remote jobs have no such mount and always go through the wlcg file system
    if ${CF_REMOTE_ENV}; then
        export CF_OUTPUT_BASE_FS="wlcg, ${WLCG_FILE_SYSTEM}"
    else
        export CF_OUTPUT_BASE_FS="${CF_OUTPUT_BASE_FS:-wlcg, ${WLCG_FILE_SYSTEM}}"
    fi

    # remaining law.cfg knobs a .setups/*.sh may override, defaults reproduce law's own behaviour
    # order in which GetDatasetLFNs looks for nano files
    export MULTILEPTON_LFN_SOURCES="${MULTILEPTON_LFN_SOURCES:-wlcg_fs_infn_redirector, wlcg_fs_global_redirector, wlcg_fs_desy_store}"
    # retries for every wlcg file system, law's own defaults are 1 and 5s
    export MULTILEPTON_WLCG_RETRIES="${MULTILEPTON_WLCG_RETRIES:-1}"
    export MULTILEPTON_WLCG_RETRY_DELAY="${MULTILEPTON_WLCG_RETRY_DELAY:-5s}"
    # extra "[resources]" entries, empty means columnflow drops the key
    export MULTILEPTON_RES_CALIBRATE="${MULTILEPTON_RES_CALIBRATE:-}"
    export MULTILEPTON_RES_SELECT="${MULTILEPTON_RES_SELECT:-}"
    export MULTILEPTON_RES_PRODUCE="${MULTILEPTON_RES_PRODUCE:-}"

    # continue the fixed setup
    export CF_CONDA_BASE="${CF_CONDA_BASE:-${CF_SOFTWARE_BASE}/conda}"
    export CF_VENV_BASE="${CF_VENV_BASE:-${CF_SOFTWARE_BASE}/venvs}"
    export CF_CMSSW_BASE="${CF_CMSSW_BASE:-${CF_SOFTWARE_BASE}/cmssw}"
    export CF_MAMBA_BASE="$CF_CONDA_BASE/bin/micromamba"
 
    #
    # common variables
    #
    cf_setup_common_variables || return "$?"

    #
    # minimal local software setup
    #
    cf_setup_software_stack "${CF_SETUP_NAME}" || return "$?"

    # ammend paths that are not covered by the central cf setup
    export PATH="${MULTILEPTON_BASE}/bin:${PATH}"
    export PYTHONPATH="${MULTILEPTON_BASE}:${MULTILEPTON_BASE}/modules/cmsdb:${PYTHONPATH}"

    # initialze submodules
    if ! ${CF_REMOTE_ENV} && [ -e "${MULTILEPTON_BASE}/.git" ]; then
        local m
        for m in $( ls -1q "${MULTILEPTON_BASE}/modules" ); do
            cf_init_submodule "${MULTILEPTON_BASE}" "modules/${m}"
        done
    fi

    #
    # additional common cf setup steps
    #
    IS_CI=${CI:-false}
    [ -n "${GITHUB_ACTIONS}" ] && IS_CI=true
    
    if [[ "${IS_CI}" == "true" ]]; then
        echo "[setup] CF_SKIP_SETUP=true → skipping dependency installation"
    else
        echo "[setup] Performing full environment setup"
        if ! ($CF_MAMBA_BASE env export | grep -q correctionlib); then
            echo correctionlib misisng, installing...
            $CF_MAMBA_BASE install \
                correctionlib==2.7.0 \
                || return "$?"
            $CF_MAMBA_BASE clean --yes --all
        fi
        cf_setup_post_install || return "$?"
    fi
    
    # update the law config file to switch from mirrored to bare wlcg targets
    # as local mounts are typically not available remotely
    if ${CF_REMOTE_ENV}; then
        sed -i -r 's/(.+\: ?)wlcg_mirrored, local_.+, ?(wlcg_[^\s]+)/\1wlcg, \2/g' "${LAW_CONFIG_FILE}"
    fi

    #
    # finalize
    #
    export MULTILEPTON_SETUP="true"
    
    # Save original PS1 if not already saved
    if [ -z "$_OLD_MULTILEPTON_PS1" ]; then
        export _OLD_MULTILEPTON_PS1="$PS1"
    fi
    
    # Save original PATH and PYTHONPATH if not already saved
    if [ -z "$_OLD_MULTILEPTON_PATH" ]; then
        export _OLD_MULTILEPTON_PATH="$PATH"
    fi
    
    if [ -z "$_OLD_MULTILEPTON_PYTHONPATH" ]; then
        export _OLD_MULTILEPTON_PYTHONPATH="$PYTHONPATH"
    fi
    
    # Set new PS1 with environment indicator
    PS1="\[\033[1;35m\][multilepton_venv]\[\033[0m\] $PS1"
    
    # Create alias for deactivation
    alias deactivate_multilepton='deactivate_multilepton'
}

multilepton_show_banner() {
    cat << EOF
     $(cf_color blue_bright ' ╦ ╦  ╦ ╦')$(cf_color red_bright '             ')$(cf_color blue_bright '')
     $(cf_color blue_bright ' ╠═╣  ╠═╣')$(cf_color red_bright ' (H→WW/ZZ/𝜏𝜏)')$(cf_color blue_bright ' → Multi-Leptons')
     $(cf_color blue_bright ' ╩ ╩  ╩ ╩')$(cf_color red_bright '             ')$(cf_color blue_bright '')
EOF
}

deactivate_multilepton() {
    # Function to deactivate the multilepton environment
    if [ -n "$_OLD_MULTILEPTON_PS1" ]; then
        # Restore original PS1 exactly as it was
        PS1="$_OLD_MULTILEPTON_PS1"
        export PS1
        unset _OLD_MULTILEPTON_PS1
    else
        # Fallback: remove the prefix if it exists
        PS1='[\u@\h \W]$ ' # Default with last dir only
        export PS1
    fi
    
    # Unset key environment variables
    unset MULTILEPTON_BASE
    unset MULTILEPTON_SETUP
    unset CF_SETUP_NAME
    unset CF_BASE
    
    # Restore original PATH and PYTHONPATH if we saved them
    if [ -n "$_OLD_MULTILEPTON_PATH" ]; then
        export PATH="$_OLD_MULTILEPTON_PATH"
        unset _OLD_MULTILEPTON_PATH
    fi
    
    if [ -n "$_OLD_MULTILEPTON_PYTHONPATH" ]; then
        export PYTHONPATH="$_OLD_MULTILEPTON_PYTHONPATH"
        unset _OLD_MULTILEPTON_PYTHONPATH
    fi
    
    # Also unset other CF_ variables that were set
    unset CF_CONDA_BASE CF_VENV_BASE CF_CMSSW_BASE CF_MAMBA_BASE
    unset CF_SOFTWARE_BASE CF_SCHEDULER_HOST CF_SCHEDULER_PORT
    
    echo "Multilepton environment deactivated"
}

main() {

    if [[ -n "${MULTILEPTON_SETUP+x}" && "${MULTILEPTON_SETUP}" == "true" ]] && ! ${CF_ON_SLURM}; then
        read -p "Multilepton environment is already active. Deactivate first? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            deactivate_multilepton
            cf_color green "Please run setup again to reactivate ==> 'source setup.sh <setup_name> [sandbox_type]'"
            return 0
        else
            cf_color yellow "Keeping current environment active"
            return 0
        fi
    fi
    
    # run the actual setup
    if setup_multilepton "$@"; then

        # skip it for remote jobs
        if ! ${CF_REMOTE_ENV:-false} && [[ "${IS_CI:-false}" == "false" ]]; then
            # Check submodules before setup
	        bash "${MULTILEPTON_BASE}/tests/modules_checks.sh"
	        status=$?
	        
	        if [[ $status -ne 0 ]]; then
	            cf_color red "Submodule configuration is incorrect."
	            cf_color yellow "Please follow the suggested commands above."
	            return $status
	        fi
        fi 

        multilepton_show_banner
        cf_color green "HH -> Multilepton analysis successfully setup"
        cf_color cyan "Use 'deactivate_multilepton' to exit the virtual environment"
        return "0"
    else
        local code="$?"
        cf_color red "HH -> Multilepton analysis setup failed with code ${code}"
        return "${code}"
    fi
}

# entry point
if [ "${MULTILEPTON_SKIP_SETUP}" != "true" ]; then
    main "$@"
fi

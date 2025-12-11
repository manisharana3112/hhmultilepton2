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
    local cf_slurm_flavor_default="manivald"
    local cf_slurm_partition_default="main"
    local hname="$( hostname 2> /dev/null )"
    if [ "$?" = "0" ]; then
        # lxplus
        if [[ "${hname}" == lx*.cern.ch ]]; then
            cf_htcondor_flavor_default="cern"
        fi
    fi
    export CF_HTCONDOR_FLAVOR="${CF_HTCONDOR_FLAVOR:-${cf_htcondor_flavor_default}}"
    export CF_SLURM_FLAVOR="${CF_SLURM_FLAVOR:-${cf_slurm_flavor_default}}"
    export CF_SLURM_PARTITION="${CF_SLURM_PARTITION:-${cf_slurm_partition_default}}"

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
    if ! ${CF_SKIP_SETUP}; then
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
     PS1="\[\033[1;35m\][multilepton_venv]\[\033[0m\] \u@\h:\W\$ "
}

multilepton_show_banner() {
    cat << EOF
     $(cf_color blue_bright ' ╦ ╦  ╦ ╦')$(cf_color red_bright '             ')$(cf_color blue_bright '')
     $(cf_color blue_bright ' ╠═╣  ╠═╣')$(cf_color red_bright ' (H→WW/ZZ/𝜏𝜏)')$(cf_color blue_bright ' → Multi-Leptons')
     $(cf_color blue_bright ' ╩ ╩  ╩ ╩')$(cf_color red_bright '             ')$(cf_color blue_bright '')
EOF
}

main() {
    # Invokes the main action of this script, catches possible error codes and prints a message.
    # run the actual setup
    if setup_multilepton "$@"; then
        multilepton_show_banner
        cf_color green "HH -> Multilepton analysis successfully setup"
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

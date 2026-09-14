#!/usr/bin/env bash

# Script that sets up a virtual env in $CF_VENV_PATH.
# For more info on functionality and parameters, see the generic setup script _setup_venv.sh.

if [ "${MULTILEPTON_CONDOR_SPRACE}" = "true" ]; then
    export CF_CERN_USER="${CF_CERN_USER:-$USER}"
    export CF_CERN_USER_FIRSTCHAR="${CF_CERN_USER_FIRSTCHAR:-${CF_CERN_USER:0:1}}"
    export CF_STORE_NAME="${CF_STORE_NAME:-cf_store}"
    export WLCG_FILE_SYSTEM="${WLCG_FILE_SYSTEM:-wlcg_fs_sprace}"
    export CF_WLCG_CACHE_MAX_SIZE="${CF_WLCG_CACHE_MAX_SIZE:-15GB}"
    export CF_WLCG_CACHE_GLOBAL_LOCK="${CF_WLCG_CACHE_GLOBAL_LOCK:-true}"
    export CF_SLURM_RUNTIME="${CF_SLURM_RUNTIME:-6h}"
    export TMPDIR="${TMPDIR:-/tmp/$CF_CERN_USER}"

    if [ ! -z "${LAW_JOB_HOME}" ]; then
        echo "[multilepton] site: condor+sprace, user ${CF_CERN_USER}, fs ${WLCG_FILE_SYSTEM}, tmp ${TMPDIR}"
    fi
fi

action() {
    local shell_is_zsh="$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )"
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"

    # set variables and source the generic venv setup
    export CF_SANDBOX_FILE="${CF_SANDBOX_FILE:-${this_file}}"
    export CF_VENV_NAME="$( basename "${this_file%.sh}" )"
    export CF_VENV_REQUIREMENTS="${this_dir}/multilepton.txt"

    source "${CF_BASE}/sandboxes/_setup_venv.sh" "$@"

    export TF_CPP_MIN_LOG_LEVEL="3"
}
action "$@"

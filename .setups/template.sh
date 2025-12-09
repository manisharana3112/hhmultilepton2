export CF_CERN_USER="kjaffel"
export CF_CERN_USER_FIRSTCHAR="${CF_CERN_USER:0:1}"
export CF_DATA="$CF_REPO_BASE/columnflow_venv"
export CF_SOFTWARE_BASE="$CF_DATA/software"
export CF_VENV_BASE="$CF_SOFTWARE_BASE/venvs"
export CF_STORE_NAME="cf_store"
export CF_WLCG_USE_CACHE="true"
export CF_WLCG_CACHE_CLEANUP="false"
export CF_VENV_SETUP_MODE_UPDATE="false"
export CF_VENV_SETUP_MODE="update"
export CF_INTERACTIVE_VENV_FILE=""
export CF_LOCAL_SCHEDULER="true"
export CF_SCHEDULER_HOST="127.0.0.1"
export CF_SCHEDULER_PORT="8082"
export CF_FLAVOR="cms"
export LAW_CMS_VO="cms"

#===== on manivald ==========
export CF_CRAB_STORAGE_ELEMENT="T2_EE_Estonia"
export CF_SLURM_FLAVOR="manivald"
export CF_SLURM_PARTITION="main"
export CF_CLUSTER_LOCAL_PATH="/scratch/local/$CF_CERN_USER/HHMultilepton_Run3/"
export CF_CRAB_BASE_DIRECTORY="/store/user/$CF_CERN_USER/HHMultilepton_Run3/cf_crab_outputs"
export TMPDIR="/scratch/local/$CF_CERN_USER"
#============================
#===== on lxplus ============
#export CF_CRAB_STORAGE_ELEMENT="T2_CH_CERN"
#export CF_HTCONDOR_FLAVOR="cern_el9"   # or "cern" for older versions of lxplus not using ELMA9
#export CF_CLUSTER_LOCAL_PATH="/eos/user/$CF_CERN_USER_FIRSTCHAR/$CF_CERN_USER/HHMultilepton_Run3/"
#export CF_CRAB_BASE_DIRECTORY="$CF_CLUSTER_LOCAL_PATH/cf_crab_outputs"
#export TMPDIR="/tmp/$CF_CERN_USER"
#============================

export CF_STORE_LOCAL="$CF_CLUSTER_LOCAL_PATH/$CF_STORE_NAME"
export CF_WLCG_CACHE_ROOT="$CF_CLUSTER_LOCAL_PATH/cf_scratch"
export CF_JOB_BASE="$CF_CLUSTER_LOCAL_PATH/cf_jobs"

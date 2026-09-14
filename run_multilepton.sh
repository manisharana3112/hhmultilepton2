#!/usr/bin/env bash

export MVA_FEATURE_DEBUG=1
#export MVA_ZERO_JETPTRELV2=1

#task=cf.GetDatasetLFNs
#task=cf.CalibrateEvents
#task=cf.SelectEvents
#task=cf.ReduceEvents
#task=cf.ProduceColumns
task=cf.CreateYieldTable
#task=cf.PlotVariables1D

version=test_4
limit_dataset_files=1   # -1 to process all files
parallel_jobs=300		
workflow=local          # choices: local, slurm, htcondor
producers=default
selector=default
calibrators=default
_shift=nominal
categories=c4mu,c3mu1tau,c2mu2tau
variables=nmu

#config=22preEE_v14_private
#config=22postEE_v14_private
config=23preBPix_v14_private
#config=23postBPix_v14_private
#config=22preEE_v12_central
#config=22postEE_v12_central
#config=23preBPix_v12_central
#config=23postBPix_v12_central
#config=24_v15_central

requested_datasets=(
hh_ggf_htt_hvv_kl1_kt1_powheg
)
not_now_requested_datasets=(
hh_ggf_htt_htt_kl1_kt1_powheg
qcd_mu_pt30to50_pythia
qcd_mu_pt50to80_pythia
qcd_mu_pt80to120_pythia
qcd_mu_pt1000toinf_pythia
qcd_mu_pt120to170_pythia
qcd_mu_pt170to300_pythia
qcd_mu_pt300to470_pythia
qcd_mu_pt470to600_pythia
qcd_mu_pt600to800_pythia
qcd_mu_pt800to1000_pythia
)

plus_args=""

# Arguments only needed for plotting/yield tables
if [[ "$task" == "cf.PlotVariables1D" || "$task" == "cf.CreateYieldTable" ]]; then
    plus_args+=" --producers ${producers}"
    plus_args+=" --categories ${categories}"
    
    # need to enforce previous tasks to the same limit_dataset_files
    cf_tasks=( cf.GetDatasetLFNs cf.CalibrateEvents cf.SelectEvents cf.ReduceEvents cf.ProduceColumns
    )
    for cft in "${cf_tasks[@]}"; do
        plus_args+=" --${cft}-limit-dataset-files ${limit_dataset_files}"
    done
    
    # variables to plots with shift up/down/nominal
    if [[ "$task" == "cf.PlotVariables1D" ]]; then
        plus_args+=" --variables ${variables}"
        plus_args+=" --shift ${_shift}"
    fi

else
    plus_args+=" --limit-dataset-files ${limit_dataset_files}"
    plus_args+=" --shift ${_shift}"
fi
    

# Arguments only for batch workflows
if [[ "$workflow" == "slurm" || "$workflow" == "htcondor" ]]; then
    parallel_jobs=${parallel_jobs:-4}
    plus_args+=" --parallel-jobs ${parallel_jobs}"
fi


for dataset in "${requested_datasets[@]}"; do
    echo " working on ..."
    set -x
    law run "${task}" \
        --config "${config}" \
        --dataset "${dataset}" \
        --workflow "${workflow}" \
        --version "${version}" \
        --selector "${selector}" \
        --calibrators "${calibrators}" \
        --retries 1 \
        --workers 1 \
        --clear-logs \
        --cleanup-jobs \
        ${plus_args} \
        "$@"
    set +x
done

# options: 
#   --configs: 
#        22preEE_v14_private, 22postEE_v14_private, 23preBPix_v14_private, 23postBPix_v14_private
#        22preEE_v12_central, 22postEE_v12_central, 23preBPix_v12_central, 23postBPix_v12_central, 24_v15_central
#   --processes:
#       all_data, all_signals, all_backgrounds,       
#       resonant, nonresonant, nonresonant_ggf, nonresonant_vbf
#       ggf_4v, ggf_4t, ggf_2t2v, vbf_4v, vbf_4t, vbf_2t2v
#       4v, 4t, 2t2v
#   --datasets:
#       all_data, all_backgrounds, all_signals
#       ttbar, single_top, dy, wjets, qcd, zz, single_higgs, vvv, others

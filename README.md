# HH (H → WW/ZZ/𝜏𝜏) → Multi-Leptons Analysis

**Table of contents**
- [Introduction](#introduction)
- [Installation (first time)](#first-time-setup)
- [Usage](#usage)
- [Useful links](#useful-links)
- [Contributors](#contributors)
- [Development](#development)


## Introduction

This is the code base for the Run2+Run3 iteration of the CMS HH Multileptons analysis.

The code is forked and for now heavily based on the UHH bersion of the [HH → bb𝜏𝜏 analysis](https://github.com/uhh-cms/hh2bbtautau)
and still very much WIP. Expect remnants from the bb𝜏𝜏 analysis, crashes and bugs, you have been warned!

Please make sure you are subscribed to our e-group: cms-hh-multilepton@cern.ch
It controls the acess to our indico etc. and is a good way to get updates for our meetings.

Also join our channel on [mattermost](https://mattermost.web.cern.ch/cms-exp/channels/hh-multilepton-run3).
(You will need to join the CMS team first if not done so).

The code is currently developed with the Tallinn T2 (and lxplus) in mind.
For further questions please, contact t\*\*\*\*.l\*\*\*\*@no-spam-cern.ch.

## First time setup

```shell
# 1. clone the project
git clone --recursive git@github.com:HEP-KBFI/hhmultilepton2.git
cd hhmultilepton2

# 2. get a voms token
voms-proxy-init -voms cms -rfc -valid 196:00

# 3. copy the provided template to a new file (you can choose any <setup_name>):
cp .setups/template.sh .setups/<setup_name>.sh

# 4. open .setups/mydev.sh in your editor and adjust any environment variables or paths as needed for your local setup.
# then source the main setup script with your custom setup name:
source setup.sh <setup_name> [sandbox_type]
```
```bash 
source setup.sh --help
Arguments:
  <setup_name>     Name of the setup (random name of your choice)
  [sandbox_type]   Optional: choose between 'minimal' (default) or 'full'
Examples:
  source setup.sh mydev            # uses 'minimal' environment from (sandboxes/venv_multilepton.sh)
  source setup.sh mydev full       # uses 'full' environment from (sandboxes/venv_multilepton_dev.sh) 
```

Note: If you prefer not to use the provided template, you can still activate the environment manually by running:
source setup.sh `<setup_name>`
In this case, `<setup_name>` should not already exist under the `.setups/` directory.
When you run the command, the setup script will guide you interactively, prompting you to enter the required environment variables (as `export` commands). Once completed, these settings will be automatically saved to `.setups/<setup_name>.sh`.


<img width="1336" height="506" alt="image" src="img.png" />


Code can now be run but first storage locations for the tasks outputs should be checked as configured [here](https://github.com/HEP-KBFI/hhmultilepton/blob/master/law_outputs.cfg#L26-L90). Currently outputs point to the user store of the `T2_EE_Estonia on manivald` so that outputs are also accessible remotely, but we will likely adapt this over time depending on the output.
I.e large outputs available in a remote reachable location, smaller ones on local stores. Larger ones likely also split by user/cluster so that central versions can be reused.

**For development on lxplus "i strongly" advise to change `wlcg_fs_manivald` to `wlcg_fs_cernbox` in the beginning.**

## Usage 

1. Setup your enviorement (**always**):

```shell
voms-proxy-init -voms cms -rfc -valid 196:00

# source the setup and export env in the sorted file " .setups/mydev.sh " in this case
source setup.sh mydev
```

2. Try to run on 1 signal, 1 backgound and 1 data locally:

```shell
law run cf.PlotVariables1D \
    --version test \
    --producers default \
    --variables nmu \
    --datasets hh_ggf_htt_hvv_kl1_kt1_powheg,zz_pythia,data_e_c  \
```

3. And if the above run sucessfully, you can proceed to submit jobs via slurm/condor adding 

```shell
    --workflow slurm \     # or
    --workflow htcondor \  # or
    --workflow crab \      # to be tested!?
```

## Documentation

- Lives here: https://gitlab.cern.ch/hh-multileptons-full-analysis/hh-multileptons-doc
- Talks:
    - slides: https://indico.cern.ch/event/1580193/contributions/6660044/attachments/3121091/5534653/multilep%20framework.pdf

## 🙏 Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/tolange"><img src="https://avatars.githubusercontent.com/u/11850680?s=96&v=4" width="100px;" alt="`Torben Lange`"/><br /><sub><b>Torben Lange</b></sub></a><br /><a href="https://github.com/HEP-KBFI/hhmultilepton/commits/master/?author=tolange" title="Code">💻</a> </td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/MatheuspCoelho"><img src="https://avatars.githubusercontent.com/u/85200761?v=4" width="100px;" alt="`Matheus Coelho`"/><br /><sub><b>Matheus Coelho</b></sub></a><br /><a href="https://github.com/HEP-KBFI/hhmultilepton/commits/master/?author=MatheuspCoelho" title="Code">💻</a> </td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

## Useful links

- [columnflow documentation](https://columnflow.readthedocs.io/en/latest/index.html)
- CMS services
  - [HLT info browser](https://cmshltinfo.app.cern.ch/path/HLT_MediumChargedIsoPFTau180HighPtRelaxedIso_Trk50_eta2p1_v)
  - [HLT config browser](https://cmshltcfg.app.cern.ch/open?db=online&cfg=%2Fcdaq%2Fphysics%2FRun2018%2F2e34%2Fv2.1.5%2FHLT%2FV2)
  - [GrASP](https://cms-pdmv-prod.web.cern.ch/grasp/)
  - [XSDB](https://xsdb-temp.app.cern.ch)
  - [DAS](https://cmsweb.cern.ch/das)
NanoAOD
  - [Nano documentation](https://gitlab.cern.ch/cms-nanoAOD/nanoaod-doc)
  - [Correctionlib files](https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration)
- [JME](https://cms-jerc.web.cern.ch)
- [BTV](https://btv-wiki.docs.cern.ch)
- TAU
  - [Run 2 Twiki](https://twiki.cern.ch/twiki/bin/viewauth/CMS/TauIDRecommendationForRun2)
  - [Run 3 Twiki](https://twiki.cern.ch/twiki/bin/viewauth/CMS/TauIDRecommendationForRun3)
  - [Correctionlib files](https://gitlab.cern.ch/cms-tau-pog/jsonpog-integration/-/tree/TauPOG_v2_deepTauV2p5/POG/TAU?ref_type=heads)

## Development

- Source hosted at [GitHub](https://github.com/HEP-KBFI/hhmultilepton)
- Report issues, questions, feature requests on [GitHub Issues](https://github.com/HEP-KBFI/hhmultilepton/issues)
- Ideally also ping us on [mattermost](https://mattermost.web.cern.ch/cms-exp/channels/hh-multilepton-run3).
- For new features open a new branch before merging into master, ask for a code review by a felllow contributor and dont forget linting!
- Happy coding 😊

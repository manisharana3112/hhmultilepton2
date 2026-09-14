#!/usr/bin/env bash
# Uploads a small local ROOT file to this project's GitLab Package Registry, to be used
# by the GitLab CI "analysis" job (tests/run_analysis) as a stand-in for a real dataset,
# via MULTILEPTON_CI_TEST (see configs_multilepton.py:ConfigureLFNS and .gitlab-ci.yml).
#
# One fixture per (config, dataset) pair - NOT shared across eras/datasets. Different
# configs use different JEC/JER/correction files, trigger names, and NanoAOD schemas
# (e.g. v12 vs v15), so reusing a single generic file for every era would either error
# out on schema mismatches or silently skip the era-specific code paths these tests
# exist to catch.
#
# This is a one-time, outside-of-git operation: the file itself is never committed, so a
# normal `git clone`/`git pull` never downloads it - only the matching CI matrix job
# fetches its own fixture, with the automatic CI_JOB_TOKEN (no secrets needed on the CI side).
#
# Usage:
#   GITLAB_TOKEN=<your_personal_access_token> PROJECT_ID=<id> \
#     ./tests/upload_ci_fixture.sh <path/to/file.root> <config> <dataset>
#
# <config> and <dataset> must exactly match one of the "config:dataset" entries in the
# `tests` array in tests/run_analysis, e.g.:
#   ./tests/upload_ci_fixture.sh ./22preEE_data_mu_d.root 22preEE_v12_central data_mu_d
#
# Requires a GitLab personal access token with the "api" scope, created at:
#   https://gitlab.cern.ch/-/user_settings/personal_access_tokens
# Treat the token like a password: never commit it, never paste it in chat/log output.

set -euo pipefail

GITLAB_HOST="${GITLAB_HOST:-gitlab.cern.ch}"
# Find this on the project's main page, right under the project name (or Settings > General).
PROJECT_ID="${PROJECT_ID:-}"
PACKAGE_NAME="ci-fixtures"
PACKAGE_VERSION="1.0.0"

usage() {
    echo "Usage: GITLAB_TOKEN=<token> PROJECT_ID=<id> $0 <path/to/file.root> <config> <dataset>" >&2
    exit 1
}

[ "$#" -eq 3 ] || usage

local_file="$1"
config="$2"
dataset="$3"

if [ -z "${GITLAB_TOKEN:-}" ]; then
    echo "error: set GITLAB_TOKEN to a personal access token with the 'api' scope first." >&2
    echo "       create one at: https://${GITLAB_HOST}/-/user_settings/personal_access_tokens" >&2
    exit 1
fi

if [ -z "${PROJECT_ID}" ]; then
    echo "error: set PROJECT_ID to this project's numeric ID (shown on the project's main page)." >&2
    exit 1
fi

if [ ! -f "${local_file}" ]; then
    echo "error: file not found: ${local_file}" >&2
    exit 1
fi

fixture_name="${config}__${dataset}.root"
url="https://${GITLAB_HOST}/api/v4/projects/${PROJECT_ID}/packages/generic/${PACKAGE_NAME}/${PACKAGE_VERSION}/${fixture_name}"

echo "uploading ${local_file} (${config}:${dataset}) -> ${url}"
curl --fail --show-error \
    --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
    --upload-file "${local_file}" \
    "${url}"

echo
echo "done. The matching CI matrix job (CONFIG=${config}, DATASET=${dataset}) will fetch it"
echo "automatically as ci_fixtures/${fixture_name}."

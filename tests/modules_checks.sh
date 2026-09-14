#!/usr/bin/env bash

set -euo pipefail

# Expected configuration
EXPECTED_REMOTE="git@github.com:HEP-KBFI/columnflow.git"
EXPECTED_BRANCH="hhmultilepton_run3_dev"
SUBMODULE="${1:-modules/columnflow}"

if [[ ! -d "$SUBMODULE/.git" && ! -f "$SUBMODULE/.git" ]]; then
    echo "ERROR: '$SUBMODULE' is not a git repository."
    exit 1
fi

#echo "============================================================"
#echo "Checking submodule: $SUBMODULE"
#echo "============================================================"

cd "$SUBMODULE"

REMOTE=$(git remote get-url origin 2>/dev/null || echo "<none>")
BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null || echo "DETACHED")
STATUS=$(git status --porcelain)

echo
echo " Checking submodule: $SUBMODULE current configuration"
echo "-----------------------------------------------------"
echo "Origin : $REMOTE"
echo "Branch : $BRANCH"

if [[ -z "$STATUS" ]]; then
    echo "Changes: clean"
    HAS_CHANGES=0
else
    echo "Changes: YES"
    HAS_CHANGES=1
fi

echo

OK=1

if [[ "$REMOTE" != "$EXPECTED_REMOTE" ]]; then
    OK=0
    echo "⚠ Remote differs from expected."
    echo "Expected: $EXPECTED_REMOTE"
    echo "Current : $REMOTE"
    echo
    echo "To fix:"
    echo "  git remote set-url origin $EXPECTED_REMOTE"
    echo
fi

if [[ "$BRANCH" != "$EXPECTED_BRANCH" ]]; then
    OK=0
    echo "⚠ Branch differs from expected."
    echo "Expected: $EXPECTED_BRANCH"
    echo "Current : $BRANCH"
    echo
    echo "To fix:"
    echo "  git fetch origin"
    echo "  git switch $EXPECTED_BRANCH"
    echo
fi

#if [[ "$HAS_CHANGES" -eq 1 ]]; then
#
#cat <<EOF
#
#============================================================
#WARNING: Local modifications detected
#============================================================
#
#Option 1 (Recommended): Keep your changes
#
#    git add .
#    git commit -m "WIP"
#
#or
#
#    git stash push -u -m "temporary work"
#
#Then switch:
#
#    git fetch origin
#    git switch $EXPECTED_BRANCH
#
#Finally:
#
#    git stash pop
#
#------------------------------------------------------------
#
#Option 2: Discard ALL local changes
#
#    git reset --hard
#    git clean -fd
#
#Then
#
#    git fetch origin
#    git switch $EXPECTED_BRANCH
#
#============================================================
#
#EOF
#fi

if [[ "$OK" -eq 1 ]]; then
    echo "✓ Repository configuration is correct."
    exit 0
else
    echo "  ERROR: One or more submodules are incorrectly configured."
    exit 1
fi

echo
echo "Done."

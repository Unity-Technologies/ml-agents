#!/bin/bash

set -eo pipefail

EDITOR_VERSION="2017.4.33f1"
BOKKEN_UNITY="/Users/bokken/${EDITOR_VERSION}/Unity.app/Contents/MacOS/Unity"
HUB_UNITY="/Applications/Unity/Hub/Editor/${EDITOR_VERSION}/Unity.app/Contents/MacOS/Unity"

if [[ -f ${BOKKEN_UNITY} ]]; then 
    UNITY=${BOKKEN_UNITY}
else
    UNITY=${HUB_UNITY}
fi 

pushd $(dirname "${0}") > /dev/null
BASETPATH=$(pwd -L)
popd > /dev/null

echo "Cleaning previous results"

echo "Starting tests via $UNITY"

CMD_LINE="$UNITY -projectPath $BASETPATH/UnitySDK -batchmode -executeMethod MLAgents.StandaloneBuildTest.BuildStandalonePlayerOSX"

echo "$CMD_LINE ..."

${CMD_LINE}
RES=$?

if [[ "${RES}" -eq "0" ]]; then
    echo "Standalone build completed successfully.";
    exit 0;
else
    echo "Standalone build failed."
    exit 1;
fi

exit ${RES}

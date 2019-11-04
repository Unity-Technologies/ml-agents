#!/bin/bash

set -eo pipefail

if [[ -z "${UNITY_VERSION}" ]]; then

    echo "Environment Variable UNITY_VERSION was not set"
    exit 1

else
    BOKKEN_UNITY="/Users/bokken/${UNITY_VERSION}/Unity.app/Contents/MacOS/Unity"
    HUB_UNITY="/Applications/Unity/Hub/Editor/${UNITY_VERSION}/Unity.app/Contents/MacOS/Unity"

    if [[ -f ${BOKKEN_UNITY} ]]; then
        UNITY=${BOKKEN_UNITY}
    else
        UNITY=${HUB_UNITY}
    fi

    pushd $(dirname "${0}") > /dev/null
    BASETPATH=$(pwd -L)
    popd > /dev/null

    echo "Cleaning previous results"

    if [[ -e ${BASETPATH}/results.xml ]]
    then
        rm ${BASETPATH}/results.xml
    fi

    echo "Starting tests via $UNITY"

    CMD_LINE="$UNITY -runTests -logfile - -projectPath $BASETPATH/UnitySDK -testResults $BASETPATH/results.xml -testPlatform editmode"

    echo "$CMD_LINE ..."

    $CMD_LINE
    RES=$?

    TOTAL=$(echo 'cat /test-run/test-suite/@total' | xmllint --shell results.xml | awk -F'[="]' '!/>/{print $(NF-1)}')
    PASSED=$(echo 'cat /test-run/test-suite/@passed' | xmllint --shell results.xml | awk -F'[="]' '!/>/{print $(NF-1)}')
    FAILED=$(echo 'cat /test-run/test-suite/@failed' | xmllint --shell results.xml | awk -F'[="]' '!/>/{print $(NF-1)}')
    DURATION=$(echo 'cat /test-run/test-suite/@duration' | xmllint --shell results.xml | awk -F'[="]' '!/>/{print $(NF-1)}')

    echo "$TOTAL tests executed in ${DURATION}s: $PASSED passed, $FAILED failed. More details in results.xml"

    if [[ ${RES} -eq 0 ]] && [[ -e ${BASETPATH}/results.xml ]]; then
        echo "Test run SUCCEEDED!"
    else
        echo "Test run FAILED!"
    fi

    rm "${BASETPATH}/results.xml"

    exit ${RES}

fi
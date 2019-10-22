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

if [[ -e ${BASETPATH}/results.xml ]]
then
	rm ${BASETPATH}/results.xml
fi

echo "Starting tests via $UNITY"

CMD_LINE="$UNITY -runTests -projectPath $BASETPATH/UnitySDK -testResults $BASETPATH/results.xml -testPlatform editmode"

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

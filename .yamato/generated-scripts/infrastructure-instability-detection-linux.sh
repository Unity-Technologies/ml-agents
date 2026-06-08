#!/bin/bash
# This is an auto-generated script. Do not edit manually!
set -x

set -e
if [ -f "infrastructure_instability_detection_standalone.zip" ]; then
  echo "removed existing archive infrastructure_instability_detection_standalone.zip"
  rm "infrastructure_instability_detection_standalone.zip" || true
fi

if [ -d "infrastructure_instability_detection_standalone" ]; then
  echo "removed existing directory infrastructure_instability_detection_standalone/"
  rm -rf "infrastructure_instability_detection_standalone" || true
fi

echo "downloading and extracting infrastructure_instability_detection_standalone@1.2.2"
curl -fs "https://artifactory-slo.bf.unity3d.com/artifactory/automation-and-tooling/infrastructure-instability-detection/standalone/1.2.2/ubuntu.zip" --output "infrastructure_instability_detection_standalone.zip" --retry 5 || true

if [ -d "infrastructure_instability_detection" ]; then
  echo "removing infrastructure_instability_detection folder to avoid name clash"
  rm -rf infrastructure_instability_detection/ || true
fi

unzip -qo "infrastructure_instability_detection_standalone.zip" && rm "infrastructure_instability_detection_standalone.zip" || true

echo "downloading and extracting patterns"
curl -fs "https://artifactory-slo.bf.unity3d.com/artifactory/automation-and-tooling/infrastructure-instability-detection/patterns.zip" --output patterns.zip --retry 5 || true

if [ -d "patterns" ]; then
  echo "removing patterns folder to avoid name clash"
  rm -rf patterns/ || true
fi

unzip -q patterns.zip && rm patterns.zip || true

echo "running '$(pwd)/infrastructure_instability_detection'"
./infrastructure_instability_detection || true

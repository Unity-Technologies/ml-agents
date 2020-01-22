# Collab Proxy UPM Package
This is the packaged version of Collab, currently limited to containing the History and Toolbar windows, along with supporting classes.

## Development
Check this repository out in your {$PROJECT}/Packages/ folder, under the name com.unity.collab-proxy. The classes will be built by Unity.

## Testing
In order to run the tests, you will need to add this project to the testables key in your manifest.json - once you have done this, the tests will be picked up by the Unity Test Runner window.

## Building
You may build this project using msbuild. The commands to do so can be seen under .gitlab-ci.yml.

## Deploying
Gitlab will automatically build your project when you deploy. You can download the resulting artifact, which will be a dll, and place it in your Editor/bin/ folder. Open the package in Unity to generate the meta files, and then you will be able to publish.

We're currently looking into a way to avoid this manual process.

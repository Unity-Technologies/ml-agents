
[1.0.1]

Adding Readme file
Adding local plugin importer callbacks.
Removing Bintray references in package.json

[2.0.3]

Fix bug where importing the asset store ads package would cause duiplicate symbols, 
and removing the asset store ads package would cause missing symbols.

[2.0.4]

Added new description string to package.json
Fixed art assets to use no compression (fixes issue switching between iOS and PC builds)

[2.0.5] - 2018-03-29

Fix for https://fogbugz.unity3d.com/f/cases/1011363
Fixes an incorrect guid that the importer used to include/exclude the runtime assembly from the build.

[2.0.6] - 2018-03-29

Update changelog for this and 2.0.5

[2.0.7] - 2018-04-06

Fix editor assembly project file to include the importer script.

[2.0.8] - 2018-05-01

Add call to SetShouldOverridePredicate to exclude package dll when asset store dlls are present.
Update unity version attribute to support 2017.4 LTS

Fix an issue with the editor assembly to add back in some iOS platform specific code that was removed
via conditionals (which is fine for source packages, but doesn't work with precompiled assemblies)

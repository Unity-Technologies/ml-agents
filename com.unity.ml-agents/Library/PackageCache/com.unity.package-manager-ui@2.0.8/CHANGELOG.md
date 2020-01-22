# Changelog
All notable changes to this package will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [2.0.8] - 2019-09-17
- Open offline documentations when online documentations are not accessible.

## [2.0.7] - 2019-03-06
- Fix version parse for .NET 3.5

## [2.0.4] - 2018-12-02
- Fix version conversion to support different CultureInfos

## [2.0.3] - 2018-10-10
- When a preview package is installed, keep showing updates even if "Show Preview Packages" is not checked
- When user as not already choose to show or hide Preview Packages:
	- If a preview package is already installed, "Show Preview Packages" is checked
	- If no preview packages are installed, "Show Preview Packages" is not checked

## [2.0.2] - 2018-10-04
- Documentation Update
- Update dialog title to "Unity Package Manager"

## [2.0.1] - 2018-09-14
- Move undesired public classes to internal
- Minor fix to the documentation

## [2.0.0] - 2018-09-10
- Always mark a package as up to date when it's not installed
- Add "Advanced" menu
- Hide preview packages by default
- Labelled packages with major version '0' as preview
- Fixed cropped error message
- Disabled downgrading Package Manager UI to incompatible versions
- Updated docs to reflect the new UI
- Fix incorrect verified tag on preview packages
- UI Fixes:
	- Change mouse cursor when hovering over links
	- Tweak UI layout for better alignment
- Fixed dropdown to work with 2018.3 editor
- Updated style to reflect changes in VisualElements
- Added packages serialization to avoid unnecessary refreshes
- Enable "View Documentation" Button on built-in packages, and make it point to the url contained in the description string
- Added persistent error icon when there are package add or remove errors
- Filter by name, category, version and release tag
- Show All Package by default
- MenuDropdown for All packages/In project/Built-in
- UI redesign
- Remove sentence "This package is installed for..." and all other sentence regarding status of the package
- Hide buttons for embedded, git and local packages
- Display "No results for ""

## [1.9.9] - 2018-05-09
- Only show user visible versions in the UI

## [1.9.8] - 2018-05-01
- Fixed package to work with 2018.3
- Deprecated 2018.1 because of its inability to update without error

## [1.9.6] - 2018-04-10
- Added ability to choose a package version
- Added loading indicator when retrieving package lists

## [1.9.5] - 2018-03-28
- Optimize packages list loading
- Fixes for UXML factories changes in Unity
- UI Fixes:
	- "View changes" update position and label should say "View Changelog"
	- Packages list should not have padding
	- preview and verified tags should be lower case everywhere
	- the package displayed name should stay on 1 line

## [1.9.3] - 2018-03-11
- Added caching for faster UI response time
- Exposed APIs for the Package Manager UI extension mechanism

## [1.8.2] - 2018-03-02
- Modified Tags to reflect new package workflow (Preview -> Released(no tag) -> Verified)

## [1.8.1] - 2018-02-23
- Removed Recommended tag if package version is alpha, beta or experimental

## [1.8.0] - 2018-02-16
- Removed support built-in packages
- Fixed packages sorting in All tab
- Fixed error reporting with an invalid manifest

## [1.7.2] - 2018-02-08
- Fixed errors when an exception is thrown during an operation
- Changed to only show "View Changes" when there is an update button
- Fixed typos in dialog when updating package manager ui

## [1.7.0] - 2018-02-05
- Added 'View Documentation' link to package details
- Added 'View changes' link to package details

## [1.6.1] - 2018-01-30
### Fixes
- When updating from 1.5.1, ask user to confirm close of window prior to update
- Made window dockable
- Reworked UI styles
- Enhanced keyboard navigation

## [1.5.1] - 2018-01-18
### Fixes
- Replaced VisualContainer by VisualElement in code and templates
- Moved "Project->Packages->Manage" menu item to "Window->Package Manager"
- Showed the latest version and description in tab "Install" instead of the current version
- Added "Recommended" tag properly
- Added "Go back to" when latest version is less than current one
- Removed "Update to" when current version is greater than latest one. (case for embedded or local packages)
- Replaced packages action button label:
	- "Install" instead of "Add" for packages
	- "Enable/Disable" instead of "Add/Remove" for built-in packages
- Added "alpha", "beta", "experimental" and "recommended" tags support
- Added loading progress while opening window
- Added package description and display name update
- Added extra messaging on package state
- Performed Documentation update

## [1.3.0] - 2017-12-12
### Changes
- Added assembly definition files
- Forced SemVer to use .NetStandard
- Fixed ValidationSuiteTests tests
- Handled compatible versions returned in PackageInfo

## [1.2.0] - 2017-11-16
### Fixes
- Fixed flickering test When_Default_FirstPackageUIElement_HasSelectedClass, use package only
- Fixed documentation
- Added Doxygen configuration file
- Removed unused fields in package.json
- Changed 'Modules' for 'Built In Packages'
- Removed version display for Built In Packages

## [1.0.0] - 2017-11-10
### This is the first release of *Unity Package Manager UI*.

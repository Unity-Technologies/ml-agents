## [3.2.2] - 2018-11-02
- Removed FetchOptOutStatus and Initialize call. All application of opt out
  status will be handled by the engine. The Analytics/Data Privacy package still
  provides FetchPrivacyUrl to provide a URL from which to opt out.

## [3.2.1] - 2018-10-25
- Move editor and playmode tests to be packed within the package.

## [3.2.0] - 2018-10-11
- Prevent double-registration of standard events.
- Fixed build error on platforms that don't support analytics.
- Update package docs so they can be built and published and be accessible from
  the Package Manager UI.
- Fixed a crash occurring on iOS device when the device has cellular capability
  but was never configured with any carrier service.
- Fixed an android build failure occurring due to conflicting install referrer
  AIDL files.

## [3.1.1] - 2018-08-21
- Add DataPrivacy plugin into package.
- Fixed an issue where Android project build would fail when proguard is enabled
  in publishing settings.
- Fixed an issue where iOS product archive would fail because bitcode was not
  enabled.

## [3.0.9] - 2018-07-31
- Fixing issue with NullReferenceException during editor playmode

## [3.0.8] - 2018-07-26
- Fixing linking issue when building Android il2cpp

## [3.0.7] - 2018-07-10
- Adding in continuous events for signal strength, battery level, battery
  temperature, memory usage, available storage

## [3.0.6] - 2018-06-01
- Reorganizing platformInfo event around session start/resume/pause

## [3.0.5] - 2018-05-29
- Fixing cellular signal strength incorrect array format

## [3.0.4] - 2018-05-04
- Breaking change to only work with 2018.2 (change name of whitelisted dll's in
  engine to conform to PackageManager standard)
- Changed name of old Analytics dll to the Unity.Analytics.Tracker.dll and
  replaced the old one with the new platform information package.
- Changed naming convention of dlls to the PackageManager Standard:
  Unity.Analytics.dll, Unity.Analytics.Editor.dll, Unity.Analytics.Tracker.dll,
  Unity.Analytics.StandardEvents.dll.
- Deprecated old Analytics tracker and removed it from the add component menu.
- Merged Standardevents package into Analytics package.

## [2.0.14] - 2018-02-08
- Added proper documentation and better description text.

## [2.0.5] -
- Update analytics tracker to 2.0 (1.0 version is still available)

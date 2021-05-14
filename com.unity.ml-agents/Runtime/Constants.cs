using System;

namespace Unity.MLAgents
{
    /// <summary>
    /// Grouping for use in AddComponentMenu (instead of nesting the menus).
    /// </summary>
    internal enum MenuGroup
    {
        Default = 0,
        Sensors = 50
    }

    internal static class PythonTrainerVersions
    {
        // The python package version should be >= s_MinSupportedVersion
        // and <= s_MaxSupportedVersion.
        internal static Version s_MinSupportedVersion = new Version("0.16.1");
        internal static Version s_MaxSupportedVersion = new Version("0.20.0");

        // Any version > to this is known to be incompatible and we will block training.
        // Covers any patch to the release before the 2.0.0 package release.
        internal static Version s_MaxCompatibleVersion = new Version("0.25.999");
    }

}

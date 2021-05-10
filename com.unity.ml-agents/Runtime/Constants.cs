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
        // The python package version must be >= s_MinSupportedPythonPackageVersion
        // and <= s_MaxSupportedPythonPackageVersion.
        internal static Version s_MinSupportedVersion = new Version("0.16.1");
        internal static Version s_MaxSupportedVersion = new Version("0.20.0");
    }

}

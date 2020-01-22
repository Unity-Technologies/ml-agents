namespace Semver
{
    internal static class SemVersionExtension
    {
        public static string VersionOnly(this SemVersion version)
        {
            return "" + version.Major + "." + version.Minor + "." + version.Patch;
        }
        
        public static string ShortVersion(this SemVersion version)
        {
            return version.Major + "." + version.Minor;
        }                
    }
}
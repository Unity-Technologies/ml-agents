using System.Collections.Generic;

namespace UnityEditor.PackageManager.UI
{
    /// <summary>
    /// Package Manager UI Extensions
    /// </summary>
    public static class PackageManagerExtensions
    {
        internal static List<IPackageManagerExtension> Extensions { get { return extensions ?? (extensions = new List<IPackageManagerExtension>()); } }
        private static List<IPackageManagerExtension> extensions;

        /// <summary>
        /// Registers a new Package Manager UI extension
        /// </summary>
        /// <param name="extension">A Package Manager UI extension</param>
        public static void RegisterExtension(IPackageManagerExtension extension)
        {
            if (extension == null)
                return;
            
            Extensions.Add(extension);
        }
    }
}
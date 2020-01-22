using System.Linq;

namespace UnityEditor.PackageManager.UI
{
    internal static class PackageManagerPrefs
    {
        private const string kShowPreviewPackagesPrefKeyPrefix = "PackageManager.ShowPreviewPackages_";
        private const string kShowPreviewPackagesWarningPrefKey = "PackageManager.ShowPreviewPackagesWarning";

        private static string GetProjectIdentifier()
        {
            // PlayerSettings.productGUID is already used as LocalProjectID by Analytics, so we use it too
            return PlayerSettings.productGUID.ToString();
        }

        public static bool ShowPreviewPackages
        {
            get
            {
                var key = kShowPreviewPackagesPrefKeyPrefix + GetProjectIdentifier();

                // If user manually choose to show or not preview packages, use this value
                if (EditorPrefs.HasKey(key))
                    return EditorPrefs.GetBool(key);

                // Returns true if at least one preview package is installed, false otherwise
                return PackageCollection.Instance.LatestListPackages.Any(p => p.IsPreview && p.IsCurrent);
            }
            set
            {
                EditorPrefs.SetBool(kShowPreviewPackagesPrefKeyPrefix + GetProjectIdentifier(), value);
            }
        }

        public static bool ShowPreviewPackagesWarning
        {
            get { return EditorPrefs.GetBool(kShowPreviewPackagesWarningPrefKey, true); }
            set { EditorPrefs.SetBool(kShowPreviewPackagesWarningPrefKey, value); }
        }
    }
}

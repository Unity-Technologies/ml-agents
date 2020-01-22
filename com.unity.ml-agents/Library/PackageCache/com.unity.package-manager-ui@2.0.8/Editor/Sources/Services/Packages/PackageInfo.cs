using System;
using System.Collections.Generic;
using System.Linq;
using Semver;
using System.IO;

namespace UnityEditor.PackageManager.UI
{
    [Serializable]
    internal class PackageInfo : IEquatable<PackageInfo>
    {
        // Module package.json files contain a documentation url embedded in the description.
        // We parse that to have the "View Documentation" button direct to it, instead of showing
        // the link in the description text.
        private const string builtinPackageDocsUrlKey = "Scripting API: ";

        public string Name;
        public string DisplayName;
        private string _PackageId;
        public SemVersion Version;
        public string Description;
        public string Category;
        public PackageState State;
        public bool IsCurrent;
        public bool IsLatest;
        public string Group;
        public PackageSource Origin;
        public List<Error> Errors;
        public bool IsVerified;
        public string Author;

        public PackageManager.PackageInfo Info { get; set; }
        
        public string PackageId {
            get
            {
                if (!string.IsNullOrEmpty(_PackageId )) 
                    return _PackageId;
                return string.Format("{0}@{1}", Name.ToLower(), Version);
            }
            set
            {
                _PackageId = value;
            }
        }

        // This will always be <name>@<version>, even for an embedded package.
        public string VersionId { get { return string.Format("{0}@{1}", Name.ToLower(), Version); } }
        public string ShortVersionId { get { return string.Format("{0}@{1}", Name.ToLower(), Version.ShortVersion()); } }

        public string BuiltInDescription { get {
            if (IsBuiltIn)
                return string.Format("This built in package controls the presence of the {0} module.", DisplayName);
            else
                return Description.Split(new[] {builtinPackageDocsUrlKey}, StringSplitOptions.None)[0];
        } }

        private static Version ParseShortVersion(string shortVersionId)
        {
            try
            {
                var versionToken = shortVersionId.Split('@')[1];
                return new Version(versionToken);
            }
            catch (Exception)
            {
                // Keep default version 0.0 on exception
                return new Version();
            }
        }

        // Method content must be matched in package manager UI
        public static string GetPackageUrlRedirect(string packageName, string shortVersionId)
        {
            var redirectUrl = "";
            if (packageName == "com.unity.ads")
                redirectUrl = "https://docs.unity3d.com/Manual/UnityAds.html";
            else if (packageName == "com.unity.analytics")
            {
                if (ParseShortVersion(shortVersionId) < new Version(3, 2))
                    redirectUrl = "https://docs.unity3d.com/Manual/UnityAnalytics.html";
            }
            else if (packageName == "com.unity.purchasing")
                redirectUrl = "https://docs.unity3d.com/Manual/UnityIAP.html";
            else if (packageName == "com.unity.standardevents")
                redirectUrl = "https://docs.unity3d.com/Manual/UnityAnalyticsStandardEvents.html";
            else if (packageName == "com.unity.xiaomi")
                redirectUrl = "https://unity3d.com/cn/partners/xiaomi/guide";
            else if (packageName == "com.unity.shadergraph")
            {
                if (ParseShortVersion(shortVersionId) < new Version(4, 1))
                    redirectUrl = "https://github.com/Unity-Technologies/ShaderGraph/wiki";
            }

            return redirectUrl;
        }

        public bool RedirectsToManual(PackageInfo packageInfo)
        {
            return !string.IsNullOrEmpty(GetPackageUrlRedirect(packageInfo.Name, packageInfo.ShortVersionId));
        }

        public bool HasChangelog(PackageInfo packageInfo)
        {
            // Packages with no docs have no third party notice
            return !RedirectsToManual(packageInfo);
        }

        public string GetDocumentationUrl()
        {
            if (IsBuiltIn)
            {
                if (!string.IsNullOrEmpty(Description))
                {
                    var split = Description.Split(new[] {builtinPackageDocsUrlKey}, StringSplitOptions.None);
                    if (split.Length > 1)
                        return split[1];
                }
            }
            return string.Format("http://docs.unity3d.com/Packages/{0}/index.html", ShortVersionId);
        }

        public string GetOfflineDocumentationUrl()
        {
            var docsFolder = Path.Combine(Info.resolvedPath, "Documentation~");
            if (!Directory.Exists(docsFolder))
                docsFolder = Path.Combine(Info.resolvedPath, "Documentation");
            if (Directory.Exists(docsFolder))
            {
                var mdFiles = Directory.GetFiles(docsFolder, "*.md", SearchOption.TopDirectoryOnly);
                var docsMd = mdFiles.FirstOrDefault(d => Path.GetFileName(d).ToLower() == "index.md")
                    ?? mdFiles.FirstOrDefault(d => Path.GetFileName(d).ToLower() == "tableofcontents.md") ?? mdFiles.FirstOrDefault();
                if (!string.IsNullOrEmpty(docsMd))
                    return new Uri(docsMd).AbsoluteUri;
            }
            return string.Empty;
        }

        public string GetChangelogUrl()
        {
            return string.Format("http://docs.unity3d.com/Packages/{0}/changelog/CHANGELOG.html", ShortVersionId);
        }

        public string GetOfflineChangelogUrl()
        {
            var changelogFile = Path.Combine(Info.resolvedPath, "CHANGELOG.md");
            return File.Exists(changelogFile) ? new Uri(changelogFile).AbsoluteUri : string.Empty;
        }

        public string GetLicensesUrl()
        {
            var url = string.Format("http://docs.unity3d.com/Packages/{0}/license/index.html", ShortVersionId);
            if (RedirectsToManual(this))
                url = "https://unity3d.com/legal/licenses/Unity_Companion_License";

            return url;
        }

        public string GetOfflineLicensesUrl()
        {
            var licenseFile = Path.Combine(Info.resolvedPath, "LICENSE.md");
            return File.Exists(licenseFile) ? new Uri(licenseFile).AbsoluteUri : string.Empty;
        }

        public bool Equals(PackageInfo other)
        {
            if (other == null) 
                return false;
            if (other == this)
                return true;
            
            return Name == other.Name && Version == other.Version;
        }

        public override int GetHashCode()
        {
            return PackageId.GetHashCode();
        }

        public bool HasVersionTag(string tag)
        {
            if (string.IsNullOrEmpty(Version.Prerelease))
                return false;

            return String.Equals(Version.Prerelease.Split('.').First(), tag, StringComparison.CurrentCultureIgnoreCase);
        }

        public bool HasVersionTag(PackageTag tag)
        {
            return HasVersionTag(tag.ToString());
        }

        // Is it a pre-release (alpha/beta/experimental/preview)?
        //        Current logic is any tag is considered pre-release, except recommended
        public bool IsPreRelease
        {
            get { return !string.IsNullOrEmpty(Version.Prerelease) || Version.Major == 0; }
        }

        public bool IsPreview
        {
            get { return HasVersionTag(PackageTag.preview) || Version.Major == 0; }
        }

        // A version is user visible if it has a supported tag (or no tag at all)
        public bool IsUserVisible
        {
            get { return IsCurrent || string.IsNullOrEmpty(Version.Prerelease) || HasVersionTag(PackageTag.preview) || IsVerified; }
        }

        public bool IsInDevelopment { get { return Origin == PackageSource.Embedded; } }
        public bool IsLocal { get { return Origin == PackageSource.Local; } }
        public bool IsBuiltIn { get { return Origin == PackageSource.BuiltIn; } }
        
        public string VersionWithoutTag { get { return Version.VersionOnly(); } }
        
        public bool IsVersionLocked
        {
            get { return Origin == PackageSource.Embedded || Origin == PackageSource.Git || Origin == PackageSource.BuiltIn; }
        }

        public bool CanBeRemoved
        {
            get { return Origin == PackageSource.Registry || Origin == PackageSource.BuiltIn || Origin == PackageSource.Local; }
        }
    }
}

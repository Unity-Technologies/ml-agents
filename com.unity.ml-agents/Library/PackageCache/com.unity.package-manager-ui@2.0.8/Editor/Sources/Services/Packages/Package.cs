using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;

namespace UnityEditor.PackageManager.UI
{    
    // History of a single package
    internal class Package : IEquatable<Package>
    {
        static public bool ShouldProposeLatestVersions
        {
            get
            {
                // Until we figure out a way to test this properly, alway show standard behavior
                //    return InternalEditorUtility.IsUnityBeta() && !Unsupported.IsDeveloperMode();
                return false;
            }
        }

        // There can only be one package add/remove operation.
        private static IBaseOperation addRemoveOperationInstance;

        public static bool AddRemoveOperationInProgress
        {
            get { return addRemoveOperationInstance != null && !addRemoveOperationInstance.IsCompleted; }
        }

        internal const string packageManagerUIName = "com.unity.package-manager-ui";
        private readonly string packageName;
        private IEnumerable<PackageInfo> source;

        internal Package(string packageName, IEnumerable<PackageInfo> infos)
        {
            if (string.IsNullOrEmpty(packageName))
                throw new ArgumentException("Cannot be empty or null", "packageName");

            if (!infos.Any())
                throw new ArgumentException("Cannot be empty", "infos");
            
            this.packageName = packageName;
            UpdateSource(infos);
        }

        internal void UpdateSource(IEnumerable<PackageInfo> source)
        {
            this.source = source;
#if UNITY_2018_3_OR_NEWER
            if (IsPackageManagerUI)
                this.source = this.source.Where(p => p != null && p.Version.Major >= 2);
#endif
        }

        public PackageInfo Current { get { return Versions.FirstOrDefault(package => package.IsCurrent); } }

        // This is the latest verified or official release (eg: 1.3.2). Not necessarily the latest verified release (eg: 1.2.4) or that latest candidate (eg: 1.4.0-beta)
        public PackageInfo LatestUpdate
        {
            get
            {
                // We want to show the absolute latest when in beta mode
                if (ShouldProposeLatestVersions)
                    return Latest;

                // Override with current when it's version locked
                var current = Current;
                if (current != null && current.IsVersionLocked)
                    return current;

                // Get all the candidates versions (verified, release, preview) that are newer than current
                var verified = Verified;
                var latestRelease = LatestRelease;
                var latestPreview = Versions.LastOrDefault(package => package.IsPreview);
                var candidates = new List<PackageInfo>
                {
                    verified,
                    latestRelease,
                    latestPreview,
                }.Where(package => package != null && (current == null || current == package || current.Version < package.Version)).ToList();

                if (candidates.Contains(verified))
                    return verified;
                if ((current == null || !current.IsVerified ) && candidates.Contains(latestRelease))
                    return latestRelease;
                if ((current == null || current.IsPreview) && candidates.Contains(latestPreview))
                    return latestPreview;

                // Show current if it exists, otherwise latest user visible, and then otherwise show the absolute latest
                return current ?? Latest;
            }
        }

        public PackageInfo LatestPatch
        {
            get
            {
                if (Current == null)
                    return null;
                
                // Get all version that have the same Major/Minor
                var versions = Versions.Where(package => package.Version.Major == Current.Version.Major && package.Version.Minor == Current.Version.Minor);

                return versions.LastOrDefault();
            }
        }

        // This is the very latest version, including pre-releases (eg: 1.4.0-beta).
        internal PackageInfo Latest { get { return Versions.FirstOrDefault(package => package.IsLatest) ?? Versions.LastOrDefault(); } }

        // Returns the current version if it exist, otherwise returns the latest user visible version.
        internal PackageInfo VersionToDisplay { get { return Current ?? LatestUpdate; } }

        // Every version available for this package
        internal IEnumerable<PackageInfo> Versions { get { return source.OrderBy(package => package.Version); } }

        // Every version that's not a pre-release (eg: not beta/alpha/preview).
        internal IEnumerable<PackageInfo> ReleaseVersions
        {
            get { return Versions.Where(package => !package.IsPreRelease); }
        }
        
        internal PackageInfo LatestRelease { get {return ReleaseVersions.LastOrDefault();}}
        internal PackageInfo Verified { get {return Versions.FirstOrDefault(package => package.IsVerified);}}

        internal bool IsAfterCurrentVersion(PackageInfo packageInfo) { return Current == null || (packageInfo != null  && packageInfo.Version > Current.Version); }

        internal bool IsBuiltIn {get { return Versions.Any() && Versions.First().IsBuiltIn; }}

        public string Name { get { return packageName; } }

        public bool IsPackageManagerUI
        {
            get { return Name == packageManagerUIName; }
        }
        
        public bool Equals(Package other)
        {
            if (other == null) 
                return false;
            
            return packageName == other.packageName;
        }

        public override int GetHashCode()
        {
            return packageName.GetHashCode();
        }
        
        [SerializeField]
        internal readonly OperationSignal<IAddOperation> AddSignal = new OperationSignal<IAddOperation>();

        private Action OnAddOperationFinalizedEvent;
        
        internal void Add(PackageInfo packageInfo)
        {
            if (packageInfo == Current || AddRemoveOperationInProgress)
                return;

            var operation = OperationFactory.Instance.CreateAddOperation();
            addRemoveOperationInstance = operation;
            OnAddOperationFinalizedEvent = () =>
            {
                AddSignal.Operation = null;
                operation.OnOperationFinalized -= OnAddOperationFinalizedEvent;
                PackageCollection.Instance.FetchListOfflineCache(true);
            };

            operation.OnOperationFinalized += OnAddOperationFinalizedEvent;

            AddSignal.SetOperation(operation);
            operation.AddPackageAsync(packageInfo);
        }

        internal void Update()
        {
            Add(Latest);
        }

        internal static void AddFromLocalDisk(string path)
        {
            if (AddRemoveOperationInProgress)
                return;

            var packageJson = PackageJsonHelper.Load(path);
            if (null == packageJson)
            {
                Debug.LogError(string.Format("Invalid package path: cannot find \"{0}\".", path));
                return;
            }

            var operation = OperationFactory.Instance.CreateAddOperation();
            addRemoveOperationInstance = operation;
            operation.AddPackageAsync(packageJson.PackageInfo);
        }

        [SerializeField]
        internal readonly OperationSignal<IRemoveOperation> RemoveSignal = new OperationSignal<IRemoveOperation>();

        private Action OnRemoveOperationFinalizedEvent;

        public void Remove()
        {
            if (Current == null || AddRemoveOperationInProgress)
                return;

            var operation = OperationFactory.Instance.CreateRemoveOperation();
            addRemoveOperationInstance = operation;
            OnRemoveOperationFinalizedEvent = () =>
            {
                RemoveSignal.Operation = null;
                operation.OnOperationFinalized -= OnRemoveOperationFinalizedEvent;
                PackageCollection.Instance.FetchListOfflineCache(true);
            };

            operation.OnOperationFinalized += OnRemoveOperationFinalizedEvent;
            RemoveSignal.SetOperation(operation);

            operation.RemovePackageAsync(Current);
        }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace UnityEditor.PackageManager.UI
{
    [Serializable]
    internal class PackageCollection
    {
        private static PackageCollection instance = new PackageCollection();
        public static PackageCollection Instance { get { return instance; } }

        public event Action<IEnumerable<Package>> OnPackagesChanged = delegate { };
        public event Action<PackageFilter> OnFilterChanged = delegate { };

        private readonly Dictionary<string, Package> packages;

        private PackageFilter filter;

        private string selectedListPackage;
        private string selectedSearchPackage;

        internal string lastUpdateTime;
        private List<PackageInfo> listPackagesOffline;
        private List<PackageInfo> listPackages;
        private List<PackageInfo> searchPackages;

        private List<PackageError> packageErrors;

        private int listPackagesVersion;
        private int listPackagesOfflineVersion;

        private bool searchOperationOngoing;
        private bool listOperationOngoing;
        private bool listOperationOfflineOngoing;

        private IListOperation listOperationOffline;
        private IListOperation listOperation;
        private ISearchOperation searchOperation;

        public readonly OperationSignal<ISearchOperation> SearchSignal = new OperationSignal<ISearchOperation>();
        public readonly OperationSignal<IListOperation> ListSignal = new OperationSignal<IListOperation>();

        public static void InitInstance(ref PackageCollection value)
        {
            if (value == null)  // UI window opened
            {
                value = instance;

                Instance.OnPackagesChanged = delegate { };
                Instance.OnFilterChanged = delegate { };
                Instance.SearchSignal.ResetEvents();
                Instance.ListSignal.ResetEvents();

                Instance.FetchListOfflineCache(true);
                Instance.FetchListCache(true);
                Instance.FetchSearchCache(true);
            }
            else // Domain reload
            {
                instance = value;

                Instance.RebuildPackageDictionary();

                // Resume operations interrupted by domain reload
                Instance.FetchListOfflineCache(Instance.listOperationOfflineOngoing);
                Instance.FetchListCache(Instance.listOperationOngoing);
                Instance.FetchSearchCache(Instance.searchOperationOngoing);
            }
        }

        public PackageFilter Filter
        {
            get { return filter; }
            
            // For public usage, use SetFilter() instead
            private set
            {
                var changed = value != filter;
                filter = value;
                
                if (changed)
                    OnFilterChanged(filter);
            }
        }

        public List<PackageInfo> LatestListPackages
        {
            get { return listPackagesVersion > listPackagesOfflineVersion? listPackages : listPackagesOffline; }
        }

        public List<PackageInfo> LatestSearchPackages { get { return searchPackages; } }

        public string SelectedPackage
        {
            get { return PackageFilter.All == Filter ? selectedSearchPackage : selectedListPackage; }
            set
            {
                if (PackageFilter.All == Filter)
                    selectedSearchPackage = value;
                else
                    selectedListPackage = value;
            }
        }
        
        private PackageCollection()
        {
            packages = new Dictionary<string, Package>();

            listPackagesOffline = new List<PackageInfo>();
            listPackages = new List<PackageInfo>();
            searchPackages = new List<PackageInfo>();

            packageErrors = new List<PackageError>();

            listPackagesVersion = 0;
            listPackagesOfflineVersion = 0;

            searchOperationOngoing = false;
            listOperationOngoing = false;
            listOperationOfflineOngoing = false;

            Filter = PackageFilter.All;
        }

        public bool SetFilter(PackageFilter value, bool refresh = true)
        {
            if (value == Filter) 
                return false;
            
            Filter = value;
            if (refresh)
            {
                UpdatePackageCollection();
            }
            return true;
        }

        public void UpdatePackageCollection(bool rebuildDictionary = false)
        {
            if (rebuildDictionary)
            {
                lastUpdateTime = DateTime.Now.ToString("HH:mm");
                RebuildPackageDictionary();
            }
            if (packages.Any())
                OnPackagesChanged(OrderedPackages());
        }

        internal void FetchListOfflineCache(bool forceRefetch = false)
        {
            if (!forceRefetch && (listOperationOfflineOngoing || listPackagesOffline.Any())) return;
            if (listOperationOffline != null)
                listOperationOffline.Cancel();
            listOperationOfflineOngoing = true;
            listOperationOffline = OperationFactory.Instance.CreateListOperation(true);
            listOperationOffline.OnOperationFinalized += () =>
            {
                listOperationOfflineOngoing = false;
                UpdatePackageCollection(true);
            };
            listOperationOffline.GetPackageListAsync(
                infos =>
                {
                    var version = listPackagesVersion;
                    UpdateListPackageInfosOffline(infos, version);
                },
                error => { Debug.LogError("Error fetching package list (offline mode)."); });
        }

        internal void FetchListCache(bool forceRefetch = false)
        {
            if (!forceRefetch && (listOperationOngoing || listPackages.Any())) return;
            if (listOperation != null)
                listOperation.Cancel();
            listOperationOngoing = true;
            listOperation = OperationFactory.Instance.CreateListOperation();
            listOperation.OnOperationFinalized += () =>
            {
                listOperationOngoing = false;
                UpdatePackageCollection(true);
            };
            listOperation.GetPackageListAsync(UpdateListPackageInfos,
                error => { Debug.LogError("Error fetching package list."); });
            ListSignal.SetOperation(listOperation);
        }

        internal void FetchSearchCache(bool forceRefetch = false)
        {
            if (!forceRefetch && (searchOperationOngoing || searchPackages.Any())) return;
            if (searchOperation != null)
                searchOperation.Cancel();
            searchOperationOngoing = true;
            searchOperation = OperationFactory.Instance.CreateSearchOperation();
            searchOperation.OnOperationFinalized += () =>
            {
                searchOperationOngoing = false;
                UpdatePackageCollection(true);
            };
            searchOperation.GetAllPackageAsync(UpdateSearchPackageInfos,
                error => { Debug.LogError("Error searching packages online."); });
            SearchSignal.SetOperation(searchOperation);
        }

        private void UpdateListPackageInfosOffline(IEnumerable<PackageInfo> newInfos, int version)
        {
            listPackagesOfflineVersion = version;
            listPackagesOffline = newInfos.Where(p => p.IsUserVisible).ToList();
        }

        private void UpdateListPackageInfos(IEnumerable<PackageInfo> newInfos)
        {
            // Each time we fetch list packages, the cache for offline mode will be updated
            // We keep track of the list packages version so that we know which version of cache
            // we are getting with the offline fetch operation.
            listPackagesVersion++;
            listPackages = newInfos.Where(p => p.IsUserVisible).ToList();
            listPackagesOffline = listPackages;
        }

        private void UpdateSearchPackageInfos(IEnumerable<PackageInfo> newInfos)
        {
            searchPackages = newInfos.Where(p => p.IsUserVisible).ToList();
        }

        private IEnumerable<Package> OrderedPackages()
        {
            return packages.Values.OrderBy(pkg => pkg.Versions.LastOrDefault() == null ? pkg.Name : pkg.Versions.Last().DisplayName).AsEnumerable();
        }

        public Package GetPackageByName(string name)
        {
            Package package;
            packages.TryGetValue(name, out package);
            return package;
        }

        public Error GetPackageError(Package package)
        {
            if (null == package) return null;
            var firstMatchingError = packageErrors.FirstOrDefault(p => p.PackageName == package.Name);
            return firstMatchingError != null ? firstMatchingError.Error : null;
        }

        public void AddPackageError(Package package, Error error)
        {
            if (null == package || null == error) return;
            packageErrors.Add(new PackageError(package.Name, error));
        }

        public void RemovePackageErrors(Package package)
        {
            if (null == package) return;
            packageErrors.RemoveAll(p => p.PackageName == package.Name);
        }

        private void RebuildPackageDictionary()
        {
            // Merge list & search packages
            var allPackageInfos = new List<PackageInfo>(LatestListPackages);
            var installedPackageIds = new HashSet<string>(allPackageInfos.Select(p => p.PackageId));
            allPackageInfos.AddRange(searchPackages.Where(p => !installedPackageIds.Contains(p.PackageId)));

            if (!PackageManagerPrefs.ShowPreviewPackages)
            {
                allPackageInfos = allPackageInfos.Where(p => !p.IsPreRelease || installedPackageIds.Contains(p.PackageId)).ToList();
            }

            // Rebuild packages dictionary
            packages.Clear();
            foreach (var p in allPackageInfos)
            {
                var packageName = p.Name;
                if (packages.ContainsKey(packageName))
                    continue;

                var packageQuery = from pkg in allPackageInfos where pkg.Name == packageName select pkg;
                var package = new Package(packageName, packageQuery);
                packages[packageName] = package;
            }
        }
    }
}

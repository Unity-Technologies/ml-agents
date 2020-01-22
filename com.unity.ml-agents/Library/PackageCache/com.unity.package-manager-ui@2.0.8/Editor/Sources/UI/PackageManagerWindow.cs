using System.Linq;
using UnityEngine;
using UnityEngine.Experimental.UIElements;
using UnityEditor.Experimental.UIElements;

namespace UnityEditor.PackageManager.UI
{
    internal class PackageManagerWindow : EditorWindow
    {
        public const string PackagePath = "Packages/com.unity.package-manager-ui/";
        public const string ResourcesPath = PackagePath + "Editor/Resources/";
        private const string TemplatePath = ResourcesPath + "Templates/PackageManagerWindow.uxml";
        private const string DarkStylePath = ResourcesPath + "Styles/Main_Dark.uss";
        private const string LightStylePath = ResourcesPath + "Styles/Main_Light.uss";

        private const double targetVersionNumber = 2018.1;

        public PackageCollection Collection;
        public PackageSearchFilter SearchFilter;

#if UNITY_2018_1_OR_NEWER

        public void OnEnable()
        {
            PackageCollection.InitInstance(ref Collection);
            PackageSearchFilter.InitInstance(ref SearchFilter);

            this.GetRootVisualContainer().AddStyleSheetPath(EditorGUIUtility.isProSkin ? DarkStylePath : LightStylePath);

            var windowResource = AssetDatabase.LoadAssetAtPath<VisualTreeAsset>(TemplatePath);
            if (windowResource != null)
            {
                var template = windowResource.CloneTree(null);
                this.GetRootVisualContainer().Add(template);
                template.StretchToParentSize();

                PackageList.OnSelected += OnPackageSelected;
                PackageList.OnLoaded += OnPackagesLoaded;
                PackageList.OnFocusChange += OnListFocusChange;
                
                PackageManagerToolbar.SearchToolbar.OnSearchChange += OnSearchChange;
                PackageManagerToolbar.SearchToolbar.OnFocusChange += OnToolbarFocusChange;

                // Disable filter while fetching first results
                if (!PackageCollection.Instance.LatestListPackages.Any())
                    PackageManagerToolbar.SetEnabled(false);
                else
                    PackageList.SelectLastSelection();
            }
        }

        private void OnListFocusChange()
        {
            PackageManagerToolbar.GrabFocus();
        }

        private void OnToolbarFocusChange()
        {
            PackageList.GrabFocus();
        }

        private void OnSearchChange(string searchText)
        {
            PackageSearchFilter.Instance.SearchText = searchText;
            PackageFiltering.FilterPackageList(PackageList);
        }

        public void OnDisable()
        {
            // Package list item may not be valid here.
            if (PackageList != null)
            {
                PackageList.OnSelected -= OnPackageSelected;
                PackageList.OnLoaded -= OnPackagesLoaded;
            }

            if (PackageManagerToolbar != null)
            {
                PackageManagerToolbar.SearchToolbar.OnSearchChange -= OnSearchChange;
                PackageManagerToolbar.SearchToolbar.OnFocusChange -= OnToolbarFocusChange;
            }
        }
        
        public void OnDestroy()
        {
            PackageSearchFilter.Instance.ResetSearch();
            PackageCollection.Instance.SetFilter(PackageFilter.All, false);
        }

        private void OnPackageSelected(Package package)
        {
            PackageDetails.SetPackage(package);
        }

        private void OnPackagesLoaded()
        {
            PackageManagerToolbar.SetEnabled(true);
        }

        private PackageList PackageList
        {
            get {return this.GetRootVisualContainer().Q<PackageList>("packageList");}
        }

        private PackageDetails PackageDetails
        {
            get {return this.GetRootVisualContainer().Q<PackageDetails>("detailsGroup");}
        }

        private PackageManagerToolbar PackageManagerToolbar
        {
            get {return this.GetRootVisualContainer().Q<PackageManagerToolbar>("toolbarContainer");}
        }

        internal Alert ErrorBanner { get { return this.GetRootVisualContainer().Q<Alert>("errorBanner"); } }
        
#endif

        [MenuItem("Window/Package Manager", priority = 1500)]
        internal static void ShowPackageManagerWindow()
        {
#if UNITY_2018_1_OR_NEWER
            var window = GetWindow<PackageManagerWindow>(false, "Packages", true);
            window.minSize = new Vector2(700, 250);
            window.Show();
#else
            EditorUtility.DisplayDialog("Unsupported Unity Version", string.Format("The Package Manager requires Unity Version {0} or higher to operate.", targetVersionNumber), "Ok");
#endif
        }
    }
}

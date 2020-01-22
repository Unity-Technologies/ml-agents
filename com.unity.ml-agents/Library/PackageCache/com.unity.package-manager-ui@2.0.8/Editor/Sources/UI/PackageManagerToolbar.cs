using System.Linq;
using UnityEngine;
using UnityEngine.Experimental.UIElements;

namespace UnityEditor.PackageManager.UI
{
#if !UNITY_2018_3_OR_NEWER
    internal class PackageManangerToolbarFactory : UxmlFactory<PackageManagerToolbar>
    {
        protected override PackageManagerToolbar DoCreate(IUxmlAttributes bag, CreationContext cc)
        {
            return new PackageManagerToolbar();
        }
    }
#endif

    internal class PackageManagerToolbar : VisualElement
    {
#if UNITY_2018_3_OR_NEWER
        internal new class UxmlFactory : UxmlFactory<PackageManagerToolbar> { }
#endif
        private readonly VisualElement root;

        [SerializeField]
        private PackageFilter selectedFilter = PackageFilter.All;

        public PackageManagerToolbar()
        {
            root = Resources.GetTemplate("PackageManagerToolbar.uxml");
            Add(root);
            root.StretchToParentSize();

            SetFilter(PackageCollection.Instance.Filter);
            
            RegisterCallback<AttachToPanelEvent>(OnEnterPanel);
            RegisterCallback<DetachFromPanelEvent>(OnLeavePanel);
        }

        public void GrabFocus()
        {
            SearchToolbar.GrabFocus();
        }

        public new void SetEnabled(bool enable)
        {
            base.SetEnabled(enable);
            FilterButton.SetEnabled(enable);
            AdvancedButton.SetEnabled(enable);
            SearchToolbar.SetEnabled(enable);
        }

        private static string GetFilterDisplayName(PackageFilter filter)
        {
            switch (filter)
            {
                case PackageFilter.All:
                    return "All packages";
                case PackageFilter.Local:
                    return "In Project";
                case PackageFilter.Modules:
                    return "Built-in packages";
                case PackageFilter.None:
                    return "None";
                default:
                    return filter.ToString();
            }
        }

        private void SetFilter(object obj)
        {
            var previouSelectedFilter = selectedFilter;
            selectedFilter = (PackageFilter) obj;
            FilterButton.text = string.Format("{0} ▾", GetFilterDisplayName(selectedFilter));

            if (selectedFilter != previouSelectedFilter)
                PackageCollection.Instance.SetFilter(selectedFilter);
        }

        private void OnFilterButtonMouseDown(MouseDownEvent evt)
        {
            if (evt.propagationPhase != PropagationPhase.AtTarget)
                return;

            var menu = new GenericMenu();
            menu.AddItem(new GUIContent(GetFilterDisplayName(PackageFilter.All)), 
                selectedFilter == PackageFilter.All, 
                SetFilter, PackageFilter.All);
            menu.AddItem(new GUIContent(GetFilterDisplayName(PackageFilter.Local)), 
                selectedFilter == PackageFilter.Local, 
                SetFilter, PackageFilter.Local);
            menu.AddSeparator("");
            menu.AddItem(new GUIContent(GetFilterDisplayName(PackageFilter.Modules)), 
                selectedFilter == PackageFilter.Modules, 
                SetFilter, PackageFilter.Modules);
            var menuPosition = new Vector2(FilterButton.layout.xMin, FilterButton.layout.center.y);
            menuPosition = this.LocalToWorld(menuPosition);
            var menuRect = new Rect(menuPosition, Vector2.zero);
            menu.DropDown(menuRect);
        }

        private void OnAdvancedButtonMouseDown(MouseDownEvent evt)
        {
            if (evt.propagationPhase != PropagationPhase.AtTarget)
                return;

            var menu = new GenericMenu();

            menu.AddItem(new GUIContent("Show preview packages"), PackageManagerPrefs.ShowPreviewPackages, TogglePreviewPackages);
            //menu.AddItem(new GUIContent("Reset to defaults"), false, ResetToDefaultsClick);

            var menuPosition = new Vector2(AdvancedButton.layout.xMax + 30, AdvancedButton.layout.center.y);
            menuPosition = this.LocalToWorld(menuPosition);
            var menuRect = new Rect(menuPosition, Vector2.zero);
            menu.DropDown(menuRect);
        }

        private static void TogglePreviewPackages()
        {
            var showPreviewPackages = PackageManagerPrefs.ShowPreviewPackages;
            if (!showPreviewPackages && PackageManagerPrefs.ShowPreviewPackagesWarning)
            {
                const string message = "Preview packages are not verified with Unity, may be unstable, and are unsupported in production. Are you sure you want to show preview packages?";
                if (!EditorUtility.DisplayDialog("Unity Package Manager", message, "Yes", "No"))
                    return;
                PackageManagerPrefs.ShowPreviewPackagesWarning = false;
            }
            PackageManagerPrefs.ShowPreviewPackages = !showPreviewPackages;
            PackageCollection.Instance.UpdatePackageCollection(true);
        }

        private void ResetToDefaultsClick()
        {
            if (!EditorUtility.DisplayDialog("Unity Package Manager", "Operation will reset all your packages to Editor defaults. Do you want to continue?", "Yes", "No"))
                return;

            // Registered on callback
            AssemblyReloadEvents.beforeAssemblyReload += PackageManagerWindow.ShowPackageManagerWindow;

            Client.ResetToEditorDefaults();

            var windows = UnityEngine.Resources.FindObjectsOfTypeAll<PackageManagerWindow>();
            if (windows.Length > 0)
            {
                windows[0].Close();
            }
        }

        private void OnEnterPanel(AttachToPanelEvent evt)
        {
            FilterButton.RegisterCallback<MouseDownEvent>(OnFilterButtonMouseDown);
            AdvancedButton.RegisterCallback<MouseDownEvent>(OnAdvancedButtonMouseDown);
        }

        private void OnLeavePanel(DetachFromPanelEvent evt)
        {
            FilterButton.UnregisterCallback<MouseDownEvent>(OnFilterButtonMouseDown);
            AdvancedButton.UnregisterCallback<MouseDownEvent>(OnAdvancedButtonMouseDown);
        }

        private Label FilterButton { get { return root.Q<Label>("toolbarFilterButton"); } }
        private Label AdvancedButton { get { return root.Q<Label>("toolbarAdvancedButton"); } }
        internal PackageSearchToolbar SearchToolbar { get { return root.Q<PackageSearchToolbar>("toolbarSearch"); } }
    }
}
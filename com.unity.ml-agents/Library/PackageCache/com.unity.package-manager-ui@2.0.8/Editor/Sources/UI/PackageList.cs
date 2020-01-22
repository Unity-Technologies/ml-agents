using UnityEngine.Experimental.UIElements;
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace UnityEditor.PackageManager.UI
{
#if !UNITY_2018_3_OR_NEWER
    internal class PackageListFactory : UxmlFactory<PackageList>
    {
        protected override PackageList DoCreate(IUxmlAttributes bag, CreationContext cc)
        {
            return new PackageList();
        }
    }
#endif

    internal class PackageList : VisualElement
    {
#if UNITY_2018_3_OR_NEWER
        internal new class UxmlFactory : UxmlFactory<PackageList> { }
#endif

        public event Action<Package> OnSelected = delegate { };
        public event Action OnLoaded = delegate { };
        public event Action OnFocusChange = delegate { };

        private readonly VisualElement root;
        internal PackageItem selectedItem;
        private List<PackageGroup> Groups;

        public PackageList()
        {
            Groups = new List<PackageGroup>();

            root = Resources.GetTemplate("PackageList.uxml");
            Add(root);
            root.StretchToParentSize();

            UIUtils.SetElementDisplay(Empty, false);
            UIUtils.SetElementDisplay(NoResult, false);

            PackageCollection.Instance.OnPackagesChanged += SetPackages;

            RegisterCallback<AttachToPanelEvent>(OnEnterPanel);
            RegisterCallback<DetachFromPanelEvent>(OnLeavePanel);

            Reload();
        }

        public void GrabFocus()
        {
            if (selectedItem == null)
                return;
            
            selectedItem.Focus();
        }

        public void ShowResults(PackageItem item)
        {
            NoResultText.text = string.Empty;
            UIUtils.SetElementDisplay(NoResult, false);

            Select(item);

            EditorApplication.delayCall += ScrollIfNeeded;

            UpdateGroups();
        }

        public void ShowNoResults()
        {
            NoResultText.text = string.Format("No results for \"{0}\"", PackageSearchFilter.Instance.SearchText);
            UIUtils.SetElementDisplay(NoResult, true);
            foreach (var group in Groups)
            {
                UIUtils.SetElementDisplay(group, false);
            }
            Select(null);
            OnSelected(null);
        }

        private void UpdateGroups()
        {
            foreach (var group in Groups)
            {
                PackageItem firstPackage = null;
                PackageItem lastPackage = null;

                var listGroup = group.Query<PackageItem>().ToList();
                foreach (var item in listGroup)
                {
                    if (!item.visible)
                        continue;

                    if (firstPackage == null) firstPackage = item;
                    lastPackage = item;
                }

                if (firstPackage == null && lastPackage == null)
                {
                    UIUtils.SetElementDisplay(group, false);
                }
                else 
                {
                    UIUtils.SetElementDisplay(group, true);
                    group.firstPackage = firstPackage;
                    group.lastPackage = lastPackage;
                }
            }
        }

        private void OnEnterPanel(AttachToPanelEvent e)
        {
            panel.visualTree.RegisterCallback<KeyDownEvent>(OnKeyDownShortcut);
        }

        private void OnLeavePanel(DetachFromPanelEvent e)
        {
            panel.visualTree.UnregisterCallback<KeyDownEvent>(OnKeyDownShortcut);
        }

        private void ScrollIfNeeded()
        {
            EditorApplication.delayCall -= ScrollIfNeeded;
            
            if (selectedItem == null)
                return;

            var minY = List.worldBound.yMin;
            var maxY = List.worldBound.yMax;
            var itemMinY = selectedItem.worldBound.yMin;
            var itemMaxY = selectedItem.worldBound.yMax;
            var scroll = List.scrollOffset;

            if (itemMinY < minY)
            {
                scroll.y -= minY - itemMinY;
                if (scroll.y <= minY)
                    scroll.y = 0;
                List.scrollOffset = scroll;
            }
            else if (itemMaxY > maxY)
            {
                scroll.y += itemMaxY - maxY;
                List.scrollOffset = scroll;
            }
        }

        private void OnKeyDownShortcut(KeyDownEvent evt)
        {
            if (selectedItem == null)
                return;

            if (evt.keyCode == KeyCode.Tab)
            {
                OnFocusChange();
                evt.StopPropagation();
                return;
            }
            
            if (evt.keyCode == KeyCode.UpArrow)
            {
                if (selectedItem.previousItem != null)
                {
                    Select(selectedItem.previousItem);
                    ScrollIfNeeded();
                }
                else if (selectedItem.packageGroup.previousGroup != null && selectedItem.packageGroup.previousGroup.visible)
                {
                    Select(selectedItem.packageGroup.previousGroup.lastPackage);
                    ScrollIfNeeded();
                }
                evt.StopPropagation();
                return;
            }

            if (evt.keyCode == KeyCode.DownArrow)
            {
                if (selectedItem.nextItem != null)
                {
                    Select(selectedItem.nextItem);
                    ScrollIfNeeded();
                }
                else if (selectedItem.packageGroup.nextGroup != null && selectedItem.packageGroup.nextGroup.visible)
                {
                    Select(selectedItem.packageGroup.nextGroup.firstPackage);
                    ScrollIfNeeded();
                }
                evt.StopPropagation();
                return;
            }

            if (evt.keyCode == KeyCode.PageUp)
            {
                if (selectedItem.packageGroup != null)
                {
                    if (selectedItem == selectedItem.packageGroup.lastPackage && selectedItem != selectedItem.packageGroup.firstPackage)
                    {
                        Select(selectedItem.packageGroup.firstPackage);
                        ScrollIfNeeded();
                    }
                    else if (selectedItem == selectedItem.packageGroup.firstPackage && selectedItem.packageGroup.previousGroup != null && selectedItem.packageGroup.previousGroup.visible)
                    {
                        Select(selectedItem.packageGroup.previousGroup.lastPackage);
                        ScrollIfNeeded();
                    }
                    else if (selectedItem != selectedItem.packageGroup.lastPackage && selectedItem != selectedItem.packageGroup.firstPackage)
                    {
                        Select(selectedItem.packageGroup.firstPackage);
                        ScrollIfNeeded();
                    }
                }
                evt.StopPropagation();
                return;
            }

            if (evt.keyCode == KeyCode.PageDown)
            {
                if (selectedItem.packageGroup != null)
                {
                    if (selectedItem == selectedItem.packageGroup.firstPackage && selectedItem != selectedItem.packageGroup.lastPackage)
                    {
                        Select(selectedItem.packageGroup.lastPackage);
                        ScrollIfNeeded();
                    }
                    else if (selectedItem == selectedItem.packageGroup.lastPackage && selectedItem.packageGroup.nextGroup != null && selectedItem.packageGroup.nextGroup.visible)
                    {
                        Select(selectedItem.packageGroup.nextGroup.firstPackage);
                        ScrollIfNeeded();
                    }
                    else if (selectedItem != selectedItem.packageGroup.firstPackage && selectedItem != selectedItem.packageGroup.lastPackage)
                    {
                        Select(selectedItem.packageGroup.lastPackage);
                        ScrollIfNeeded();
                    }
                }
                evt.StopPropagation();
            }
        }

        private void Reload()
        {
            // Force a re-init to initial condition
            PackageCollection.Instance.UpdatePackageCollection();
            SelectLastSelection();
        }

        private void ClearAll()
        {
            List.Clear();
            Groups.Clear();

            UIUtils.SetElementDisplay(Empty, false);
            UIUtils.SetElementDisplay(NoResult, false);
        }

        private void SetPackages(IEnumerable<Package> packages)
        {
            if (PackageCollection.Instance.Filter == PackageFilter.Modules)
            {
                packages = packages.Where(pkg => pkg.IsBuiltIn);
            }
            else if (PackageCollection.Instance.Filter== PackageFilter.All)
            {
                packages = packages.Where(pkg => !pkg.IsBuiltIn);
            }
            else
            {
                packages = packages.Where(pkg => !pkg.IsBuiltIn);
                packages = packages.Where(pkg => pkg.Current != null);
            }

            OnLoaded();
            ClearAll();

            var packagesGroup = new PackageGroup(PackageGroupOrigins.Packages.ToString());
            Groups.Add(packagesGroup);
            List.Add(packagesGroup);
            packagesGroup.previousGroup = null;

            var builtInGroup = new PackageGroup(PackageGroupOrigins.BuiltInPackages.ToString());
            Groups.Add(builtInGroup);
            List.Add(builtInGroup);

            if ((PackageCollection.Instance.Filter & PackageFilter.Modules) == PackageFilter.Modules)
            {
                packagesGroup.nextGroup = builtInGroup;
                builtInGroup.previousGroup = packagesGroup;
                builtInGroup.nextGroup = null;
            }
            else
            {
                packagesGroup.nextGroup = null;
                UIUtils.SetElementDisplay(builtInGroup, false);
            }

            var lastSelection = PackageCollection.Instance.SelectedPackage;
            Select(null);

            PackageItem defaultSelection = null;

            foreach (var package in packages.OrderBy(pkg => pkg.Versions.FirstOrDefault() == null ? pkg.Name : pkg.Versions.FirstOrDefault().DisplayName))
            {
                var item = AddPackage(package);

                if (null == selectedItem && defaultSelection == null)
                    defaultSelection = item;

                if (null == selectedItem && !string.IsNullOrEmpty(lastSelection) && package.Name.Equals(lastSelection))
                    Select(item);
            }

            if (selectedItem == null)
                Select(defaultSelection);

            PackageFiltering.FilterPackageList(this);
        }

        public void SelectLastSelection()
        {
            var lastSelection = PackageCollection.Instance.SelectedPackage;
            if (lastSelection == null)
                return;
            
            var list = List.Query<PackageItem>().ToList();
            PackageItem defaultSelection = null;

            foreach (var item in list)
            {
                if (defaultSelection == null)
                    defaultSelection = item;

                if (!string.IsNullOrEmpty(lastSelection) && item.package.Name.Equals(lastSelection))
                {
                    defaultSelection = item;
                    break;
                }
            }

            selectedItem = null;
            Select(defaultSelection);
        }

        private PackageItem AddPackage(Package package)
        {
            var groupName = package.Latest != null ? package.Latest.Group : package.Current.Group;
            var group = GetOrCreateGroup(groupName);
            var packageItem = group.AddPackage(package);

            packageItem.OnSelected += Select;

            return packageItem;
        }

        private PackageGroup GetOrCreateGroup(string groupName)
        {
            foreach (var g in Groups)
            {
                if (g.name == groupName)
                    return g;
            }

            var group = new PackageGroup(groupName);
            var latestGroup = Groups.LastOrDefault();
            Groups.Add(group);
            List.Add(group);

            group.previousGroup = null;
            if (latestGroup != null)
            {
                latestGroup.nextGroup = group;
                group.previousGroup = latestGroup;
                group.nextGroup = null;
            }
            return group;
        }

        private void Select(PackageItem packageItem)
        {
            if (packageItem == selectedItem)
                return;

            var selectedPackageName = packageItem != null ? packageItem.package.Name : null;
            PackageCollection.Instance.SelectedPackage = selectedPackageName;

            if (selectedItem != null)
                selectedItem.SetSelected(false); // Clear Previous selection

            selectedItem = packageItem;
            if (selectedItem == null)
            {
                OnSelected(null);
                return;
            }

            selectedItem.SetSelected(true);
            ScrollIfNeeded();
            OnSelected(selectedItem.package);
        }

        private ScrollView List { get { return root.Q<ScrollView>("scrollView"); } }
        private VisualElement Empty { get { return root.Q<VisualElement>("emptyArea"); } }
        private VisualElement NoResult { get { return root.Q<VisualElement>("noResult"); } }
        private Label NoResultText { get { return root.Q<Label>("noResultText"); } }
    }
}

using System.Linq;
using UnityEngine.Experimental.UIElements;

namespace UnityEditor.PackageManager.UI
{
#if !UNITY_2018_3_OR_NEWER
    internal class PackageGroupFactory : UxmlFactory<PackageGroup>
    {
        protected override PackageGroup DoCreate(IUxmlAttributes bag, CreationContext cc)
        {
            return new PackageGroup(bag.GetPropertyString("name"));
        }
    }
#endif

    internal class PackageGroup : VisualElement
    {
#if UNITY_2018_3_OR_NEWER
        internal new class UxmlFactory : UxmlFactory<PackageGroup> { }
#endif

        private readonly VisualElement root;
        internal readonly PackageGroupOrigins Origin;

        public PackageGroup previousGroup;
        public PackageGroup nextGroup;

        public PackageItem firstPackage;
        public PackageItem lastPackage;

        public PackageGroup() : this(string.Empty)
        {
        }

        public PackageGroup(string groupName)
        {
            name = groupName;
            root = Resources.GetTemplate("PackageGroup.uxml");
            Add(root);

            if (string.IsNullOrEmpty(groupName) || groupName != PackageGroupOrigins.BuiltInPackages.ToString())
            {
                HeaderTitle.text = "Packages";
                Origin = PackageGroupOrigins.Packages;
            }
            else
            {
                HeaderTitle.text = "Built In Packages";
                Origin = PackageGroupOrigins.BuiltInPackages;
            }
        }

        internal PackageItem AddPackage(Package package)
        {
            var packageItem = new PackageItem(package) {packageGroup = this};
            var lastItem = List.Children().LastOrDefault() as PackageItem;
            if (lastItem != null)
            {
                lastItem.nextItem = packageItem;
                packageItem.previousItem = lastItem;
                packageItem.nextItem = null;
            }

            List.Add(packageItem);

            if (firstPackage == null) firstPackage = packageItem;
            lastPackage = packageItem;

            return packageItem;
        }

        private VisualElement List { get { return root.Q<VisualElement>("groupContainer"); } }
        private Label HeaderTitle { get { return root.Q<Label>("headerTitle"); } }
    }
}

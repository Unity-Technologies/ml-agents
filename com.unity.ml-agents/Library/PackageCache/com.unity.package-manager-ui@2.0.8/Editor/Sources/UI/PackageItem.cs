using UnityEngine.Experimental.UIElements;
using System;

namespace UnityEditor.PackageManager.UI
{
#if !UNITY_2018_3_OR_NEWER
    internal class PackageItemFactory : UxmlFactory<PackageItem>
    {
        protected override PackageItem DoCreate(IUxmlAttributes bag, CreationContext cc)
        {
            return new PackageItem();
        }
    }
#endif

    internal class PackageItem : VisualElement
    {
#if UNITY_2018_3_OR_NEWER
        internal new class UxmlFactory : UxmlFactory<PackageItem> { }
#endif

        public static string SelectedClassName = "selected";

        public event Action<PackageItem> OnSelected = delegate { };

        private readonly VisualElement root;
        private string currentStateClass;
        public Package package { get; private set; }

        public PackageItem previousItem;
        public PackageItem nextItem;

        public PackageGroup packageGroup;

        public PackageItem() : this(null)
        {
        }

        public PackageItem(Package package)
        {
            root = Resources.GetTemplate("PackageItem.uxml");
            Add(root);

            root.AddManipulator(new Clickable(Select));
            SetItem(package);
        }

        private void Select()
        {
            OnSelected(this);
        }

        public void SetSelected(bool value)
        {
            if (value)
                PackageContainer.AddToClassList(SelectedClassName);
            else
                PackageContainer.RemoveFromClassList(SelectedClassName);

            Spinner.InvertColor = value;
        }

        private void SetItem(Package package)
        {
            var displayPackage = package != null ? package.VersionToDisplay : null;
            if (displayPackage == null)
                return;
            
            this.package = package;
            OnPackageChanged();
            this.package.AddSignal.WhenOperation(OnPackageAdd);
            this.package.RemoveSignal.WhenOperation(OnPackageRemove);
        }

        private void OnPackageRemove(IRemoveOperation operation)
        {
            operation.OnOperationError += error => StopSpinner();
            OnPackageUpdate();
        }

        private void OnPackageAdd(IAddOperation operation)
        {
            operation.OnOperationError += error => StopSpinner();
            OnPackageUpdate();
        }

        private void OnPackageChanged()
        {
            var displayPackage = package != null ? package.VersionToDisplay : null;
            if (displayPackage == null)
                return;

            NameLabel.text = displayPackage.DisplayName;
            VersionLabel.text = displayPackage.Version.ToString();

            var stateClass = GetIconStateId(displayPackage);
            if (displayPackage.State == PackageState.Outdated && package.LatestUpdate == package.Current)
                stateClass = GetIconStateId(PackageState.UpToDate);
            if (PackageCollection.Instance.GetPackageError(package) != null)
                stateClass = GetIconStateId(PackageState.Error);
            if (stateClass ==  GetIconStateId(PackageState.UpToDate) && package.Current != null)
                stateClass = "installed";

            StateLabel.RemoveFromClassList(currentStateClass);
            StateLabel.AddToClassList(stateClass);

            UIUtils.SetElementDisplay(VersionLabel, !displayPackage.IsBuiltIn);

            currentStateClass = stateClass;
            if (displayPackage.State != PackageState.InProgress && Spinner.Started)
                StopSpinner();
        }

        private void OnPackageUpdate()
        {
            StartSpinner();
        }

        private void StartSpinner()
        {
            Spinner.Start();
            StateLabel.AddToClassList("no-icon");
        }
        
        private void StopSpinner()
        {
            Spinner.Stop();
            StateLabel.RemoveFromClassList("no-icon");
        }

        private Label NameLabel { get { return root.Q<Label>("packageName"); } }
        private Label StateLabel { get { return root.Q<Label>("packageState"); } }
        private Label VersionLabel { get { return root.Q<Label>("packageVersion"); } }
        private VisualElement PackageContainer { get { return root.Q<VisualElement>("packageContainer"); } }
        private LoadingSpinner Spinner { get { return root.Q<LoadingSpinner>("packageSpinner"); } }

        public static string GetIconStateId(PackageInfo packageInfo)
        {
            if (packageInfo == null)
                return "";

            return GetIconStateId(packageInfo.State);
        }

        public static string GetIconStateId(PackageState state)
        {
            return state.ToString().ToLower();
        }
    }
}
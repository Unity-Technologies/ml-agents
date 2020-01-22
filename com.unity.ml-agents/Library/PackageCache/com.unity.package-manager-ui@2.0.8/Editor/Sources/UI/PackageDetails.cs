using System;
using System.Collections.Generic;
using System.Linq;
using Semver;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.Experimental.UIElements;

namespace UnityEditor.PackageManager.UI
{
#if !UNITY_2018_3_OR_NEWER
    internal class PackageDetailsFactory : UxmlFactory<PackageDetails>
    {
        protected override PackageDetails DoCreate(IUxmlAttributes bag, CreationContext cc)
        {
            return new PackageDetails();
        }
    }
#endif

    internal class PackageDetails : VisualElement
    {
#if UNITY_2018_3_OR_NEWER
        internal new class UxmlFactory : UxmlFactory<PackageDetails> { }
#endif

        private readonly VisualElement root;
        private Package package;
        private const string emptyDescriptionClass = "empty";
        private List<VersionItem> VersionItems;
        internal PopupField<VersionItem> VersionPopup;
        private PackageInfo DisplayPackage;

        private PackageInfo SelectedPackage
        {
            get { return VersionPopup.value.Version != null ? VersionPopup.value.Version : null; }
        }

        internal enum PackageAction
        {
            Add,
            Remove,
            Update,
            Downgrade,
            Enable,
            Disable,
            UpToDate,
            Current,
            Local,
            Git,
            Embedded
        }

        private static readonly VersionItem EmptyVersion = new VersionItem {Version = null};
        internal static readonly string[] PackageActionVerbs = { "Install", "Remove", "Update to", "Update to",  "Enable", "Disable", "Up to date", "Current", "Local", "Git", "Embedded" };
        internal static readonly string[] PackageActionInProgressVerbs = { "Installing", "Removing", "Updating to", "Updating to", "Enabling", "Disabling", "Up to date", "Current", "Local", "Git", "Embedded" };

        public PackageDetails()
        {
            UIUtils.SetElementDisplay(this, false);

            root = Resources.GetTemplate("PackageDetails.uxml");
            Add(root);

            foreach (var extension in PackageManagerExtensions.Extensions)
                CustomContainer.Add(extension.CreateExtensionUI());

            root.StretchToParentSize();

            SetUpdateVisibility(false);
            RemoveButton.visible = false;
            UpdateBuiltIn.visible = false;

            UpdateButton.clickable.clicked += UpdateClick;
            UpdateBuiltIn.clickable.clicked += UpdateClick;
            RemoveButton.clickable.clicked += RemoveClick;
            ViewDocButton.clickable.clicked += ViewDocClick;
            ViewChangelogButton.clickable.clicked += ViewChangelogClick;
            ViewLicenses.clickable.clicked += ViewLicensesClick;

            VersionItems = new List<VersionItem> {EmptyVersion};
            VersionPopup = new PopupField<VersionItem>(VersionItems, 0);
            VersionPopup.SetLabelCallback(VersionSelectionSetLabel);
            VersionPopup.AddToClassList("popup");
            VersionPopup.OnValueChanged(VersionSelectionChanged);
            
            if (VersionItems.Count == 1)
                VersionPopup.SetEnabled(false);
                        
            UpdateDropdownContainer.Add(VersionPopup);
            VersionPopup.StretchToParentSize();
            

            // Fix button on dark skin but overlapping edge pixel perfectly
            if (EditorGUIUtility.isProSkin)
            {
                VersionPopup.style.positionLeft = -1;
                UpdateDropdownContainer.style.sliceLeft = 4;
            }
        }

        private string VersionSelectionSetLabel(VersionItem item)
        {
            return item.Label;
        }

        private void VersionSelectionChanged(ChangeEvent<VersionItem> e)
        {
            RefreshAddButton();
        }

        private void SetUpdateVisibility(bool value)
        {
            if (UpdateContainer != null)
                UIUtils.SetElementDisplay(UpdateContainer, value);
        }

        internal void SetDisplayPackage(PackageInfo packageInfo)
        {
            DisplayPackage = packageInfo;
            
            var detailVisible = true;
            Error error = null;

            if (package == null || DisplayPackage == null)
            {
                detailVisible = false;
                UIUtils.SetElementDisplay(DocumentationContainer, false);
                UIUtils.SetElementDisplay(CustomContainer, false);
                UIUtils.SetElementDisplay(UpdateBuiltIn, false);

                foreach (var extension in PackageManagerExtensions.Extensions)
                    extension.OnPackageSelectionChange(null);
            }
            else
            {
                SetUpdateVisibility(true);
                UIUtils.SetElementDisplay(ViewDocButton, true);
                RemoveButton.visible = true;

                if (string.IsNullOrEmpty(DisplayPackage.Description))
                {
                    DetailDesc.text = "There is no description for this package.";
                    DetailDesc.AddToClassList(emptyDescriptionClass);
                }
                else
                {
                    DetailDesc.text = DisplayPackage.Description;
                    DetailDesc.RemoveFromClassList(emptyDescriptionClass);
                }

                root.Q<Label>("detailTitle").text = DisplayPackage.DisplayName;
                DetailVersion.text = "Version " + DisplayPackage.VersionWithoutTag;

                if (DisplayPackage.IsInDevelopment || DisplayPackage.HasVersionTag(PackageTag.preview))
                    UIUtils.SetElementDisplay(GetTag(PackageTag.verified), false);
                else
                {
                    var unityVersionParts = Application.unityVersion.Split('.');
                    var unityVersion = string.Format("{0}.{1}", unityVersionParts[0], unityVersionParts[1]);
                    VerifyLabel.text = unityVersion + " verified";
                    UIUtils.SetElementDisplay(GetTag(PackageTag.verified), DisplayPackage.IsVerified);
                }

                UIUtils.SetElementDisplay(GetTag(PackageTag.inDevelopment), DisplayPackage.IsInDevelopment);
                UIUtils.SetElementDisplay(GetTag(PackageTag.local), DisplayPackage.IsLocal);
                UIUtils.SetElementDisplay(GetTag(PackageTag.preview), DisplayPackage.IsPreview);

                UIUtils.SetElementDisplay(DocumentationContainer, DisplayPackage.Origin != PackageSource.BuiltIn);
                UIUtils.SetElementDisplay(ChangelogContainer, DisplayPackage.HasChangelog(DisplayPackage));

                root.Q<Label>("detailName").text = DisplayPackage.Name;
                root.Q<ScrollView>("detailView").scrollOffset = new Vector2(0, 0);

                DetailModuleReference.text = "";
                var isBuiltIn = DisplayPackage.IsBuiltIn;
                if (isBuiltIn)
                    DetailModuleReference.text = DisplayPackage.BuiltInDescription;

                DetailAuthor.text = "";
                if (!string.IsNullOrEmpty(DisplayPackage.Author))
                    DetailAuthor.text = string.Format("Author: {0}", DisplayPackage.Author);

                UIUtils.SetElementDisplay(DetailDesc, !isBuiltIn);
                UIUtils.SetElementDisplay(DetailVersion, !isBuiltIn);
                UIUtils.SetElementDisplayNonEmpty(DetailModuleReference);
                UIUtils.SetElementDisplayNonEmpty(DetailAuthor);

                if (DisplayPackage.Errors.Count > 0)
                    error = DisplayPackage.Errors.First();

                RefreshAddButton();
                RefreshRemoveButton();
                UIUtils.SetElementDisplay(CustomContainer, true);

                package.AddSignal.OnOperation += OnAddOperation;
                package.RemoveSignal.OnOperation += OnRemoveOperation;
                foreach (var extension in PackageManagerExtensions.Extensions)
                    extension.OnPackageSelectionChange(DisplayPackage.Info);
            }

            // Set visibility
            root.Q<VisualElement>("detail").visible = detailVisible;

            if (null == error)
                error = PackageCollection.Instance.GetPackageError(package);

            if (error != null)
                SetError(error);
            else
                DetailError.ClearError();            
        }

        private void ResetVersionItems(PackageInfo displayPackage)
        {
            VersionItems.Clear();            
            VersionPopup.SetEnabled(true);

            if (displayPackage == null)
                return;
            
            //
            // Get key versions -- Latest, Verified, LatestPatch, Current.
            var keyVersions = new List<PackageInfo>();
            if (package.LatestRelease != null) keyVersions.Add(package.LatestRelease);
            if (package.Current != null) keyVersions.Add(package.Current);
            if (package.Verified != null && package.Verified != package.Current) keyVersions.Add(package.Verified);
            if (package.LatestPatch != null && package.IsAfterCurrentVersion(package.LatestPatch)) keyVersions.Add(package.LatestPatch);
            if (package.Current == null && package.LatestRelease == null && package.Latest != null) keyVersions.Add(package.Latest);
            if (Package.ShouldProposeLatestVersions && package.Latest != package.LatestRelease && package.Latest != null) keyVersions.Add(package.Latest);
            keyVersions.Add(package.LatestUpdate);        // Make sure LatestUpdate is always in the list.

            foreach (var version in keyVersions.OrderBy(package => package.Version).Reverse())
            {
                var item = new VersionItem {Version = version};
                VersionItems.Add(item);
                
                if (version == package.LatestUpdate)
                    VersionPopup.value = item;
            }

            //
            // Add all versions
            foreach (var version in package.Versions.Reverse())
            {
                var item = new VersionItem {Version = version};
                item.MenuName = "All Versions/";
                VersionItems.Add(item);
            }
            
            if (VersionItems.Count == 0)
            {
                VersionItems.Add(EmptyVersion);
                VersionPopup.value = EmptyVersion;
                VersionPopup.SetEnabled(false);
            }
        }
        
        public void SetPackage(Package package)
        {
            if (this.package != null)
            {
                if (this.package.AddSignal.Operation != null)
                {
                    this.package.AddSignal.Operation.OnOperationError -= OnAddOperationError;
                    this.package.AddSignal.Operation.OnOperationSuccess -= OnAddOperationSuccess;
                }
                this.package.AddSignal.ResetEvents();

                if (this.package.RemoveSignal.Operation != null)
                {
                    this.package.RemoveSignal.Operation.OnOperationSuccess -= OnRemoveOperationSuccess;
                    this.package.RemoveSignal.Operation.OnOperationError -= OnRemoveOperationError;
                }
                this.package.RemoveSignal.ResetEvents();
            }

            UIUtils.SetElementDisplay(this, true);

            this.package = package;
            var displayPackage = package != null ? package.VersionToDisplay : null;
            ResetVersionItems(displayPackage);
            SetDisplayPackage(displayPackage);
        }

        private void SetError(Error error)
        {
            DetailError.AdjustSize(DetailView.verticalScroller.visible);
            DetailError.SetError(error);
            DetailError.OnCloseError = () =>
            {
                PackageCollection.Instance.RemovePackageErrors(package);
                PackageCollection.Instance.UpdatePackageCollection();
            };
        }

        private void OnAddOperation(IAddOperation operation)
        {
            operation.OnOperationError += OnAddOperationError;
            operation.OnOperationSuccess += OnAddOperationSuccess;
        }

        private void OnAddOperationError(Error error)
        {
            if (package != null && package.AddSignal.Operation != null)
            {
                package.AddSignal.Operation.OnOperationSuccess -= OnAddOperationSuccess;
                package.AddSignal.Operation.OnOperationError -= OnAddOperationError;
                package.AddSignal.Operation = null;
            }

            PackageCollection.Instance.AddPackageError(package, error);

            SetError(error);
            if (package != null)
                ResetVersionItems(package.VersionToDisplay);
            PackageCollection.Instance.UpdatePackageCollection();
        }

        private void OnAddOperationSuccess(PackageInfo packageInfo)
        {
            if (package != null && package.AddSignal.Operation != null)
            {
                package.AddSignal.Operation.OnOperationSuccess -= OnAddOperationSuccess;
                package.AddSignal.Operation.OnOperationError -= OnAddOperationError;
                package.AddSignal.Operation = null;
            }

            foreach (var extension in PackageManagerExtensions.Extensions)
                extension.OnPackageAddedOrUpdated(packageInfo.Info);
        }

        private void OnRemoveOperation(IRemoveOperation operation)
        {
            // Make sure we are not already registered
            operation.OnOperationError -= OnRemoveOperationError;
            operation.OnOperationSuccess -= OnRemoveOperationSuccess;
            
            operation.OnOperationError += OnRemoveOperationError;
            operation.OnOperationSuccess += OnRemoveOperationSuccess;
        }

        private void OnRemoveOperationError(Error error)
        {
            if (package != null && package.RemoveSignal.Operation != null)
            {
                package.RemoveSignal.Operation.OnOperationSuccess -= OnRemoveOperationSuccess;
                package.RemoveSignal.Operation.OnOperationError -= OnRemoveOperationError;
                package.RemoveSignal.Operation = null;
            }

            PackageCollection.Instance.AddPackageError(package, error);

            SetError(error);
            PackageCollection.Instance.UpdatePackageCollection();
        }

        private void OnRemoveOperationSuccess(PackageInfo packageInfo)
        {
            if (package != null && package.RemoveSignal.Operation != null)
            {
                package.RemoveSignal.Operation.OnOperationSuccess -= OnRemoveOperationSuccess;
                package.RemoveSignal.Operation.OnOperationError -= OnRemoveOperationError;
                package.RemoveSignal.Operation = null;
            }

            foreach (var extension in PackageManagerExtensions.Extensions)
                extension.OnPackageRemoved(packageInfo.Info);
        }

        private void RefreshAddButton()
        {
            if (package.Current != null && package.Current.IsInDevelopment)
            {
                UIUtils.SetElementDisplay(UpdateBuiltIn, false);
                UIUtils.SetElementDisplay(UpdateCombo, false);
                UIUtils.SetElementDisplay(UpdateButton, false);
                return;
            }

            var targetVersion = SelectedPackage;
            if (targetVersion == null)
                return;
            
            var enableButton = !Package.AddRemoveOperationInProgress;
            var enableVersionButton = true;
            
            var action = PackageAction.Update;
            var inprogress = false;
            var isBuiltIn = package.IsBuiltIn;
            SemVersion version = null;
            
            if (package.AddSignal.Operation != null)
            {
                if (isBuiltIn)
                {
                    action = PackageAction.Enable;
                    inprogress = true;
                    enableButton = false;                    
                }
                else
                {
                    var addOperationVersion = package.AddSignal.Operation.PackageInfo.Version;
                    if (package.Current == null)
                    {
                        action = PackageAction.Add;
                        inprogress = true;
                    }
                    else
                    {
                        action = addOperationVersion.CompareByPrecedence(package.Current.Version) >= 0
                            ? PackageAction.Update : PackageAction.Downgrade;
                        inprogress = true;
                    }
                
                    enableButton = false;
                    enableVersionButton = false;
                }
            } 
            else 
            {
                if (package.Current != null)
                {
                    // Installed
                    if (package.Current.IsVersionLocked)
                    {
                        if (package.Current.Origin == PackageSource.Embedded)
                            action = PackageAction.Embedded;
                        else if (package.Current.Origin == PackageSource.Git)
                            action = PackageAction.Git;
                        
                        enableButton = false;
                        enableVersionButton = false;
                    }
                    else
                    {
                        if (targetVersion.IsCurrent)
                        {
                            if (targetVersion == package.LatestUpdate)
                                action = PackageAction.UpToDate;
                            else
                                action = PackageAction.Current;
                            
                            enableButton = false;
                        }
                        else
                        {
                            action = targetVersion.Version.CompareByPrecedence(package.Current.Version) >= 0
                                ? PackageAction.Update : PackageAction.Downgrade;
                        }
                    }
                }
                else
                {
                    // Not Installed
                    if (package.Versions.Any())
                    {
                        if (isBuiltIn)
                            action = PackageAction.Enable;
                        else
                            action = PackageAction.Add;
                    }
                }
            }

            if (package.RemoveSignal.Operation != null)
                enableButton = false;

            if (EditorApplication.isCompiling)
            {
                enableButton = false;
                enableVersionButton = false;

                EditorApplication.update -= CheckCompilationStatus;
                EditorApplication.update += CheckCompilationStatus;
            }
            
            var button = isBuiltIn ? UpdateBuiltIn : UpdateButton;
            button.SetEnabled(enableButton);
            VersionPopup.SetEnabled(enableVersionButton);
            button.text = GetButtonText(action, inprogress, version);

            var visibleFlag = !(package.Current != null && package.Current.IsVersionLocked);
            UIUtils.SetElementDisplay(UpdateBuiltIn, isBuiltIn && visibleFlag);
            UIUtils.SetElementDisplay(UpdateCombo, !isBuiltIn && visibleFlag);
            UIUtils.SetElementDisplay(UpdateButton, !isBuiltIn && visibleFlag);
        }

        private void RefreshRemoveButton()
        {
            var visibleFlag = false;

            var current = package.Current;
            
            // Show only if there is a current package installed
            if (current != null)
            {
                visibleFlag = current.CanBeRemoved && !package.IsPackageManagerUI;

                var action = current.IsBuiltIn ? PackageAction.Disable : PackageAction.Remove;
                var inprogress = package.RemoveSignal.Operation != null;

                var enableButton = visibleFlag && !EditorApplication.isCompiling && !inprogress && !Package.AddRemoveOperationInProgress;

                if (EditorApplication.isCompiling)
                {
                    EditorApplication.update -= CheckCompilationStatus;
                    EditorApplication.update += CheckCompilationStatus;
                }

                RemoveButton.SetEnabled(enableButton);
                RemoveButton.text = GetButtonText(action, inprogress);                   
            }

            UIUtils.SetElementDisplay(RemoveButton, visibleFlag);
        }

        private void CheckCompilationStatus()
        {
            if (EditorApplication.isCompiling)
                return;

            RefreshAddButton();
            RefreshRemoveButton();
            EditorApplication.update -= CheckCompilationStatus;
        }

        private static string GetButtonText(PackageAction action, bool inProgress = false, SemVersion version = null)
        {
            return version == null ?
                string.Format("{0}", inProgress ? PackageActionInProgressVerbs[(int) action] : PackageActionVerbs[(int) action]) :
                string.Format("{0} {1}", inProgress ? PackageActionInProgressVerbs[(int) action] : PackageActionVerbs[(int) action], version);
        }

        private void UpdateClick()
        {
            if (package.IsPackageManagerUI)
            {
                // Let's not allow updating of the UI if there are build errrors, as for now, that will prevent the UI from reloading properly.
                if (EditorUtility.scriptCompilationFailed)
                {
                    EditorUtility.DisplayDialog("Unity Package Manager", "The Package Manager UI cannot be updated while there are script compilation errors in your project.  Please fix the errors and try again.", "Ok");
                    return;
                }

                if (!EditorUtility.DisplayDialog("Unity Package Manager", "Updating this package will close the Package Manager window. You will have to re-open it after the update is done. Do you want to continue?", "Yes", "No"))
                    return;

                if (package.AddSignal.Operation != null)
                {
                    package.AddSignal.Operation.OnOperationSuccess -= OnAddOperationSuccess;
                    package.AddSignal.Operation.OnOperationError -= OnAddOperationError;
                    package.AddSignal.ResetEvents();
                    package.AddSignal.Operation = null;
                }

                DetailError.ClearError();
                EditorApplication.update += CloseAndUpdate;

                return;
            }

            DetailError.ClearError();
            package.Add(SelectedPackage);
            RefreshAddButton();
            RefreshRemoveButton();
        }

        private void CloseAndUpdate()
        {
            EditorApplication.update -= CloseAndUpdate;
            package.Add(SelectedPackage);

            var windows = UnityEngine.Resources.FindObjectsOfTypeAll<PackageManagerWindow>();
            if (windows.Length > 0)
            {
                windows[0].Close();
            }
        }


        private void RemoveClick()
        {
            DetailError.ClearError();
            package.Remove();
            RefreshRemoveButton();
            RefreshAddButton();
        }

        private static void ViewOfflineUrl(Func<string> getOfflineUrl, string messageOnNotFound)
        {
            var offlineUrl = getOfflineUrl();
            if (!string.IsNullOrEmpty(offlineUrl))
                Application.OpenURL(offlineUrl);
            else
                EditorUtility.DisplayDialog("Unity Package Manager", messageOnNotFound, "Ok");
        }

        private static void ViewUrl(Func<string> getUrl, Func<string> getOfflineUrl, string messageOnNotFound)
        {
            if (Application.internetReachability != NetworkReachability.NotReachable)
            {
                var onlineUrl = getUrl();
                var request = UnityWebRequest.Head(onlineUrl);
                var operation = request.SendWebRequest();
                operation.completed += (op) =>
                {
                    if (request.responseCode != 404)
                    {
                        Application.OpenURL(onlineUrl);
                    }
                    else
                    {
                        ViewOfflineUrl(getOfflineUrl, messageOnNotFound);
                    }
                };
            }
            else

            {
                ViewOfflineUrl(getOfflineUrl, messageOnNotFound);
            }
        }

        private void ViewDocClick()
        {
            ViewUrl(DisplayPackage.GetDocumentationUrl, DisplayPackage.GetOfflineDocumentationUrl, "Unable to find documentation.");
        } 

        private void ViewChangelogClick()
        {
            ViewUrl(DisplayPackage.GetChangelogUrl, DisplayPackage.GetOfflineChangelogUrl, "Unable to find changelog.");
        }

        private void ViewLicensesClick()
        {    
            ViewUrl(DisplayPackage.GetLicensesUrl, DisplayPackage.GetOfflineLicensesUrl, "Unable to find licenses.");
        }
        
        private Label DetailDesc { get { return root.Q<Label>("detailDesc"); } }
        internal Button UpdateButton { get { return root.Q<Button>("update"); } }
        private Button RemoveButton { get { return root.Q<Button>("remove"); } }
        private Button ViewDocButton { get { return root.Q<Button>("viewDocumentation"); } }
        private VisualElement DocumentationContainer { get { return root.Q<VisualElement>("documentationContainer"); } }
        private Button ViewChangelogButton { get { return root.Q<Button>("viewChangelog"); } }
        private VisualElement ChangelogContainer { get { return root.Q<VisualElement>("changeLogContainer"); } }
        private VisualElement ViewLicensesContainer { get { return root.Q<VisualElement>("viewLicensesContainer"); } }
        private Button ViewLicenses { get { return root.Q<Button>("viewLicenses"); } }        
        private VisualElement UpdateContainer { get { return root.Q<VisualElement>("updateContainer"); } }
        private Alert DetailError { get { return root.Q<Alert>("detailError"); } }
        private ScrollView DetailView { get { return root.Q<ScrollView>("detailView"); } }
        private Label DetailModuleReference { get { return root.Q<Label>("detailModuleReference"); } }
        private Label DetailVersion { get { return root.Q<Label>("detailVersion");  }}
        private Label DetailAuthor { get { return root.Q<Label>("detailAuthor");  }}
        private Label VerifyLabel { get { return root.Q<Label>("tagVerify"); } }
        private VisualElement CustomContainer { get { return root.Q<VisualElement>("detailCustomContainer"); } }
        internal VisualElement GetTag(PackageTag tag) {return root.Q<VisualElement>("tag-" + tag); }
        private VisualElement UpdateDropdownContainer { get { return root.Q<VisualElement>("updateDropdownContainer"); } }
        internal VisualElement UpdateCombo { get { return root.Q<VisualElement>("updateCombo"); } }
        internal Button UpdateBuiltIn { get { return root.Q<Button>("updateBuiltIn"); } }        
    }
}

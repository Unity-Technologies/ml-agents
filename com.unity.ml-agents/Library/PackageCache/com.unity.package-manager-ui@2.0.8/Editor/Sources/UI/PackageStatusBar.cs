using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Experimental.UIElements;

namespace UnityEditor.PackageManager.UI
{
#if !UNITY_2018_3_OR_NEWER
    internal class PackageStatusBarFactory : UxmlFactory<PackageStatusBar>
    {
        protected override PackageStatusBar DoCreate(IUxmlAttributes bag, CreationContext cc)
        {
            return new PackageStatusBar();
        }
    }
#endif
    
    internal class PackageStatusBar : VisualElement
    {
#if UNITY_2018_3_OR_NEWER
        internal new class UxmlFactory : UxmlFactory<PackageStatusBar> { }
#endif

        private readonly VisualElement root;
        private string LastErrorMessage;

        private List<IBaseOperation> operationsInProgress;

        private enum StatusType {Normal, Loading, Error};  

        public PackageStatusBar()
        {
            root = Resources.GetTemplate("PackageStatusBar.uxml");
            Add(root);

            MoreAddOptionsButton.clickable.clicked += OnMoreAddOptionsButtonClick;

            LastErrorMessage = string.Empty;
            operationsInProgress = new List<IBaseOperation>();

            SetDefaultMessage();

            PackageCollection.Instance.ListSignal.WhenOperation(OnListOrSearchOperation);
            PackageCollection.Instance.SearchSignal.WhenOperation(OnListOrSearchOperation);
        }

        private void SetDefaultMessage()
        {
            if(!string.IsNullOrEmpty(PackageCollection.Instance.lastUpdateTime))
                SetStatusMessage(StatusType.Normal, "Last update " + PackageCollection.Instance.lastUpdateTime);
            else
                SetStatusMessage(StatusType.Normal, string.Empty);
        }

        private void OnListOrSearchOperation(IBaseOperation operation)
        {
            if (operation == null || operation.IsCompleted)
                return;
            operationsInProgress.Add(operation);
            operation.OnOperationFinalized += () => { OnOperationFinalized(operation); };
            operation.OnOperationError += OnOperationError;

            SetStatusMessage(StatusType.Loading, "Loading packages...");
        }

        private void OnOperationFinalized(IBaseOperation operation)
        {
            operationsInProgress.Remove(operation);

            if (operationsInProgress.Any()) return;

            var errorMessage = LastErrorMessage;

            if (Application.internetReachability == NetworkReachability.NotReachable)
            {
                EditorApplication.update -= CheckInternetReachability;
                EditorApplication.update += CheckInternetReachability;

                errorMessage = "You seem to be offline.";
            }

            if (!string.IsNullOrEmpty(errorMessage))
                SetStatusMessage(StatusType.Error, errorMessage);
            else
                SetDefaultMessage();
        }

        private void OnOperationError(Error error)
        {
            LastErrorMessage = "Cannot load packages, see console.";
        }

        private void CheckInternetReachability()
        {
            if (Application.internetReachability == NetworkReachability.NotReachable) return;

            PackageCollection.Instance.FetchListCache(true);
            PackageCollection.Instance.FetchSearchCache(true);
            EditorApplication.update -= CheckInternetReachability;
        }

        private void SetStatusMessage(StatusType status, string message)
        {
            if (status == StatusType.Loading)
                LoadingSpinner.Start();
            else
                LoadingSpinner.Stop();

            UIUtils.SetElementDisplay(LoadingIcon, status == StatusType.Error);
            if (status == StatusType.Error)
                LoadingText.AddToClassList("icon");
            else
                LoadingText.RemoveFromClassList("icon");

            LoadingText.text = message;
        }

        private void OnMoreAddOptionsButtonClick()
        {
            var menu = new GenericMenu();

            var addPackageFromDiskItem = new GUIContent("Add package from disk...");

            /* // Disable adding from url field before the feature is ready
            var addPackageFromUrlItem = new GUIContent("Add package from URL...");
            menu.AddItem(addPackageFromUrlItem, false, delegate
            {
                AddFromUrlField.Show(true);
            });
            */

            menu.AddItem(addPackageFromDiskItem, false, delegate
            {
                var path = EditorUtility.OpenFilePanelWithFilters("Select package on disk", "", new[] { "package.json file", "json" });
                if (!string.IsNullOrEmpty(path) && !Package.AddRemoveOperationInProgress)
                    Package.AddFromLocalDisk(path);
            });
            var menuPosition = MoreAddOptionsButton.LocalToWorld(new Vector2(MoreAddOptionsButton.layout.width, 0));
            var menuRect = new Rect(menuPosition, Vector2.zero);
            menu.DropDown(menuRect);
        }

        private PackageAddFromUrlField AddFromUrlField { get { return root.Q<PackageAddFromUrlField>("packageAddFromUrlField");  }}
        private VisualElement LoadingSpinnerContainer { get { return root.Q<VisualElement>("loadingSpinnerContainer");  }}
        private LoadingSpinner LoadingSpinner { get { return root.Q<LoadingSpinner>("packageSpinner");  }}
        private Label LoadingIcon { get { return root.Q<Label>("loadingIcon");  }}
        private Label LoadingText { get { return root.Q<Label>("loadingText");  }}
        private Button MoreAddOptionsButton{ get { return root.Q<Button>("moreAddOptionsButton");  }}
    }
}
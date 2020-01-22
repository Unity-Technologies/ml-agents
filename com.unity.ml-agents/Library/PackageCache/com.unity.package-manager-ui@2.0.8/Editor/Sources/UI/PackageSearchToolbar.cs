using System;
using UnityEngine;
using UnityEngine.Experimental.UIElements;

namespace UnityEditor.PackageManager.UI
{
#if !UNITY_2018_3_OR_NEWER
    internal class PackageSearchToolbarFactory : UxmlFactory<PackageSearchToolbar>
    {
        protected override PackageSearchToolbar DoCreate(IUxmlAttributes bag, CreationContext cc)
        {
            return new PackageSearchToolbar();
        }
    }
#endif

    internal class PackageSearchToolbar : VisualElement
    {
#if UNITY_2018_3_OR_NEWER
        internal new class UxmlFactory : UxmlFactory<PackageSearchToolbar> { }
#endif
        private const string kPlaceHolder = "Search by package name, verified, preview or version number...";
        
        public event Action OnFocusChange = delegate { };
        public event Action<string> OnSearchChange = delegate { };
        
        private string searchText;
        private bool showingPlaceHolder;

        private readonly VisualElement root;

        public PackageSearchToolbar()
        {
            root = Resources.GetTemplate("PackageSearchToolbar.uxml");
            Add(root);
            root.StretchToParentSize();

            SearchTextField.value = searchText;
            SearchTextField.maxLength = 54;
            SearchCancelButton.clickable.clicked += SearchCancelButtonClick;
            
            RegisterCallback<AttachToPanelEvent>(OnEnterPanel);
            RegisterCallback<DetachFromPanelEvent>(OnLeavePanel);

            searchText = PackageSearchFilter.Instance.SearchText;
            
            if (string.IsNullOrEmpty(searchText))
            {
                showingPlaceHolder = true;
                SearchTextField.value = kPlaceHolder;
                SearchTextField.AddToClassList("placeholder");
            }
            else
            {
                showingPlaceHolder = false;
                SearchTextField.value = searchText;
                SearchTextField.RemoveFromClassList("placeholder");
            }
        }
        
        public void GrabFocus()
        {
            SearchTextField.Focus();
        }

        public new void SetEnabled(bool enable)
        {
            base.SetEnabled(enable);
            SearchTextField.SetEnabled(enable);
            SearchCancelButton.SetEnabled(enable);
        }

        private void OnSearchTextFieldChange(ChangeEvent<string> evt)
        {
            if (showingPlaceHolder && evt.newValue == kPlaceHolder)
                return;
            if (!string.IsNullOrEmpty(evt.newValue))
                SearchCancelButton.AddToClassList("on");
            else
                SearchCancelButton.RemoveFromClassList("on");

            searchText = evt.newValue;
            OnSearchChange(searchText);
        }

        private void OnSearchTextFieldFocus(FocusEvent evt)
        {
            if (showingPlaceHolder)
            {
                SearchTextField.value = string.Empty;
                SearchTextField.RemoveFromClassList("placeholder");
                showingPlaceHolder = false;
            }
        }

        private void OnSearchTextFieldFocusOut(FocusOutEvent evt)
        {
            if (string.IsNullOrEmpty(searchText))
            {
                showingPlaceHolder = true;
                SearchTextField.AddToClassList("placeholder");
                SearchTextField.value = kPlaceHolder;
            }
        }

        private void SearchCancelButtonClick()
        {
            if (!string.IsNullOrEmpty(SearchTextField.value))
            {
                SearchTextField.value = string.Empty;
            }
            
            showingPlaceHolder = true;
            SearchTextField.AddToClassList("placeholder");
            SearchTextField.value = kPlaceHolder;
        }

        private void OnEnterPanel(AttachToPanelEvent evt)
        {
            SearchTextField.RegisterCallback<FocusEvent>(OnSearchTextFieldFocus);
            SearchTextField.RegisterCallback<FocusOutEvent>(OnSearchTextFieldFocusOut);
            SearchTextField.RegisterCallback<ChangeEvent<string>>(OnSearchTextFieldChange);
            SearchTextField.RegisterCallback<KeyDownEvent>(OnKeyDownShortcut);
        }

        private void OnLeavePanel(DetachFromPanelEvent evt)
        {
            SearchTextField.UnregisterCallback<FocusEvent>(OnSearchTextFieldFocus);
            SearchTextField.UnregisterCallback<FocusOutEvent>(OnSearchTextFieldFocusOut);
            SearchTextField.UnregisterCallback<ChangeEvent<string>>(OnSearchTextFieldChange);
            SearchTextField.UnregisterCallback<KeyDownEvent>(OnKeyDownShortcut);
        }

        private void OnKeyDownShortcut(KeyDownEvent evt)
        {
            if (evt.keyCode == KeyCode.Escape)
            {
                SearchCancelButtonClick();
                SearchCancelButton.Focus();
                evt.StopImmediatePropagation();
                return;
            }
            
            if (evt.keyCode == KeyCode.Tab)
            {
                OnFocusChange();
                evt.StopImmediatePropagation();
            }
        }
        private TextField SearchTextField { get { return root.Q<TextField>("searchTextField"); } }
        private Button SearchCancelButton { get { return root.Q<Button>("searchCancelButton"); } }
    }
}
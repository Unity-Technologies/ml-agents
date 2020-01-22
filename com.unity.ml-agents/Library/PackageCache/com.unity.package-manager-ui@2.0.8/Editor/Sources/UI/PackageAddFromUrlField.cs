using UnityEngine;
using UnityEngine.Experimental.UIElements;

namespace UnityEditor.PackageManager.UI
{
#if !UNITY_2018_3_OR_NEWER
    internal class PackageAddFromUrlFieldFactory : UxmlFactory<PackageAddFromUrlField>
    {
        protected override PackageAddFromUrlField DoCreate(IUxmlAttributes bag, CreationContext cc)
        {
            return new PackageAddFromUrlField();
        }
    }
#endif
    
    internal class PackageAddFromUrlField : VisualElement
    {
#if UNITY_2018_3_OR_NEWER
        internal new class UxmlFactory : UxmlFactory<PackageAddFromUrlField> { }
#endif
        private string urlText;

        private readonly VisualElement root;

        public PackageAddFromUrlField()
        {
            root = Resources.GetTemplate("PackageAddFromUrlField.uxml");
            Add(root);

            UrlTextField.value = urlText;

            AddButton.SetEnabled(!string.IsNullOrEmpty(urlText));
            AddButton.clickable.clicked += OnAddButtonClick;

            RegisterCallback<AttachToPanelEvent>(OnEnterPanel);
            RegisterCallback<DetachFromPanelEvent>(OnLeavePanel);
        }

        private void OnUrlTextFieldChange(ChangeEvent<string> evt)
        {
            urlText = evt.newValue;
            AddButton.SetEnabled(!string.IsNullOrEmpty(urlText));
        }

        private void OnUrlTextFieldFocus(FocusEvent evt)
        {
            Show();
        }

        private void OnUrlTextFieldFocusOut(FocusOutEvent evt)
        {
            Hide();
        }

        private void OnContainerFocus(FocusEvent evt)
        {
            UrlTextField.Focus();
        }

        private void OnContainerFocusOut(FocusOutEvent evt)
        {
            Hide();
        }

        private void OnEnterPanel(AttachToPanelEvent evt)
        {
            AddFromUrlFieldContainer.RegisterCallback<FocusEvent>(OnContainerFocus);
            AddFromUrlFieldContainer.RegisterCallback<FocusOutEvent>(OnContainerFocusOut);
            UrlTextField.RegisterCallback<FocusEvent>(OnUrlTextFieldFocus);
            UrlTextField.RegisterCallback<FocusOutEvent>(OnUrlTextFieldFocusOut);
            UrlTextField.RegisterCallback<ChangeEvent<string>>(OnUrlTextFieldChange);
            UrlTextField.RegisterCallback<KeyDownEvent>(OnKeyDownShortcut);
            Hide();
        }

        private void OnLeavePanel(DetachFromPanelEvent evt)
        {
            AddFromUrlFieldContainer.UnregisterCallback<FocusEvent>(OnContainerFocus);
            AddFromUrlFieldContainer.UnregisterCallback<FocusOutEvent>(OnContainerFocusOut);
            UrlTextField.UnregisterCallback<FocusEvent>(OnUrlTextFieldFocus);
            UrlTextField.UnregisterCallback<FocusOutEvent>(OnUrlTextFieldFocusOut);
            UrlTextField.UnregisterCallback<ChangeEvent<string>>(OnUrlTextFieldChange);
            UrlTextField.UnregisterCallback<KeyDownEvent>(OnKeyDownShortcut);
        }

        private void OnKeyDownShortcut(KeyDownEvent evt)
        {
            switch (evt.keyCode)
            {
                case KeyCode.Escape:
                    Hide();
                    break;
                case KeyCode.Return:
                case KeyCode.KeypadEnter:
                    OnAddButtonClick();
                    break;
            }
        }

        private void OnAddButtonClick()
        {
            var path = urlText;
            if (!string.IsNullOrEmpty(path) && !Package.AddRemoveOperationInProgress)
            {
                Package.AddFromLocalDisk(path);
                Hide();
            }
        }

        internal void Hide()
        {
            UIUtils.SetElementDisplay(this, false);
        }

        internal void Show(bool reset = false)
        {
            if (reset)
                Reset();
            UIUtils.SetElementDisplay(this, true);
        }

        private void Reset()
        {
            UrlTextField.value = string.Empty;
            urlText = string.Empty;
            AddButton.SetEnabled(false);
            UrlTextField.Focus();
        }

        private VisualElement AddFromUrlFieldContainer { get { return root.Q<VisualElement>("addFromUrlFieldContainer");  }}
        private TextField UrlTextField { get { return root.Q<TextField>("urlTextField"); } }
        private Button AddButton{ get { return root.Q<Button>("addFromUrlButton");  }}
    }
}
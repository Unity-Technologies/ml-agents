using System;
using UnityEngine.Experimental.UIElements;

namespace UnityEditor.PackageManager.UI
{
#if !UNITY_2018_3_OR_NEWER
    internal class AlertFactory : UxmlFactory<Alert>
    {
        protected override Alert DoCreate(IUxmlAttributes bag, CreationContext cc)
        {
            return new Alert();
        }
    }
#endif

    internal class Alert : VisualElement
    {
#if UNITY_2018_3_OR_NEWER
        internal new class UxmlFactory : UxmlFactory<Alert> { }
#endif

        private const string TemplatePath = PackageManagerWindow.ResourcesPath + "Templates/Alert.uxml";
        private readonly VisualElement root;
        private const float originalPositionRight = 5.0f;
        private const float positionRightWithScroll = 12.0f;

        public Action OnCloseError;

        public Alert()
        {
            UIUtils.SetElementDisplay(this, false);

            root = AssetDatabase.LoadAssetAtPath<VisualTreeAsset>(TemplatePath).CloneTree(null);
            Add(root);
            root.StretchToParentSize();

            CloseButton.clickable.clicked += () =>
            {
                if (null != OnCloseError)
                    OnCloseError();
                ClearError();
            };
        }

        public void SetError(Error error)
        {
            var message = "An error occured.";
            if (error != null)
                message = error.message ?? string.Format("An error occurred ({0})", error.errorCode.ToString());

            AlertMessage.text = message;
            UIUtils.SetElementDisplay(this, true);
        }

        public void ClearError()
        {
            UIUtils.SetElementDisplay(this, false);
            AdjustSize(false);
            AlertMessage.text = "";
            OnCloseError = null;
        }

        public void AdjustSize(bool verticalScrollerVisible)
        {
            if (verticalScrollerVisible)
                style.positionRight = originalPositionRight + positionRightWithScroll;
            else
                style.positionRight = originalPositionRight;
        }

        private Label AlertMessage { get { return root.Q<Label>("alertMessage"); } }
        private Button CloseButton { get { return root.Q<Button>("close"); } }
    }
}

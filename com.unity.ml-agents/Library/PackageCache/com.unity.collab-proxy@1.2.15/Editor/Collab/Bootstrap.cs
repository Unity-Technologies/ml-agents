using UnityEditor;
using UnityEditor.Collaboration;
using UnityEngine;

namespace CollabProxy.UI
{
    [InitializeOnLoad]
    public class Bootstrap
    {
        private const float kCollabToolbarButtonWidth = 78.0f;
        
        static Bootstrap()
        {
            Collab.ShowHistoryWindow = CollabHistoryWindow.ShowHistoryWindow;
            Collab.ShowToolbarAtPosition = CollabToolbarWindow.ShowCenteredAtPosition;
            Collab.IsToolbarVisible = CollabToolbarWindow.IsVisible;
            Collab.CloseToolbar = CollabToolbarWindow.CloseToolbar;
            Toolbar.AddSubToolbar(new CollabToolbarButton
            {
                Width = kCollabToolbarButtonWidth
            });
        }
    }
}
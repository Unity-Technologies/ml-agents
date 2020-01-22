using System;

namespace UnityEditor.Collaboration
{
    internal static class CollabAnalytics
    {
        [Serializable]
        private struct CollabUserActionAnalyticsEvent
        {
            public string category;
            public string action;
        }

        public static void SendUserAction(string category, string action)
        {
            EditorAnalytics.SendCollabUserAction(new CollabUserActionAnalyticsEvent() { category = category, action = action });
        }

        public static readonly string historyCategoryString = "History";
    };
}

using System;
using UnityEngine;

namespace Unity.MLAgents.Analytics
{
    internal static class AnalyticsUtils
    {
        public static string Hash(string s)
        {
            var behaviorNameHash = Hash128.Compute(s);
            return behaviorNameHash.ToString();
        }

        internal static bool s_SendEditorAnalytics = true;

        internal class DisableAnalyticsSending : IDisposable
        {
            private bool m_PreviousSendEditorAnalytics;

            public DisableAnalyticsSending()
            {
                m_PreviousSendEditorAnalytics = s_SendEditorAnalytics;
                s_SendEditorAnalytics = false;
            }

            public void Dispose()
            {
                s_SendEditorAnalytics = m_PreviousSendEditorAnalytics;
            }
        }
    }
}

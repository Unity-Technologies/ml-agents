using System;
using UnityEngine;

namespace Unity.MLAgents.Analytics
{
    internal static class AnalyticsUtils
    {
        /// <summary>
        /// Hash a string to remove PII or secret info before sending to analytics
        /// </summary>
        /// <param name="s"></param>
        /// <returns>A string containing the Hash128 of the input string.</returns>
        public static string Hash(string s)
        {
            var behaviorNameHash = Hash128.Compute(s);
            return behaviorNameHash.ToString();
        }

        internal static bool s_SendEditorAnalytics = true;

        /// <summary>
        /// Helper class to temporarily disable sending analytics from unit tests.
        /// </summary>
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

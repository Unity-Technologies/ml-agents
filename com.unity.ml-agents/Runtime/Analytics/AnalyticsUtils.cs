using System;
using System.Text;
using System.Security.Cryptography;
using UnityEngine;

namespace Unity.MLAgents.Analytics
{
    internal static class AnalyticsUtils
    {
        /// <summary>
        /// Conversion function from byte array to hex string
        /// </summary>
        /// <param name="array"></param>
        /// <returns>A byte array to be hex encoded.</returns>
        private static string ToHexString(byte[] array)
        {
            StringBuilder hex = new StringBuilder(array.Length * 2);
            foreach (byte b in array)
            {
                hex.AppendFormat("{0:x2}", b);
            }
            return hex.ToString();
        }

        /// <summary>
        /// Hash a string to remove PII or secret info before sending to analytics
        /// </summary>
        /// <param name="key"></param>
        /// <returns>A string containing the key to be used for HMAC encoding.</returns>
        /// <param name="value"></param>
        /// <returns>A string containing the value to be encoded.</returns>
        public static string Hash(string key, string value)
        {
            string hash;
            UTF8Encoding encoder = new UTF8Encoding();
            using (HMACSHA256 hmac = new HMACSHA256(encoder.GetBytes(key)))
            {
                Byte[] hmBytes = hmac.ComputeHash(encoder.GetBytes(value));
                hash = ToHexString(hmBytes);
            }
            return hash;
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

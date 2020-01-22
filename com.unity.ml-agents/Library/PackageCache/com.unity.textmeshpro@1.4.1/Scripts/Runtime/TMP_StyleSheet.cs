using UnityEngine;
using System;
using System.Collections.Generic;


namespace TMPro
{

    [Serializable]
    public class TMP_StyleSheet : ScriptableObject
    {
        private static TMP_StyleSheet s_Instance;

        [SerializeField]
        private List<TMP_Style> m_StyleList = new List<TMP_Style>(1);
        private Dictionary<int, TMP_Style> m_StyleDictionary = new Dictionary<int, TMP_Style>();


        /// <summary>
        /// Get a singleton instance of the TMP_StyleSheet
        /// </summary>
        public static TMP_StyleSheet instance
        {
            get
            {
                if (s_Instance == null)
                {
                    s_Instance = TMP_Settings.defaultStyleSheet;

                    if (s_Instance == null)
                        s_Instance = Resources.Load<TMP_StyleSheet>("Style Sheets/Default Style Sheet");

                    if (s_Instance == null) return null;

                    // Load the style dictionary.
                    s_Instance.LoadStyleDictionaryInternal();
                }

                return s_Instance;
            }
        }


        /// <summary>
        /// Static Function to load the Default Style Sheet.
        /// </summary>
        /// <returns></returns>
        public static TMP_StyleSheet LoadDefaultStyleSheet()
        {
            return instance;
        }


        /// <summary>
        /// Function to retrieve the Style matching the HashCode.
        /// </summary>
        /// <param name="hashCode"></param>
        /// <returns></returns>
        public static TMP_Style GetStyle(int hashCode)
        {
            return instance.GetStyleInternal(hashCode);
        }


        /// <summary>
        /// Internal method to retrieve the Style matching the Hashcode.
        /// </summary>
        /// <param name="hashCode"></param>
        /// <returns></returns>
        private TMP_Style GetStyleInternal(int hashCode)
        {
            TMP_Style style;

            if (m_StyleDictionary.TryGetValue(hashCode, out style))
            {
                return style;
            }

            return null;
        }


        public void UpdateStyleDictionaryKey(int old_key, int new_key)
        {
            if (m_StyleDictionary.ContainsKey(old_key))
            {
                TMP_Style style = m_StyleDictionary[old_key];
                m_StyleDictionary.Add(new_key, style);
                m_StyleDictionary.Remove(old_key);
            }
        }


        /// <summary>
        /// Function to update the internal reference to a newly assigned style sheet in the TMP Settings.
        /// </summary>
        public static void UpdateStyleSheet()
        {
            // Reset instance
            s_Instance = null;

            RefreshStyles();
        }


        /// <summary>
        /// Function to refresh the Style Dictionary.
        /// </summary>
        public static void RefreshStyles()
        {
            instance.LoadStyleDictionaryInternal();
        }


        /// <summary>
        /// 
        /// </summary>
        private void LoadStyleDictionaryInternal()
        {
            m_StyleDictionary.Clear();
            
            // Read Styles from style list and store them into dictionary for faster access.
            for (int i = 0; i < m_StyleList.Count; i++)
            {
                m_StyleList[i].RefreshStyle();
              
                if (!m_StyleDictionary.ContainsKey(m_StyleList[i].hashCode))
                    m_StyleDictionary.Add(m_StyleList[i].hashCode, m_StyleList[i]);
            }
        }
    }

}
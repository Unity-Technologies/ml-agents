using UnityEngine;
using System.Collections;

#pragma warning disable 0649 // Disabled warnings.

namespace TMPro
{
    
    [System.Serializable]
    public class TMP_Style
    {
        // PUBLIC PROPERTIES

        /// <summary>
        /// The name identifying this style. ex. <style="name">.
        /// </summary>
        public string name
        { get { return m_Name; } set { if (value != m_Name) m_Name = value; } }
       
        /// <summary>
        /// The hash code corresponding to the name of this style.
        /// </summary>
        public int hashCode
        { get { return m_HashCode; } set { if (value != m_HashCode) m_HashCode = value; } }

        /// <summary>
        /// The initial definition of the style. ex. <b> <u>.
        /// </summary>
        public string styleOpeningDefinition
        { get { return m_OpeningDefinition; } }

        /// <summary>
        /// The closing definition of the style. ex. </b> </u>.
        /// </summary>
        public string styleClosingDefinition
        { get { return m_ClosingDefinition; } }


        public int[] styleOpeningTagArray
        { get { return m_OpeningTagArray; } }


        public int[] styleClosingTagArray
        { get { return m_ClosingTagArray; } }

       
        // PRIVATE FIELDS
        [SerializeField]
        private string m_Name;

        [SerializeField]
        private int m_HashCode;

        [SerializeField]
        private string m_OpeningDefinition;

        [SerializeField]
        private string m_ClosingDefinition;

        [SerializeField]
        private int[] m_OpeningTagArray;

        [SerializeField]
        private int[] m_ClosingTagArray;


        //public TMP_Style()
        //{
            //Debug.Log("New Style with Name: " + m_Name + " was created. ID: ");
        //}


        /// <summary>
        /// Function to update the content of the int[] resulting from changes to OpeningDefinition & ClosingDefinition.
        /// </summary>
        public void RefreshStyle()
        {
            m_HashCode = TMP_TextUtilities.GetSimpleHashCode(m_Name);
            
            m_OpeningTagArray = new int[m_OpeningDefinition.Length];
            for (int i = 0; i < m_OpeningDefinition.Length; i++)       
                m_OpeningTagArray[i] = m_OpeningDefinition[i];

            m_ClosingTagArray = new int[m_ClosingDefinition.Length];
            for (int i = 0; i < m_ClosingDefinition.Length; i++)
                m_ClosingTagArray[i] = m_ClosingDefinition[i];

#if UNITY_EDITOR
            // Event to update objects when styles are changed in the editor.
            TMPro_EventManager.ON_TEXT_STYLE_PROPERTY_CHANGED(true);
#endif
        }

    }
}

using System.Collections.Generic;

namespace UnityEditor.PostProcessing
{
    public class DefaultPostFxModelEditor : PostProcessingModelEditor
    {
        List<SerializedProperty> m_Properties = new List<SerializedProperty>();

        public override void OnEnable()
        {
            var iter = m_SettingsProperty.Copy().GetEnumerator();
            while (iter.MoveNext())
                m_Properties.Add(((SerializedProperty)iter.Current).Copy());
        }

        public override void OnInspectorGUI()
        {
            foreach (var property in m_Properties)
                EditorGUILayout.PropertyField(property);
        }
    }
}

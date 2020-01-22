// When enabled, allows setting the material by dropping a material onto the MeshRenderer inspector component. 
// The drawback is that the MeshRenderer inspector will not have properties for light probes, so if you need light probe support, do not enable this.
//#define ALLOW_MESHRENDERER_MATERIAL_DRAG_N_DROP

using UnityEngine;
using UnityEditor;
using System.Collections;


namespace TMPro.EditorUtilities
{
    // Disabled for compatibility reason as lightprobe setup isn't supported due to inability to inherit from MeshRendererEditor class
#if ALLOW_MESHRENDERER_MATERIAL_DRAG_N_DROP
    [CanEditMultipleObjects]
    [CustomEditor(typeof(MeshRenderer))]
    public class TMP_MeshRendererEditor : Editor
    {
        private SerializedProperty m_Materials;

        void OnEnable()
        {
            m_Materials = serializedObject.FindProperty("m_Materials");
        }


        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            // Get a reference to the current material.
            SerializedProperty material_prop = m_Materials.GetArrayElementAtIndex(0);
            Material currentMaterial = material_prop.objectReferenceValue as Material;

            EditorGUI.BeginChangeCheck();
            base.OnInspectorGUI();
            if (EditorGUI.EndChangeCheck())
            {
                material_prop = m_Materials.GetArrayElementAtIndex(0);

                TMP_FontAsset newFontAsset = null;
                Material newMaterial = null;

                if (material_prop != null)
                    newMaterial = material_prop.objectReferenceValue as Material;

                // Check if the new material is referencing a different font atlas texture.
                if (newMaterial != null && currentMaterial.GetInstanceID() != newMaterial.GetInstanceID())
                {
                    // Search for the Font Asset matching the new font atlas texture.
                    newFontAsset = TMP_EditorUtility.FindMatchingFontAsset(newMaterial);
                }


                GameObject[] objects = Selection.gameObjects;

                for (int i = 0; i < objects.Length; i++)
                {
                    // Assign new font asset
                    if (newFontAsset != null)
                    {
                        TMP_Text textComponent = objects[i].GetComponent<TMP_Text>();

                        if (textComponent != null)
                        {
                            Undo.RecordObject(textComponent, "Font Asset Change");
                            textComponent.font = newFontAsset;
                        }
                    }

                    TMPro_EventManager.ON_DRAG_AND_DROP_MATERIAL_CHANGED(objects[i], currentMaterial, newMaterial);
                }
            }
        }
    }
#endif
}

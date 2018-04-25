using UnityEngine;
using UnityEngine.PostProcessing;

namespace UnityEditor.PostProcessing
{
    using Settings = UserLutModel.Settings;

    [PostProcessingModelEditor(typeof(UserLutModel))]
    public class UserLutModelEditor : PostProcessingModelEditor
    {
        SerializedProperty m_Texture;
        SerializedProperty m_Contribution;

        public override void OnEnable()
        {
            m_Texture = FindSetting((Settings x) => x.lut);
            m_Contribution = FindSetting((Settings x) => x.contribution);
        }

        public override void OnInspectorGUI()
        {
            var lut = (target as UserLutModel).settings.lut;

            // Checks import settings on the lut, offers to fix them if invalid
            if (lut != null)
            {
                var importer = (TextureImporter)AssetImporter.GetAtPath(AssetDatabase.GetAssetPath(lut));

                if (importer != null) // Fails when using an internal texture
                {
#if UNITY_5_5_OR_NEWER
                    bool valid = importer.anisoLevel == 0
                        && importer.mipmapEnabled == false
                        && importer.sRGBTexture == false
                        && (importer.textureCompression == TextureImporterCompression.Uncompressed);
#else
                    bool valid = importer.anisoLevel == 0
                        && importer.mipmapEnabled == false
                        && importer.linearTexture == true
                        && (importer.textureFormat == TextureImporterFormat.RGB24 || importer.textureFormat == TextureImporterFormat.AutomaticTruecolor);
#endif

                    if (!valid)
                    {
                        EditorGUILayout.HelpBox("Invalid LUT import settings.", MessageType.Warning);

                        GUILayout.Space(-32);
                        using (new EditorGUILayout.HorizontalScope())
                        {
                            GUILayout.FlexibleSpace();
                            if (GUILayout.Button("Fix", GUILayout.Width(60)))
                            {
                                SetLUTImportSettings(importer);
                                AssetDatabase.Refresh();
                            }
                            GUILayout.Space(8);
                        }
                        GUILayout.Space(11);
                    }
                }
                else
                {
                    m_Texture.objectReferenceValue = null;
                }
            }

            EditorGUILayout.PropertyField(m_Texture);
            EditorGUILayout.PropertyField(m_Contribution);
        }

        void SetLUTImportSettings(TextureImporter importer)
        {
#if UNITY_5_5_OR_NEWER
            importer.textureType = TextureImporterType.Default;
            importer.sRGBTexture = false;
            importer.textureCompression = TextureImporterCompression.Uncompressed;
#else
            importer.textureType = TextureImporterType.Advanced;
            importer.linearTexture = true;
            importer.textureFormat = TextureImporterFormat.RGB24;
#endif
            importer.anisoLevel = 0;
            importer.mipmapEnabled = false;
            importer.SaveAndReimport();
        }
    }
}

using UnityEngine;
using UnityEditor;

namespace TMPro.EditorUtilities
{
    public class TMP_BitmapShaderGUI : TMP_BaseShaderGUI
    {
        static bool s_Face = true;

        protected override void DoGUI()
        {
            s_Face = BeginPanel("Face", s_Face);
            if (s_Face)
            {
                DoFacePanel();
            }

            EndPanel();

            s_DebugExtended = BeginPanel("Debug Settings", s_DebugExtended);
            if (s_DebugExtended)
            {
                DoDebugPanel();
            }

            EndPanel();
        }

        void DoFacePanel()
        {
            EditorGUI.indentLevel += 1;
            if (m_Material.HasProperty(ShaderUtilities.ID_FaceTex))
            {
                DoColor("_FaceColor", "Color");
                DoTexture2D("_FaceTex", "Texture", true);
            }
            else
            {
                DoColor("_Color", "Color");
                DoSlider("_DiffusePower", "Diffuse Power");
            }

            EditorGUI.indentLevel -= 1;

            EditorGUILayout.Space();
        }

        void DoDebugPanel()
        {
            EditorGUI.indentLevel += 1;
            DoTexture2D("_MainTex", "Font Atlas");
            if (m_Material.HasProperty(ShaderUtilities.ID_VertexOffsetX))
            {
                if (m_Material.HasProperty(ShaderUtilities.ID_Padding))
                {
                    EditorGUILayout.Space();
                    DoFloat("_Padding", "Padding");
                }
                
                EditorGUILayout.Space();
                DoFloat("_VertexOffsetX", "Offset X");
                DoFloat("_VertexOffsetY", "Offset Y");
            }

            if (m_Material.HasProperty(ShaderUtilities.ID_MaskSoftnessX))
            {
                EditorGUILayout.Space();
                DoFloat("_MaskSoftnessX", "Softness X");
                DoFloat("_MaskSoftnessY", "Softness Y");
                DoVector("_ClipRect", "Clip Rect", s_LbrtVectorLabels);
            }

            if (m_Material.HasProperty(ShaderUtilities.ID_StencilID))
            {
                EditorGUILayout.Space();
                DoFloat("_Stencil", "Stencil ID");
                DoFloat("_StencilComp", "Stencil Comp");
            }

            EditorGUI.indentLevel -= 1;

            EditorGUILayout.Space();
        }
    }
}

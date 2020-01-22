using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine.UI;

namespace TMPro
{

    [DisallowMultipleComponent]
    [RequireComponent(typeof(MeshRenderer))]
    [RequireComponent(typeof(MeshFilter))] 
    [AddComponentMenu("Mesh/TextMeshPro - Text")]
    [ExecuteAlways]
    public partial class TextMeshPro : TMP_Text, ILayoutElement
    {
        // Public Properties and Serializable Properties

        /// <summary>
        /// Sets the Renderer's sorting Layer ID
        /// </summary>
        public int sortingLayerID
        {
            get { return m_renderer.sortingLayerID; }
            set { m_renderer.sortingLayerID = value; }
        }

        /// <summary>
        /// Sets the Renderer's sorting order within the assigned layer.
        /// </summary>
        public int sortingOrder
        {
            get { return m_renderer.sortingOrder; }
            set { m_renderer.sortingOrder = value; }
        }

        /// <summary>
        /// Determines if the size of the text container will be adjusted to fit the text object when it is first created.
        /// </summary>
        public override bool autoSizeTextContainer
        {
            get { return m_autoSizeTextContainer; }

            set { if (m_autoSizeTextContainer == value) return; m_autoSizeTextContainer = value; if (m_autoSizeTextContainer) { TMP_UpdateManager.RegisterTextElementForLayoutRebuild(this); SetLayoutDirty(); } }
        }


        /// <summary>
        /// Returns a reference to the Text Container
        /// </summary>
        [Obsolete("The TextContainer is now obsolete. Use the RectTransform instead.")]
        public TextContainer textContainer
        {
            get
            {
                return null;
            }
        }


        /// <summary>
        /// Returns a reference to the Transform
        /// </summary>
        public new Transform transform
        {
            get
            {
                if (m_transform == null)
                    m_transform = GetComponent<Transform>();
                
                return m_transform;
            }
        }


        #pragma warning disable 0108
        /// <summary>
        /// Returns the rendered assigned to the text object.
        /// </summary>
        public Renderer renderer
        {
            get
            {
                if (m_renderer == null)
                    m_renderer = GetComponent<Renderer>();

                return m_renderer;
            }
        }


        /// <summary>
        /// Returns the mesh assigned to the text object.
        /// </summary>
        public override Mesh mesh
        {
            get
            {
                if (m_mesh == null)
                {
                    m_mesh = new Mesh();
                    m_mesh.hideFlags = HideFlags.HideAndDontSave;
                    this.meshFilter.mesh = m_mesh;
                }

                return m_mesh;
            }
        }

        /// <summary>
        /// Returns the Mesh Filter of the text object.
        /// </summary>
        public MeshFilter meshFilter
        {
            get
            {
                if (m_meshFilter == null)
                    m_meshFilter = GetComponent<MeshFilter>();

                return m_meshFilter;
            }
        }

        // MASKING RELATED PROPERTIES
        /// <summary>
        /// Sets the mask type 
        /// </summary>
        public MaskingTypes maskType
        {
            get { return m_maskType; }
            set { m_maskType = value; SetMask(m_maskType); }
        }


        /// <summary>
        /// Function used to set the mask type and coordinates in World Space
        /// </summary>
        /// <param name="type"></param>
        /// <param name="maskCoords"></param>
        public void SetMask(MaskingTypes type, Vector4 maskCoords)
        {
            SetMask(type);

            SetMaskCoordinates(maskCoords);
        }

        /// <summary>
        /// Function used to set the mask type, coordinates and softness
        /// </summary>
        /// <param name="type"></param>
        /// <param name="maskCoords"></param>
        /// <param name="softnessX"></param>
        /// <param name="softnessY"></param>
        public void SetMask(MaskingTypes type, Vector4 maskCoords, float softnessX, float softnessY)
        {
            SetMask(type);

            SetMaskCoordinates(maskCoords, softnessX, softnessY);
        }


        /// <summary>
        /// Schedule rebuilding of the text geometry.
        /// </summary>
        public override void SetVerticesDirty()
        {
            //Debug.Log("SetVerticesDirty()");

            if (m_verticesAlreadyDirty || this == null || !this.IsActive())
                return;

            TMP_UpdateManager.RegisterTextElementForGraphicRebuild(this);
            m_verticesAlreadyDirty = true;
        }


        /// <summary>
        /// 
        /// </summary>
        public override void SetLayoutDirty()
        {
            m_isPreferredWidthDirty = true;
            m_isPreferredHeightDirty = true;

            if (m_layoutAlreadyDirty || this == null || !this.IsActive())
                return;

            //TMP_UpdateManager.RegisterTextElementForLayoutRebuild(this);
            m_layoutAlreadyDirty = true;
            //LayoutRebuilder.MarkLayoutForRebuild(this.rectTransform);
            m_isLayoutDirty = true;
        }


        /// <summary>
        /// Schedule updating of the material used by the text object.
        /// </summary>
        public override void SetMaterialDirty()
        {
            //Debug.Log("SetMaterialDirty()");

            //if (!this.IsActive())
            //    return;

            //m_isMaterialDirty = true;
            UpdateMaterial();
            //TMP_UpdateManager.RegisterTextElementForGraphicRebuild(this);
        }


        /// <summary>
        /// 
        /// </summary>
        public override void SetAllDirty()
        {
            m_isInputParsingRequired = true;

            SetLayoutDirty();
            SetVerticesDirty();
            SetMaterialDirty();
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="update"></param>
        public override void Rebuild(CanvasUpdate update)
        {
            if (this == null) return;

            if (update == CanvasUpdate.Prelayout)
            {
                if (m_autoSizeTextContainer)
                {
                    m_rectTransform.sizeDelta = GetPreferredValues(Mathf.Infinity, Mathf.Infinity);
                }
            }
            else if (update == CanvasUpdate.PreRender)
            {
                this.OnPreRenderObject();
                m_verticesAlreadyDirty = false;
                m_layoutAlreadyDirty = false;

                if (!m_isMaterialDirty) return;

                UpdateMaterial();
                m_isMaterialDirty = false;
            }
        }


        /// <summary>
        /// 
        /// </summary>
        protected override void UpdateMaterial()
        {
            //Debug.Log("*** UpdateMaterial() ***");

            //if (!this.IsActive())
            //    return;

            if (m_sharedMaterial == null)
                return;

            if (m_renderer == null) m_renderer = this.renderer;

            // Only update the material if it has changed.
            if (m_renderer.sharedMaterial.GetInstanceID() != m_sharedMaterial.GetInstanceID())
                m_renderer.sharedMaterial = m_sharedMaterial;
        }


        /// <summary>
        /// Function to be used to force recomputing of character padding when Shader / Material properties have been changed via script.
        /// </summary>
        public override void UpdateMeshPadding()
        {
            m_padding = ShaderUtilities.GetPadding(m_sharedMaterial, m_enableExtraPadding, m_isUsingBold);
            m_isMaskingEnabled = ShaderUtilities.IsMaskingEnabled(m_sharedMaterial);
            m_havePropertiesChanged = true;
            checkPaddingRequired = false;

            // Return if text object is not awake yet.
            if (m_textInfo == null) return;

            // Update sub text objects
            for (int i = 1; i < m_textInfo.materialCount; i++)
                m_subTextObjects[i].UpdateMeshPadding(m_enableExtraPadding, m_isUsingBold);
        }


        /// <summary>
        /// Function to force regeneration of the mesh before its normal process time. This is useful when changes to the text object properties need to be applied immediately.
        /// </summary>
        public override void ForceMeshUpdate()
        {
            //Debug.Log("ForceMeshUpdate() called.");
            m_havePropertiesChanged = true;
            OnPreRenderObject();
        }


        /// <summary>
        /// Function to force regeneration of the mesh before its normal process time. This is useful when changes to the text object properties need to be applied immediately.
        /// </summary>
        /// <param name="ignoreInactive">If set to true, the text object will be regenerated regardless of is active state.</param>
        public override void ForceMeshUpdate(bool ignoreInactive)
        {
            m_havePropertiesChanged = true;
            m_ignoreActiveState = true;
            OnPreRenderObject();
        }


        /// <summary>
        /// Function used to evaluate the length of a text string.
        /// </summary>
        /// <param name="text"></param>
        /// <returns></returns>
        public override TMP_TextInfo GetTextInfo(string text)
        {
            StringToCharArray(text, ref m_TextParsingBuffer);
            SetArraySizes(m_TextParsingBuffer);

            m_renderMode = TextRenderFlags.DontRender;

            ComputeMarginSize();

            GenerateTextMesh();

            m_renderMode = TextRenderFlags.Render;

            return this.textInfo;
        }


        /// <summary>
        /// Function to clear the geometry of the Primary and Sub Text objects.
        /// </summary>
        public override void ClearMesh(bool updateMesh)
        {
            if (m_textInfo.meshInfo[0].mesh == null) m_textInfo.meshInfo[0].mesh = m_mesh;

            m_textInfo.ClearMeshInfo(updateMesh);
        }


        /// <summary>
        /// Function to force the regeneration of the text object.
        /// </summary>
        /// <param name="flags"> Flags to control which portions of the geometry gets uploaded.</param>
        //public override void ForceMeshUpdate(TMP_VertexDataUpdateFlags flags) { }


        /// <summary>
        /// Function to update the geometry of the main and sub text objects.
        /// </summary>
        /// <param name="mesh"></param>
        /// <param name="index"></param>
        public override void UpdateGeometry(Mesh mesh, int index)
        {
            mesh.RecalculateBounds();
        }


        /// <summary>
        /// Function to upload the updated vertex data and renderer.
        /// </summary>
        public override void UpdateVertexData(TMP_VertexDataUpdateFlags flags)
        {
            int materialCount = m_textInfo.materialCount;

            for (int i = 0; i < materialCount; i++)
            {
                Mesh mesh;

                if (i == 0)
                    mesh = m_mesh;
                else
                {
                    // Clear unused vertices
                    // TODO: Causes issues when sorting geometry as last vertex data attribute get wiped out.
                    //m_textInfo.meshInfo[i].ClearUnusedVertices();

                    mesh = m_subTextObjects[i].mesh;
                }

                //mesh.MarkDynamic();

                if ((flags & TMP_VertexDataUpdateFlags.Vertices) == TMP_VertexDataUpdateFlags.Vertices)
                    mesh.vertices = m_textInfo.meshInfo[i].vertices;

                if ((flags & TMP_VertexDataUpdateFlags.Uv0) == TMP_VertexDataUpdateFlags.Uv0)
                    mesh.uv = m_textInfo.meshInfo[i].uvs0;

                if ((flags & TMP_VertexDataUpdateFlags.Uv2) == TMP_VertexDataUpdateFlags.Uv2)
                    mesh.uv2 = m_textInfo.meshInfo[i].uvs2;

                //if ((flags & TMP_VertexDataUpdateFlags.Uv4) == TMP_VertexDataUpdateFlags.Uv4)
                //    mesh.uv4 = m_textInfo.meshInfo[i].uvs4;

                if ((flags & TMP_VertexDataUpdateFlags.Colors32) == TMP_VertexDataUpdateFlags.Colors32)
                    mesh.colors32 = m_textInfo.meshInfo[i].colors32;

                mesh.RecalculateBounds();
            }
        }


        /// <summary>
        /// Function to upload the updated vertex data and renderer.
        /// </summary>
        public override void UpdateVertexData()
        {
            int materialCount = m_textInfo.materialCount;

            for (int i = 0; i < materialCount; i++)
            {
                Mesh mesh;

                if (i == 0)
                    mesh = m_mesh;
                else
                {
                    // Clear unused vertices
                    m_textInfo.meshInfo[i].ClearUnusedVertices();

                    mesh = m_subTextObjects[i].mesh;
                }


                //mesh.MarkDynamic();
                mesh.vertices = m_textInfo.meshInfo[i].vertices;
                mesh.uv = m_textInfo.meshInfo[i].uvs0;
                mesh.uv2 = m_textInfo.meshInfo[i].uvs2;
                //mesh.uv4 = m_textInfo.meshInfo[i].uvs4;
                mesh.colors32 = m_textInfo.meshInfo[i].colors32;

                mesh.RecalculateBounds();
            }
        }

        public void UpdateFontAsset()
        {
            LoadFontAsset();
        }


        private bool m_currentAutoSizeMode;


        public void CalculateLayoutInputHorizontal()
        {
            //Debug.Log("*** CalculateLayoutInputHorizontal() ***");

            if (!this.gameObject.activeInHierarchy)
                return;

            //IsRectTransformDriven = true;

            m_currentAutoSizeMode = m_enableAutoSizing;

            if (m_isCalculateSizeRequired || m_rectTransform.hasChanged)
            {
                //Debug.Log("Calculating Layout Horizontal");

                //m_LayoutPhase = AutoLayoutPhase.Horizontal;
                //m_isRebuildingLayout = true;

                m_minWidth = 0;
                m_flexibleWidth = 0;

                //m_renderMode = TextRenderFlags.GetPreferredSizes; // Set Text to not Render and exit early once we have new width values.

                if (m_enableAutoSizing)
                {
                    m_fontSize = m_fontSizeMax;
                }

                // Set Margins to Infinity
                m_marginWidth = k_LargePositiveFloat;
                m_marginHeight = k_LargePositiveFloat;

                if (m_isInputParsingRequired || m_isTextTruncated)
                    ParseInputText();

                GenerateTextMesh();

                m_renderMode = TextRenderFlags.Render;

                //m_preferredWidth = (int)m_preferredWidth + 1f;

                ComputeMarginSize();

                //Debug.Log("Preferred Width: " + m_preferredWidth + "  Margin Width: " + m_marginWidth + "  Preferred Height: " + m_preferredHeight + "  Margin Height: " + m_marginHeight + "  Rendered Width: " + m_renderedWidth + "  Height: " + m_renderedHeight + "  RectTransform Width: " + m_rectTransform.rect);

                m_isLayoutDirty = true;
            }
        }


        public void CalculateLayoutInputVertical()
        {
            //Debug.Log("*** CalculateLayoutInputVertical() ***");

            // Check if object is active
            if (!this.gameObject.activeInHierarchy) // || IsRectTransformDriven == false)
                return;

            //IsRectTransformDriven = true;

            if (m_isCalculateSizeRequired || m_rectTransform.hasChanged)
            {
                //Debug.Log("Calculating Layout InputVertical");

                //m_LayoutPhase = AutoLayoutPhase.Vertical;
                //m_isRebuildingLayout = true;

                m_minHeight = 0;
                m_flexibleHeight = 0;

                //m_renderMode = TextRenderFlags.GetPreferredSizes;

                if (m_enableAutoSizing)
                {
                    m_currentAutoSizeMode = true;
                    m_enableAutoSizing = false;
                }

                m_marginHeight = k_LargePositiveFloat;

                GenerateTextMesh();

                m_enableAutoSizing = m_currentAutoSizeMode;

                m_renderMode = TextRenderFlags.Render;

                //m_preferredHeight = (int)m_preferredHeight + 1f;

                ComputeMarginSize();

                //Debug.Log("Preferred Height: " + m_preferredHeight + "  Margin Height: " + m_marginHeight + "  Preferred Width: " + m_preferredWidth + "  Margin Width: " + m_marginWidth + "  Rendered Width: " + m_renderedWidth + "  Height: " + m_renderedHeight + "  RectTransform Width: " + m_rectTransform.rect);

                m_isLayoutDirty = true;
            }

            m_isCalculateSizeRequired = false;
        }
    }
}
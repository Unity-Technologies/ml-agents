using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;

#pragma warning disable 0414 // Disabled a few warnings related to serialized variables not used in this script but used in the editor.

namespace TMPro
{
    [ExecuteAlways]
    public class TMP_SubMeshUI : MaskableGraphic, IClippable, IMaskable, IMaterialModifier
    {
        /// <summary>
        /// The TMP Font Asset assigned to this sub text object.
        /// </summary>
        public TMP_FontAsset fontAsset
        {
            get { return m_fontAsset; }
            set { m_fontAsset = value; }
        }
        [SerializeField]
        private TMP_FontAsset m_fontAsset;


        /// <summary>
        /// The TMP Sprite Asset assigned to this sub text object.
        /// </summary>
        public TMP_SpriteAsset spriteAsset
        {
            get { return m_spriteAsset; }
            set { m_spriteAsset = value; }
        }
        [SerializeField]
        private TMP_SpriteAsset m_spriteAsset;


        /// <summary>
        /// 
        /// </summary>
        public override Texture mainTexture
        {
            get
            {
                if (this.sharedMaterial != null)
                    return this.sharedMaterial.GetTexture(ShaderUtilities.ID_MainTex);


                return null;
            }
        }


        /// <summary>
        /// The material to be assigned to this object. Returns an instance of the material.
        /// </summary>
        public override Material material
        {
            // Return a new Instance of the Material if none exists. Otherwise return the current Material Instance.
            get { return GetMaterial(m_sharedMaterial); }

            // Assign new font material
            set
            {
                if (m_sharedMaterial != null && m_sharedMaterial.GetInstanceID() == value.GetInstanceID())
                    return;

                m_sharedMaterial = m_material = value;

                m_padding = GetPaddingForMaterial();

                SetVerticesDirty();
                SetMaterialDirty();
            }
        }
        [SerializeField]
        private Material m_material;


        /// <summary>
        /// The material to be assigned to this text object.
        /// </summary>
        public Material sharedMaterial
        {
            get { return m_sharedMaterial; }
            set { SetSharedMaterial(value); }
        }
        [SerializeField]
        private Material m_sharedMaterial;


        /// <summary>
        /// 
        /// </summary>
        public Material fallbackMaterial
        {
            get { return m_fallbackMaterial; }
            set
            {
                if (m_fallbackMaterial == value) return;

                if (m_fallbackMaterial != null && m_fallbackMaterial != value)
                    TMP_MaterialManager.ReleaseFallbackMaterial(m_fallbackMaterial);

                m_fallbackMaterial = value;
                TMP_MaterialManager.AddFallbackMaterialReference(m_fallbackMaterial);

                SetSharedMaterial(m_fallbackMaterial);
            }
        }
        private Material m_fallbackMaterial;


        /// <summary>
        /// The source material used by the fallback font
        /// </summary>
        public Material fallbackSourceMaterial
        {
            get { return m_fallbackSourceMaterial; }
            set { m_fallbackSourceMaterial = value; }
        }
        private Material m_fallbackSourceMaterial;


        /// <summary>
        /// Get the material that will be used for rendering.
        /// </summary>
        public override Material materialForRendering
        {
            get
            {
                return TMP_MaterialManager.GetMaterialForRendering(this, m_sharedMaterial);
            }
        }


        /// <summary>
        /// Is the text object using the default font asset material.
        /// </summary>
        public bool isDefaultMaterial
        {
            get { return m_isDefaultMaterial; }
            set { m_isDefaultMaterial = value; }
        }
        [SerializeField]
        private bool m_isDefaultMaterial;


        /// <summary>
        /// Padding value resulting for the property settings on the material.
        /// </summary>
        public float padding
        {
            get { return m_padding; }
            set { m_padding = value; }
        }
        [SerializeField]
        private float m_padding;


        /// <summary>
        /// The Mesh Renderer of this text sub object.
        /// </summary>
        public new CanvasRenderer canvasRenderer
        {
            get { if (m_canvasRenderer == null) m_canvasRenderer = GetComponent<CanvasRenderer>();

                return m_canvasRenderer;
            }
        }
        [SerializeField]
        private CanvasRenderer m_canvasRenderer;


        /// <summary>
        /// The Mesh of this text sub object.
        /// </summary>
        public Mesh mesh
        {
            get
            {
                if (m_mesh == null)
                {
                    m_mesh = new Mesh();
                    m_mesh.hideFlags = HideFlags.HideAndDontSave;
                }

                return m_mesh;
            }
            set { m_mesh = value; }
        }
        private Mesh m_mesh;


        [SerializeField]
        private TextMeshProUGUI m_TextComponent;


        [System.NonSerialized]
        private bool m_isRegisteredForEvents;
        private bool m_materialDirty;
        [SerializeField]
        private int m_materialReferenceIndex;



        /// <summary>
        /// Function to add a new sub text object.
        /// </summary>
        /// <param name="textComponent"></param>
        /// <param name="materialReference"></param>
        /// <returns></returns>
        public static TMP_SubMeshUI AddSubTextObject(TextMeshProUGUI textComponent, MaterialReference materialReference)
        {
            GameObject go = new GameObject("TMP UI SubObject [" + materialReference.material.name + "]", typeof(RectTransform));

            go.transform.SetParent(textComponent.transform, false);
            go.layer = textComponent.gameObject.layer;

            RectTransform rectTransform = go.GetComponent<RectTransform>();
            rectTransform.anchorMin = Vector2.zero;
            rectTransform.anchorMax = Vector2.one;
            rectTransform.sizeDelta = Vector2.zero;
            rectTransform.pivot = textComponent.rectTransform.pivot;

            TMP_SubMeshUI subMesh = go.AddComponent<TMP_SubMeshUI>();

            subMesh.m_canvasRenderer = subMesh.canvasRenderer;
            subMesh.m_TextComponent = textComponent;

            subMesh.m_materialReferenceIndex = materialReference.index;
            subMesh.m_fontAsset = materialReference.fontAsset;
            subMesh.m_spriteAsset = materialReference.spriteAsset;
            subMesh.m_isDefaultMaterial = materialReference.isDefaultMaterial;
            subMesh.SetSharedMaterial(materialReference.material);

            return subMesh;
        }



        /// <summary>
        /// 
        /// </summary>
        protected override void OnEnable()
        {
            //Debug.Log("*** SubObject OnEnable() ***");

            // Register Callbacks for various events.
            if (!m_isRegisteredForEvents)
            {

            #if UNITY_EDITOR
                TMPro_EventManager.MATERIAL_PROPERTY_EVENT.Add(ON_MATERIAL_PROPERTY_CHANGED);
                TMPro_EventManager.FONT_PROPERTY_EVENT.Add(ON_FONT_PROPERTY_CHANGED);
                //TMPro_EventManager.TEXTMESHPRO_PROPERTY_EVENT.Add(ON_TEXTMESHPRO_PROPERTY_CHANGED);
                TMPro_EventManager.DRAG_AND_DROP_MATERIAL_EVENT.Add(ON_DRAG_AND_DROP_MATERIAL);
                //TMPro_EventManager.TEXT_STYLE_PROPERTY_EVENT.Add(ON_TEXT_STYLE_CHANGED);
                TMPro_EventManager.SPRITE_ASSET_PROPERTY_EVENT.Add(ON_SPRITE_ASSET_PROPERTY_CHANGED);
                //TMPro_EventManager.TMP_SETTINGS_PROPERTY_EVENT.Add(ON_TMP_SETTINGS_CHANGED);
            #endif

                m_isRegisteredForEvents = true;
            }

            m_ShouldRecalculateStencil = true;
            RecalculateClipping();
            RecalculateMasking();

            //SetAllDirty();
        }


        protected override void OnDisable()
        {
            //Debug.Log("*** SubObject OnDisable() ***");

            //m_canvasRenderer.Clear();
            TMP_UpdateRegistry.UnRegisterCanvasElementForRebuild(this);

            if (m_MaskMaterial != null)
            {
                TMP_MaterialManager.ReleaseStencilMaterial(m_MaskMaterial);
                m_MaskMaterial = null;
            }

            if (m_fallbackMaterial != null)
            {
                TMP_MaterialManager.ReleaseFallbackMaterial(m_fallbackMaterial);
                m_fallbackMaterial = null;
            }

            base.OnDisable();
        }


        protected override void OnDestroy()
        {
            //Debug.Log("*** OnDestroy() ***");

            // Destroy Mesh
            if (m_mesh != null) DestroyImmediate(m_mesh);

            if (m_MaskMaterial != null)
                TMP_MaterialManager.ReleaseStencilMaterial(m_MaskMaterial);

            if (m_fallbackMaterial != null)
            {
                TMP_MaterialManager.ReleaseFallbackMaterial(m_fallbackMaterial);
                m_fallbackMaterial = null;
            }

#if UNITY_EDITOR
            // Unregister the event this object was listening to
            TMPro_EventManager.MATERIAL_PROPERTY_EVENT.Remove(ON_MATERIAL_PROPERTY_CHANGED);
            TMPro_EventManager.FONT_PROPERTY_EVENT.Remove(ON_FONT_PROPERTY_CHANGED);
            //TMPro_EventManager.TEXTMESHPRO_PROPERTY_EVENT.Remove(ON_TEXTMESHPRO_PROPERTY_CHANGED);
            TMPro_EventManager.DRAG_AND_DROP_MATERIAL_EVENT.Remove(ON_DRAG_AND_DROP_MATERIAL);
            //TMPro_EventManager.TEXT_STYLE_PROPERTY_EVENT.Remove(ON_TEXT_STYLE_CHANGED);
            TMPro_EventManager.SPRITE_ASSET_PROPERTY_EVENT.Remove(ON_SPRITE_ASSET_PROPERTY_CHANGED);
            //TMPro_EventManager.TMP_SETTINGS_PROPERTY_EVENT.Remove(ON_TMP_SETTINGS_CHANGED);
        #endif

            m_isRegisteredForEvents = false;

            RecalculateClipping();
        }



#if UNITY_EDITOR
        // Event received when custom material editor properties are changed.
        void ON_MATERIAL_PROPERTY_CHANGED(bool isChanged, Material mat)
        {
            //Debug.Log("*** ON_MATERIAL_PROPERTY_CHANGED ***");

            int targetMaterialID = mat.GetInstanceID();
            int sharedMaterialID = m_sharedMaterial.GetInstanceID();
            int maskingMaterialID = m_MaskMaterial == null ? 0 : m_MaskMaterial.GetInstanceID();
            int fallbackSourceMaterialID = m_fallbackSourceMaterial == null ? 0 : m_fallbackSourceMaterial.GetInstanceID();

            // Filter events and return if the affected material is not this object's material.
            //if (targetMaterialID != sharedMaterialID && targetMaterialID != maskingMaterialID) return;

            // Filter events and return if the affected material is not this object's material.
            if (m_fallbackMaterial != null && fallbackSourceMaterialID == targetMaterialID)
                TMP_MaterialManager.CopyMaterialPresetProperties(mat, m_fallbackMaterial);

            if (m_TextComponent == null) m_TextComponent = GetComponentInParent<TextMeshProUGUI>();

            // Make sure material properties are synchronized between the assigned material and masking material.
            if (m_MaskMaterial != null)
            {
                UnityEditor.Undo.RecordObject(m_MaskMaterial, "Material Property Changes");
                UnityEditor.Undo.RecordObject(m_sharedMaterial, "Material Property Changes");

                if (targetMaterialID == sharedMaterialID)
                {
                    //Debug.Log("Copy base material properties to masking material if not null.");
                    float stencilID = m_MaskMaterial.GetFloat(ShaderUtilities.ID_StencilID);
                    float stencilComp = m_MaskMaterial.GetFloat(ShaderUtilities.ID_StencilComp);
                    m_MaskMaterial.CopyPropertiesFromMaterial(mat);
                    m_MaskMaterial.shaderKeywords = mat.shaderKeywords;

                    m_MaskMaterial.SetFloat(ShaderUtilities.ID_StencilID, stencilID);
                    m_MaskMaterial.SetFloat(ShaderUtilities.ID_StencilComp, stencilComp);
                }
                else if (targetMaterialID == maskingMaterialID)
                {
                    // Update the padding 
                    GetPaddingForMaterial(mat);

                    m_sharedMaterial.CopyPropertiesFromMaterial(mat);
                    m_sharedMaterial.shaderKeywords = mat.shaderKeywords;
                    m_sharedMaterial.SetFloat(ShaderUtilities.ID_StencilID, 0);
                    m_sharedMaterial.SetFloat(ShaderUtilities.ID_StencilComp, 8);
                }
                else if (fallbackSourceMaterialID == targetMaterialID)
                {
                    float stencilID = m_MaskMaterial.GetFloat(ShaderUtilities.ID_StencilID);
                    float stencilComp = m_MaskMaterial.GetFloat(ShaderUtilities.ID_StencilComp);
                    m_MaskMaterial.CopyPropertiesFromMaterial(m_fallbackMaterial);
                    m_MaskMaterial.shaderKeywords = m_fallbackMaterial.shaderKeywords;

                    m_MaskMaterial.SetFloat(ShaderUtilities.ID_StencilID, stencilID);
                    m_MaskMaterial.SetFloat(ShaderUtilities.ID_StencilComp, stencilComp);
                }
            }

            m_padding = GetPaddingForMaterial();

            SetVerticesDirty();
            m_ShouldRecalculateStencil = true;
            RecalculateClipping();
            RecalculateMasking();
        }


        // Event to Track Material Changed resulting from Drag-n-drop.
        void ON_DRAG_AND_DROP_MATERIAL(GameObject obj, Material currentMaterial, Material newMaterial)
        {
            // Check if event applies to this current object
            #if UNITY_2018_2_OR_NEWER
            if (obj == gameObject || UnityEditor.PrefabUtility.GetCorrespondingObjectFromSource(gameObject) == obj)
            #else
            if (obj == gameObject || UnityEditor.PrefabUtility.GetPrefabParent(gameObject) == obj)
            #endif
            {
                if (!m_isDefaultMaterial) return;

                // Make sure we have a valid reference to the renderer.
                if (m_canvasRenderer == null) m_canvasRenderer = GetComponent<CanvasRenderer>();

                UnityEditor.Undo.RecordObject(this, "Material Assignment");
                UnityEditor.Undo.RecordObject(m_canvasRenderer, "Material Assignment");

                SetSharedMaterial(newMaterial);
                m_TextComponent.havePropertiesChanged = true;
            }
        }

        // Event received when font asset properties are changed in Font Inspector
        void ON_SPRITE_ASSET_PROPERTY_CHANGED(bool isChanged, UnityEngine.Object obj)
        {
            //if (spriteSheet != null && (obj as TMP_SpriteAsset == m_spriteAsset || obj as Texture2D == m_spriteAsset.spriteSheet))
            //{
            if (m_TextComponent != null)
            {
                m_TextComponent.havePropertiesChanged = true;
                //m_TextComponent.SetVerticesDirty();
            }

            //}
        }

        // Event received when font asset properties are changed in Font Inspector
        void ON_FONT_PROPERTY_CHANGED(bool isChanged, TMP_FontAsset font)
        {
            if (m_fontAsset != null && font.GetInstanceID() == m_fontAsset.GetInstanceID())
            {
                // Copy Normal and Bold Weight
                if (m_fallbackMaterial != null)
                {
                    m_fallbackMaterial.SetFloat(ShaderUtilities.ID_WeightNormal, m_fontAsset.normalStyle);
                    m_fallbackMaterial.SetFloat(ShaderUtilities.ID_WeightBold, m_fontAsset.boldStyle);
                }
            }
        }

        /// <summary>
        /// Event received when the TMP Settings are changed.
        /// </summary>
        void ON_TMP_SETTINGS_CHANGED()
        {
            //Debug.Log("TMP Setting have changed.");
            //SetVerticesDirty();
            //SetMaterialDirty();
        }
#endif

        /// <summary>
        /// 
        /// </summary>
        protected override void OnTransformParentChanged()
        {
            if (!this.IsActive())
                return;

            m_ShouldRecalculateStencil = true;
            RecalculateClipping();
            RecalculateMasking();
        }


        /// <summary>
        /// Function returning the modified material for masking if necessary.
        /// </summary>
        /// <param name="baseMaterial"></param>
        /// <returns></returns>
        public override Material GetModifiedMaterial(Material baseMaterial)
        {
            Material mat = baseMaterial;

            if (m_ShouldRecalculateStencil)
            {
                m_StencilValue = TMP_MaterialManager.GetStencilID(gameObject);
                m_ShouldRecalculateStencil = false;
            }

            if (m_StencilValue > 0)
            {
                mat = TMP_MaterialManager.GetStencilMaterial(baseMaterial, m_StencilValue);
                if (m_MaskMaterial != null)
                    TMP_MaterialManager.ReleaseStencilMaterial(m_MaskMaterial);

                m_MaskMaterial = mat;
            }

            return mat;
        }


        /// <summary>
        /// Function called when the padding value for the material needs to be re-calculated.
        /// </summary>
        /// <returns></returns>
        public float GetPaddingForMaterial()
        {
            float padding = ShaderUtilities.GetPadding(m_sharedMaterial, m_TextComponent.extraPadding, m_TextComponent.isUsingBold);

            return padding;
        }


        /// <summary>
        /// Function called when the padding value for the material needs to be re-calculated.
        /// </summary>
        /// <returns></returns>
        public float GetPaddingForMaterial(Material mat)
        {
            float padding = ShaderUtilities.GetPadding(mat, m_TextComponent.extraPadding, m_TextComponent.isUsingBold);

            return padding;
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="isExtraPadding"></param>
        /// <param name="isBold"></param>
        public void UpdateMeshPadding(bool isExtraPadding, bool isUsingBold)
        {
            m_padding = ShaderUtilities.GetPadding(m_sharedMaterial, isExtraPadding, isUsingBold);
        }


        /// <summary>
        /// 
        /// </summary>
        public override void SetAllDirty()
        {
            //SetLayoutDirty();
            //SetVerticesDirty();
            //SetMaterialDirty();
        }


        /// <summary>
        /// 
        /// </summary>
        public override void SetVerticesDirty()
        {
            if (!this.IsActive())
                return;

            // This is called on the parent TextMeshPro component.
            if (m_TextComponent != null)
            {
                m_TextComponent.havePropertiesChanged = true;
                m_TextComponent.SetVerticesDirty();
            }
        }


        /// <summary>
        /// 
        /// </summary>
        public override void SetLayoutDirty()
        {

        }


        /// <summary>
        /// 
        /// </summary>
        public override void SetMaterialDirty()
        {
            //Debug.Log("*** STO-UI - SetMaterialDirty() *** FRAME (" + Time.frameCount + ")");

            //if (!this.IsActive())
            //    return;

            m_materialDirty = true;

            UpdateMaterial();

            if (m_OnDirtyMaterialCallback != null)
                m_OnDirtyMaterialCallback();

            //TMP_ITextElementUpdateManager.RegisterTextElementForGraphicRebuild(this);

            //TMP_UpdateRegistry.RegisterCanvasElementForGraphicRebuild((ICanvasElement)this);
            //m_TextComponent.SetMaterialDirty();
        }


        /// <summary>
        /// 
        /// </summary>
        public void SetPivotDirty()
        {
            if (!this.IsActive())
                return;

            this.rectTransform.pivot = m_TextComponent.rectTransform.pivot;
        }


        /// <summary>
        /// Override to Cull function of MaskableGraphic to prevent Culling.
        /// </summary>
        /// <param name="clipRect"></param>
        /// <param name="validRect"></param>
        public override void Cull(Rect clipRect, bool validRect)
        {
            if (m_TextComponent.ignoreRectMaskCulling) return;

            base.Cull(clipRect, validRect);
        }


        /// <summary>
        /// 
        /// </summary>
        protected override void UpdateGeometry()
        {
            // Need to override to prevent Unity from changing the geometry of the object.
            Debug.Log("UpdateGeometry()");
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="update"></param>
        public override void Rebuild(CanvasUpdate update)
        {
            if (update == CanvasUpdate.PreRender)
            {
                if (!m_materialDirty) return;

                UpdateMaterial();
                m_materialDirty = false;
            }
        }


        /// <summary>
        /// Function to update the material from the parent text object.
        /// </summary>
        public void RefreshMaterial()
        {
            UpdateMaterial();
        }


        /// <summary>
        /// 
        /// </summary>
        protected override void UpdateMaterial()
        {
            //Debug.Log("*** STO-UI - UpdateMaterial() *** FRAME (" + Time.frameCount + ")");

            //if (!this.IsActive())
            //    return;

            if (m_canvasRenderer == null) m_canvasRenderer = this.canvasRenderer;

            m_canvasRenderer.materialCount = 1;
            m_canvasRenderer.SetMaterial(materialForRendering, 0);
            m_canvasRenderer.SetTexture(mainTexture);

            #if UNITY_EDITOR
            if (m_sharedMaterial != null && gameObject.name != "TMP SubMeshUI [" + m_sharedMaterial.name + "]")
                gameObject.name = "TMP SubMeshUI [" + m_sharedMaterial.name + "]";
            #endif
        }


        // IClippable implementation
        /// <summary>
        /// Method called when the state of a parent changes.
        /// </summary>
        public override void RecalculateClipping()
        {
            //Debug.Log("*** RecalculateClipping() ***");
            base.RecalculateClipping();
        }


        /// <summary>
        /// 
        /// </summary>
        public override void RecalculateMasking()
        {
            //Debug.Log("RecalculateMasking()");

            this.m_ShouldRecalculateStencil = true;
            SetMaterialDirty();
        }



        /// <summary>
        /// Method which returns an instance of the shared material
        /// </summary>
        /// <returns></returns>
        Material GetMaterial()
        {
            // Make sure we have a valid reference to the renderer.
            //if (m_renderer == null) m_renderer = GetComponent<Renderer>();

            //if (m_material == null || m_isNewSharedMaterial)
            //{
            //    m_renderer.material = m_sharedMaterial;
            //    m_material = m_renderer.material;
            //    m_sharedMaterial = m_material;
            //    m_padding = ShaderUtilities.GetPadding(m_sharedMaterial, m_TextMeshPro.extraPadding, false);
            //    m_isNewSharedMaterial = false;
            //}

            return m_sharedMaterial;
        }


        // Function called internally when a new material is assigned via the fontMaterial property.
        Material GetMaterial(Material mat)
        {
            // Check in case Object is disabled. If so, we don't have a valid reference to the Renderer.
            // This can occur when the Duplicate Material Context menu is used on an inactive object.
            //if (m_renderer == null)
            //    m_renderer = GetComponent<Renderer>();

            // Create Instance Material only if the new material is not the same instance previously used.
            if (m_material == null || m_material.GetInstanceID() != mat.GetInstanceID())
                m_material = CreateMaterialInstance(mat);

            m_sharedMaterial = m_material;

            // Compute and Set new padding values for this new material. 
            m_padding = GetPaddingForMaterial();

            SetVerticesDirty();
            SetMaterialDirty();

            return m_sharedMaterial;
        }


        /// <summary>
        /// Method used to create an instance of the material
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        Material CreateMaterialInstance(Material source)
        {
            Material mat = new Material(source);
            mat.shaderKeywords = source.shaderKeywords;
            mat.name += " (Instance)";

            return mat;
        }


        /// <summary>
        /// Method returning the shared material assigned to the text object.
        /// </summary>
        /// <returns></returns>
        Material GetSharedMaterial()
        {
            if (m_canvasRenderer == null)
                m_canvasRenderer = GetComponent<CanvasRenderer>();

            return m_canvasRenderer.GetMaterial();
        }


        /// <summary>
        /// Method to set the shared material.
        /// </summary>
        /// <param name="mat"></param>
        void SetSharedMaterial(Material mat)
        {
            //Debug.Log("*** SetSharedMaterial UI() *** FRAME (" + Time.frameCount + ")");

            // Assign new material.
            m_sharedMaterial = mat;
            m_Material = m_sharedMaterial;

            //m_isDefaultMaterial = false;
            //if (mat.GetInstanceID() == m_fontAsset.material.GetInstanceID())
            //    m_isDefaultMaterial = true;

            // Compute and Set new padding values for this new material.
            m_padding = GetPaddingForMaterial();

            //SetVerticesDirty();
            SetMaterialDirty();

#if UNITY_EDITOR
            //if (m_sharedMaterial != null)
            //    gameObject.name = "TMP SubMesh [" + m_sharedMaterial.name + "]";
#endif
        }
    }
}

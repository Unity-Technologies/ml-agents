using UnityEngine;
using System;
using System.Collections;

#pragma warning disable 0109 // Disable warning due to conflict between Unity Editor DLL and Runtime DLL related to .renderer property being available in one but not the other. 

namespace TMPro
{
    [RequireComponent(typeof(MeshRenderer))]
    [RequireComponent(typeof(MeshFilter))]
    [ExecuteAlways]
    public class TMP_SubMesh : MonoBehaviour
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
        /// The material to be assigned to this object. Returns an instance of the material.
        /// </summary>
        public Material material
        {
            // Return a new Instance of the Material if none exists. Otherwise return the current Material Instance.
            get { return GetMaterial(m_sharedMaterial); }

            // Assign new font material
            set
            {
                if (m_sharedMaterial.GetInstanceID() == value.GetInstanceID())
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
        /// The fallback material created from the properties of the fallback source material.
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
        public new Renderer renderer
        {
            get { if (m_renderer == null) m_renderer = GetComponent<Renderer>();

                return m_renderer;
            }
        }
        [SerializeField]
        private Renderer m_renderer;


        /// <summary>
        /// The MeshFilter of this text sub object.
        /// </summary>
        public MeshFilter meshFilter
        {
            get { if (m_meshFilter == null) m_meshFilter = GetComponent<MeshFilter>();
                return m_meshFilter;
            }
        }
        [SerializeField]
        private MeshFilter m_meshFilter;


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
                    this.meshFilter.mesh = m_mesh;
                }

                return m_mesh;
            }
            set { m_mesh = value; }
        }
        private Mesh m_mesh;

        /// <summary>
        /// 
        /// </summary>
        //public BoxCollider boxCollider
        //{
        //    get
        //    {
        //        if (m_boxCollider == null)
        //        {
        //            //
        //            m_boxCollider = GetComponent<BoxCollider>();
        //            if (m_boxCollider == null)
        //            {
        //                m_boxCollider = gameObject.AddComponent<BoxCollider>();
        //                gameObject.AddComponent<Rigidbody>();
        //            }
        //        }

        //        return m_boxCollider;
        //    }
        //}
        //[SerializeField]
        //private BoxCollider m_boxCollider;

        [SerializeField]
        private TextMeshPro m_TextComponent;

        [NonSerialized]
        private bool m_isRegisteredForEvents;


        void OnEnable()
        {
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

            // Make the geometry visible when the object is enabled.
            meshFilter.sharedMesh = mesh;

            // Update _ClipRect values
            if (m_sharedMaterial != null)
                m_sharedMaterial.SetVector(ShaderUtilities.ID_ClipRect, new Vector4(-32767, -32767, 32767, 32767));
        }


        void OnDisable()
        {
            // Hide the geometry when the object is disabled.
            m_meshFilter.sharedMesh = null;

            if (m_fallbackMaterial != null)
            {
                TMP_MaterialManager.ReleaseFallbackMaterial(m_fallbackMaterial);
                m_fallbackMaterial = null;
            }


        }


        void OnDestroy()
        {
            // Destroy Mesh
            if (m_mesh != null) DestroyImmediate(m_mesh);

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
        }



        #if UNITY_EDITOR
        // Event received when custom material editor properties are changed.
        void ON_MATERIAL_PROPERTY_CHANGED(bool isChanged, Material mat)
        {
            //Debug.Log("*** ON_MATERIAL_PROPERTY_CHANGED ***");
            int targetMaterialID = mat.GetInstanceID();
            int sharedMaterialID = m_sharedMaterial.GetInstanceID();
            int fallbackSourceMaterialID = m_fallbackSourceMaterial == null ? 0 : m_fallbackSourceMaterial.GetInstanceID();

            // Filter events and return if the affected material is not this object's material.
            if (targetMaterialID != sharedMaterialID)
            {
                // Check if event applies to the source fallback material
                if (m_fallbackMaterial != null && fallbackSourceMaterialID == targetMaterialID)
                    TMP_MaterialManager.CopyMaterialPresetProperties(mat, m_fallbackMaterial);
                else
                    return;
            }

            if (m_TextComponent == null) m_TextComponent = GetComponentInParent<TextMeshPro>();

            m_padding = GetPaddingForMaterial();

            m_TextComponent.havePropertiesChanged = true;
            m_TextComponent.SetVerticesDirty();
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
                if (m_renderer == null) m_renderer = GetComponent<Renderer>();

                UnityEditor.Undo.RecordObject(this, "Material Assignment");
                UnityEditor.Undo.RecordObject(m_renderer, "Material Assignment");

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
        //    //Debug.Log("TMP Setting have changed.");
        //    //SetVerticesDirty();
        //    SetMaterialDirty();
        }
        #endif



        public static TMP_SubMesh AddSubTextObject(TextMeshPro textComponent, MaterialReference materialReference)
        {
            GameObject go = new GameObject("TMP SubMesh [" + materialReference.material.name + "]", typeof(TMP_SubMesh));

            TMP_SubMesh subMesh = go.GetComponent<TMP_SubMesh>();

            go.transform.SetParent(textComponent.transform, false);
            go.transform.localPosition = Vector3.zero;
            go.transform.localRotation = Quaternion.identity;
            go.transform.localScale = Vector3.one;
            go.layer = textComponent.gameObject.layer;

            subMesh.m_meshFilter = go.GetComponent<MeshFilter>();

            subMesh.m_TextComponent = textComponent;
            subMesh.m_fontAsset = materialReference.fontAsset;
            subMesh.m_spriteAsset = materialReference.spriteAsset;
            subMesh.m_isDefaultMaterial = materialReference.isDefaultMaterial;
            subMesh.SetSharedMaterial(materialReference.material);

            subMesh.renderer.sortingLayerID = textComponent.renderer.sortingLayerID;
            subMesh.renderer.sortingOrder = textComponent.renderer.sortingOrder;

            return subMesh;
        }


        public void DestroySelf()
        {
            Destroy(this.gameObject, 1f);
        }

        // Function called internally when a new material is assigned via the fontMaterial property.
        Material GetMaterial(Material mat)
        {
            // Check in case Object is disabled. If so, we don't have a valid reference to the Renderer.
            // This can occur when the Duplicate Material Context menu is used on an inactive object.
            if (m_renderer == null)
                m_renderer = GetComponent<Renderer>();

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
            if (m_renderer == null)
                m_renderer = GetComponent<Renderer>();

            return m_renderer.sharedMaterial;
        }


        /// <summary>
        /// Method to set the shared material.
        /// </summary>
        /// <param name="mat"></param>
        void SetSharedMaterial(Material mat)
        {
            //Debug.Log("*** SetSharedMaterial() *** FRAME (" + Time.frameCount + ")");

            // Assign new material.
            m_sharedMaterial = mat;

            // Compute and Set new padding values for this new material. 
            m_padding = GetPaddingForMaterial();

            SetMaterialDirty();

            #if UNITY_EDITOR
            if (m_sharedMaterial != null)
                gameObject.name = "TMP SubMesh [" + m_sharedMaterial.name + "]";
            #endif
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
        /// Function to update the padding values of the object.
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
        public void SetVerticesDirty()
        {
            if (!this.enabled)
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
        public void SetMaterialDirty()
        {
            //if (!this.enabled)
            //    return;

            UpdateMaterial();

            //m_materialDirty = true;
            //TMP_UpdateRegistry.RegisterCanvasElementForGraphicRebuild((ICanvasElement)this);
        }


        /// <summary>
        /// 
        /// </summary>
        protected void UpdateMaterial()
        {
            //Debug.Log("*** STO - UpdateMaterial() *** FRAME (" + Time.frameCount + ")");

            //if (!this.enabled)
            //    return;

            if (m_renderer == null) m_renderer = this.renderer;

            m_renderer.sharedMaterial = m_sharedMaterial;

            #if UNITY_EDITOR
            if (m_sharedMaterial != null && gameObject.name != "TMP SubMesh [" + m_sharedMaterial.name + "]")
                gameObject.name = "TMP SubMesh [" + m_sharedMaterial.name + "]";
            #endif
        }

        /// <summary>
        /// 
        /// </summary>
        //public void UpdateColliders(int vertexCount)
        //{
        //    if (this.boxCollider == null) return;

        //    Vector2 bl = TMP_Math.MAX_16BIT;
        //    Vector2 tr = TMP_Math.MIN_16BIT;
        //    // Compute the bounds of the sub text object mesh (excluding the transform position).
        //    for (int i = 0; i < vertexCount; i++)
        //    {
        //        bl.x = Mathf.Min(bl.x, m_mesh.vertices[i].x);
        //        bl.y = Mathf.Min(bl.y, m_mesh.vertices[i].y);

        //        tr.x = Mathf.Max(tr.x, m_mesh.vertices[i].x);
        //        tr.y = Mathf.Max(tr.y, m_mesh.vertices[i].y);
        //    }

        //    Vector3 center = (bl + tr) / 2;
        //    Vector3 size = tr - bl;
        //    size.z = .1f;
        //    this.boxCollider.center = center;
        //    this.boxCollider.size = size;
        //}
    }
}

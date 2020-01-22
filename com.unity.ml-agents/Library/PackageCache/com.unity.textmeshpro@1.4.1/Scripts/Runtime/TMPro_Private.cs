//#define TMP_PROFILE_ON
//#define TMP_PROFILE_PHASES_ON

using UnityEngine;
using UnityEngine.TextCore;
using System;
using System.Collections;
using System.Collections.Generic;

#pragma warning disable 0414 // Disabled a few warnings related to serialized variables not used in this script but used in the editor.

namespace TMPro
{

    public partial class TextMeshPro
    {
        [SerializeField]
        private bool m_hasFontAssetChanged = false; // Used to track when font properties have changed.

        float m_previousLossyScaleY = -1; // Used for Tracking lossy scale changes in the transform;

        [SerializeField]
        private Renderer m_renderer;
        private MeshFilter m_meshFilter;

        private bool m_isFirstAllocation; // Flag to determine if this is the first allocation of the buffers.
        private int m_max_characters = 8; // Determines the initial allocation and size of the character array / buffer.
        private int m_max_numberOfLines = 4; // Determines the initial allocation and maximum number of lines of text. 

        [SerializeField]
        protected TMP_SubMesh[] m_subTextObjects = new TMP_SubMesh[8];


        // MASKING RELATED PROPERTIES

        //MaterialPropertyBlock m_maskingPropertyBlock;
        //[SerializeField]
        private bool m_isMaskingEnabled;
        private bool isMaskUpdateRequired;
        //private bool m_isMaterialBlockSet;

        [SerializeField]
        private MaskingTypes m_maskType;

        // Matrix used to animated Env Map
        private Matrix4x4 m_EnvMapMatrix = new Matrix4x4();


        // Text Container / RectTransform Component
        private Vector3[] m_RectTransformCorners = new Vector3[4];

        [NonSerialized]
        private bool m_isRegisteredForEvents;


        // DEBUG Variables
        //private System.Diagnostics.Stopwatch m_StopWatch;
        //private bool isDebugOutputDone;
        //private int m_recursiveCount = 0;
        private int loopCountA;
        //private int loopCountB;
        //private int loopCountC;
        //private int loopCountD;
        //private int loopCountE;


        protected override void Awake()
        {
            //Debug.Log("***** Awake() called on object ID " + GetInstanceID() + ". *****");

            #if UNITY_EDITOR
            // Special handling for TMP Settings and importing Essential Resources
            if (TMP_Settings.instance == null)
            {
                if (m_isWaitingOnResourceLoad == false)
                    TMPro_EventManager.RESOURCE_LOAD_EVENT.Add(ON_RESOURCES_LOADED);

                m_isWaitingOnResourceLoad = true;
                return;
            }
            #endif

            // Cache Reference to the Mesh Renderer.
            m_renderer = GetComponent<Renderer>();
            if (m_renderer == null)
                m_renderer = gameObject.AddComponent<Renderer>();

            // Make sure we have a CanvasRenderer for compatibility reasons and hide it
            if (this.canvasRenderer != null)
                this.canvasRenderer.hideFlags = HideFlags.HideInInspector;
            else
            {
                CanvasRenderer canvasRenderer = gameObject.AddComponent<CanvasRenderer>();
                canvasRenderer.hideFlags = HideFlags.HideInInspector;
            }

            // Cache Reference to RectTransform
            m_rectTransform = this.rectTransform;

            // Cache Reference to the transform;
            m_transform = this.transform;

            // Cache a reference to the Mesh Filter.
            m_meshFilter = GetComponent<MeshFilter>();
            if (m_meshFilter == null)
                m_meshFilter = gameObject.AddComponent<MeshFilter>();

            // Create new Mesh if necessary and cache reference to it.
            if (m_mesh == null)
            {
                m_mesh = new Mesh();
                m_mesh.hideFlags = HideFlags.HideAndDontSave;

                m_meshFilter.mesh = m_mesh;

                // Create new TextInfo for the text object.
                m_textInfo = new TMP_TextInfo(this);
            }
            m_meshFilter.hideFlags = HideFlags.HideInInspector;

            // Load TMP Settings for new text object instances.
            LoadDefaultSettings();

            // Load the font asset and assign material to renderer.
            LoadFontAsset();

            // Load Default TMP StyleSheet
            TMP_StyleSheet.LoadDefaultStyleSheet();

            // Allocate our initial buffers.
            if (m_TextParsingBuffer == null)
                m_TextParsingBuffer = new UnicodeChar[m_max_characters];

            m_cached_TextElement = new TMP_Character();
            m_isFirstAllocation = true;

            // Check if we have a font asset assigned. Return if we don't because no one likes to see purple squares on screen.
            if (m_fontAsset == null)
            {
                Debug.LogWarning("Please assign a Font Asset to this " + transform.name + " gameobject.", this);
                return;
            }

            // Check to make sure Sub Text Objects are tracked correctly in the event a Prefab is used.
            TMP_SubMesh[] subTextObjects = GetComponentsInChildren<TMP_SubMesh>();
            if (subTextObjects.Length > 0)
            {
                for (int i = 0; i < subTextObjects.Length; i++)
                    m_subTextObjects[i + 1] = subTextObjects[i];
            }

            // Set flags to ensure our text is parsed and redrawn.
            m_isInputParsingRequired = true;
            m_havePropertiesChanged = true;
            m_isCalculateSizeRequired = true;

            m_isAwake = true;
        }


        protected override void OnEnable()
        {
            //Debug.Log("***** OnEnable() called on object ID " + GetInstanceID() + ". *****");

            // Return if Awake() has not been called on the text object.
            if (m_isAwake == false)
                return;

            // Register Callbacks for various events.
            if (!m_isRegisteredForEvents)
            {
                #if UNITY_EDITOR
                TMPro_EventManager.MATERIAL_PROPERTY_EVENT.Add(ON_MATERIAL_PROPERTY_CHANGED);
                TMPro_EventManager.FONT_PROPERTY_EVENT.Add(ON_FONT_PROPERTY_CHANGED);
                TMPro_EventManager.TEXTMESHPRO_PROPERTY_EVENT.Add(ON_TEXTMESHPRO_PROPERTY_CHANGED);
                TMPro_EventManager.DRAG_AND_DROP_MATERIAL_EVENT.Add(ON_DRAG_AND_DROP_MATERIAL);
                TMPro_EventManager.TEXT_STYLE_PROPERTY_EVENT.Add(ON_TEXT_STYLE_CHANGED);
                TMPro_EventManager.COLOR_GRADIENT_PROPERTY_EVENT.Add(ON_COLOR_GRADIENT_CHANGED);
                TMPro_EventManager.TMP_SETTINGS_PROPERTY_EVENT.Add(ON_TMP_SETTINGS_CHANGED);
                #endif
                m_isRegisteredForEvents = true;
            }

            TMP_UpdateManager.RegisterTextObjectForUpdate(this);

            meshFilter.sharedMesh = mesh;
            SetActiveSubMeshes(true);

            // Schedule potential text object update (if any of the properties have changed.
            ComputeMarginSize();

            m_isInputParsingRequired = true;
            m_havePropertiesChanged = true;
            m_verticesAlreadyDirty = false;
            SetVerticesDirty();
        }


        protected override void OnDisable()
        {
            //Debug.Log("***** OnDisable() called on object ID " + GetInstanceID() + ". *****");

            // Return if Awake() has not been called on the text object.
            if (m_isAwake == false)
                return;

            TMP_UpdateManager.UnRegisterTextElementForRebuild(this);
            TMP_UpdateManager.UnRegisterTextObjectForUpdate(this);

            m_meshFilter.sharedMesh = null;
            SetActiveSubMeshes(false);
        }


        protected override void OnDestroy()
        {
            //Debug.Log("***** OnDestroy() called on object ID " + GetInstanceID() + ". *****");

            // Destroy the mesh if we have one.
            if (m_mesh != null)
            {
                DestroyImmediate(m_mesh);
            }

            // Unregister the event this object was listening to
            #if UNITY_EDITOR
            TMPro_EventManager.MATERIAL_PROPERTY_EVENT.Remove(ON_MATERIAL_PROPERTY_CHANGED);
            TMPro_EventManager.FONT_PROPERTY_EVENT.Remove(ON_FONT_PROPERTY_CHANGED);
            TMPro_EventManager.TEXTMESHPRO_PROPERTY_EVENT.Remove(ON_TEXTMESHPRO_PROPERTY_CHANGED);
            TMPro_EventManager.DRAG_AND_DROP_MATERIAL_EVENT.Remove(ON_DRAG_AND_DROP_MATERIAL);
            TMPro_EventManager.TEXT_STYLE_PROPERTY_EVENT.Remove(ON_TEXT_STYLE_CHANGED);
            TMPro_EventManager.COLOR_GRADIENT_PROPERTY_EVENT.Remove(ON_COLOR_GRADIENT_CHANGED);
            TMPro_EventManager.TMP_SETTINGS_PROPERTY_EVENT.Remove(ON_TMP_SETTINGS_CHANGED);
            TMPro_EventManager.RESOURCE_LOAD_EVENT.Remove(ON_RESOURCES_LOADED);
            #endif

            m_isRegisteredForEvents = false;
            TMP_UpdateManager.UnRegisterTextElementForRebuild(this);
            TMP_UpdateManager.UnRegisterTextObjectForUpdate(this);
        }


        #if UNITY_EDITOR
        protected override void Reset()
        {
            //Debug.Log("Reset() has been called." + m_subTextObjects);
            
            // Return if Awake() has not been called on the text object.
            if (m_isAwake == false)
                return;

            if (m_mesh != null)
                DestroyImmediate(m_mesh);

            Awake();
        }


        protected override void OnValidate()
        {
            //Debug.Log("*** TextMeshPro OnValidate() has been called on Object ID:" + gameObject.GetInstanceID());
            
            // Return if Awake() has not been called on the text object.
            if (m_isAwake == false)
                return;

            // Additional Properties could be added to sync up Serialized Properties & Properties.

            // Handle Font Asset changes in the inspector
            if (m_fontAsset == null || m_hasFontAssetChanged)
            {
                LoadFontAsset();
                m_isCalculateSizeRequired = true;
                m_hasFontAssetChanged = false;
            }

            m_padding = GetPaddingForMaterial();
            ComputeMarginSize();

            m_isInputParsingRequired = true;
            m_inputSource = TextInputSources.Text;
            m_havePropertiesChanged = true;
            m_isCalculateSizeRequired = true;
            m_isPreferredWidthDirty = true;
            m_isPreferredHeightDirty = true;

            SetAllDirty();
        }


        // Event received when TMP resources have been loaded.
        void ON_RESOURCES_LOADED()
        {
            TMPro_EventManager.RESOURCE_LOAD_EVENT.Remove(ON_RESOURCES_LOADED);

            if (this == null)
                return;

            Awake();
            OnEnable();
        }


        // Event received when custom material editor properties are changed.
        void ON_MATERIAL_PROPERTY_CHANGED(bool isChanged, Material mat)
        {
            //Debug.Log("ON_MATERIAL_PROPERTY_CHANGED event received. Targeted Material is: " + mat.name + "  m_sharedMaterial: " + m_sharedMaterial.name + "  m_renderer.sharedMaterial: " + m_renderer.sharedMaterial);         

            if (m_renderer.sharedMaterial == null)
            {
                if (m_fontAsset != null)
                {
                    m_renderer.sharedMaterial = m_fontAsset.material;
                    Debug.LogWarning("No Material was assigned to " + name + ". " + m_fontAsset.material.name + " was assigned.", this);
                }
                else
                    Debug.LogWarning("No Font Asset assigned to " + name + ". Please assign a Font Asset.", this);
            }

            if (m_fontAsset.atlasTexture != null && m_fontAsset.atlasTexture.GetInstanceID() != m_renderer.sharedMaterial.GetTexture(ShaderUtilities.ID_MainTex).GetInstanceID())
            {
                m_renderer.sharedMaterial = m_sharedMaterial;
                //m_renderer.sharedMaterial = m_fontAsset.material;
                Debug.LogWarning("Font Asset Atlas doesn't match the Atlas in the newly assigned material. Select a matching material or a different font asset.", this);
            }

            if (m_renderer.sharedMaterial != m_sharedMaterial) //    || m_renderer.sharedMaterials.Contains(mat))
            {
                //Debug.Log("ON_MATERIAL_PROPERTY_CHANGED Called on Target ID: " + GetInstanceID() + ". Previous Material:" + m_sharedMaterial + "  New Material:" + m_renderer.sharedMaterial); // on Object ID:" + GetInstanceID() + ". m_sharedMaterial: " + m_sharedMaterial.name + "  m_renderer.sharedMaterial: " + m_renderer.sharedMaterial.name);         
                m_sharedMaterial = m_renderer.sharedMaterial;
            }

            m_padding = GetPaddingForMaterial();
            //m_sharedMaterialHashCode = TMP_TextUtilities.GetSimpleHashCode(m_sharedMaterial.name);

            UpdateMask();
            UpdateEnvMapMatrix();
            m_havePropertiesChanged = true;
            SetVerticesDirty();
        }


        // Event received when font asset properties are changed in Font Inspector
        void ON_FONT_PROPERTY_CHANGED(bool isChanged, TMP_FontAsset font)
        {
            if (MaterialReference.Contains(m_materialReferences, font))
            {
                //Debug.Log("ON_FONT_PROPERTY_CHANGED event received.");
                m_isInputParsingRequired = true;
                m_havePropertiesChanged = true;

                UpdateMeshPadding();

                SetMaterialDirty();
                SetVerticesDirty();
            }
        }

     
        // Event received when UNDO / REDO Event alters the properties of the object.
        void ON_TEXTMESHPRO_PROPERTY_CHANGED(bool isChanged, TextMeshPro obj)
        {
            if (obj == this)
            {
                //Debug.Log("Undo / Redo Event Received by Object ID:" + GetInstanceID());
                m_havePropertiesChanged = true;
                m_isInputParsingRequired = true;

                m_padding = GetPaddingForMaterial();
                ComputeMarginSize(); // Verify this change
                SetVerticesDirty();
            }
        }


        // Event to Track Material Changed resulting from Drag-n-drop.
        void ON_DRAG_AND_DROP_MATERIAL(GameObject obj, Material currentMaterial, Material newMaterial)
        {
            //Debug.Log("Drag-n-Drop Event - Receiving Object ID " + GetInstanceID()); // + ". Target Object ID " + obj.GetInstanceID() + ".  New Material is " + mat.name + " with ID " + mat.GetInstanceID() + ". Base Material is " + m_baseMaterial.name + " with ID " + m_baseMaterial.GetInstanceID());

            // Check if event applies to this current object
            #if UNITY_2018_2_OR_NEWER
            if (obj == gameObject || UnityEditor.PrefabUtility.GetCorrespondingObjectFromSource(gameObject) == obj)
            #else
            if (obj == gameObject || UnityEditor.PrefabUtility.GetPrefabParent(gameObject) == obj)
            #endif
            {
                UnityEditor.Undo.RecordObject(this, "Material Assignment");
                UnityEditor.Undo.RecordObject(m_renderer, "Material Assignment");

                m_sharedMaterial = newMaterial;

                m_padding = GetPaddingForMaterial();
                m_havePropertiesChanged = true;
                SetVerticesDirty();
                SetMaterialDirty();
            }
        }


        // Event received when Text Styles are changed.
        void ON_TEXT_STYLE_CHANGED(bool isChanged)
        {
            m_havePropertiesChanged = true;
            m_isInputParsingRequired = true;
            SetVerticesDirty();
        }


        /// <summary>
        /// Event received when a Color Gradient Preset is modified.
        /// </summary>
        /// <param name="textObject"></param>
        void ON_COLOR_GRADIENT_CHANGED(TMP_ColorGradient gradient)
        {
            if (m_fontColorGradientPreset != null && gradient.GetInstanceID() == m_fontColorGradientPreset.GetInstanceID())
            {
                m_havePropertiesChanged = true;
                SetVerticesDirty();
            }
        }


        /// <summary>
        /// Event received when the TMP Settings are changed.
        /// </summary>
        void ON_TMP_SETTINGS_CHANGED()
        {
            m_defaultSpriteAsset = null;
            m_havePropertiesChanged = true;
            m_isInputParsingRequired = true;
            SetAllDirty();
        }
#endif


            // Function which loads either the default font or a newly assigned font asset. This function also assigned the appropriate material to the renderer.
        protected override void LoadFontAsset()
        {
            //Debug.Log("TextMeshPro LoadFontAsset() has been called."); // Current Font Asset is " + (font != null ? font.name: "Null") );

            ShaderUtilities.GetShaderPropertyIDs(); // Initialize & Get shader property IDs.

            if (m_fontAsset == null)
            {
                if (TMP_Settings.defaultFontAsset != null)
                    m_fontAsset =TMP_Settings.defaultFontAsset;
                else
                    m_fontAsset = Resources.Load<TMP_FontAsset>("Fonts & Materials/LiberationSans SDF");

                if (m_fontAsset == null)
                {
                    Debug.LogWarning("The LiberationSans SDF Font Asset was not found. There is no Font Asset assigned to " + gameObject.name + ".", this);
                    return;
                }

                if (m_fontAsset.characterLookupTable == null)
                {
                    Debug.Log("Dictionary is Null!");
                }

                m_renderer.sharedMaterial = m_fontAsset.material;
                m_sharedMaterial = m_fontAsset.material;
                m_sharedMaterial.SetFloat("_CullMode", 0);
                m_sharedMaterial.SetFloat(ShaderUtilities.ShaderTag_ZTestMode, 4);
                m_renderer.receiveShadows = false;
                m_renderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off; // true;
                // Get a Reference to the Shader
            }
            else
            {
                if (m_fontAsset.characterLookupTable == null)
                {
                    //Debug.Log("Reading Font Definition and Creating Character Dictionary.");
                    m_fontAsset.ReadFontAssetDefinition();
                }

                //Debug.Log("Font Asset name:" + font.material.name);

                // If font atlas texture doesn't match the assigned material font atlas, switch back to default material specified in the Font Asset.
                if (m_renderer.sharedMaterial == null || m_renderer.sharedMaterial.GetTexture(ShaderUtilities.ID_MainTex) == null || m_fontAsset.atlasTexture.GetInstanceID() != m_renderer.sharedMaterial.GetTexture(ShaderUtilities.ID_MainTex).GetInstanceID())
                {
                    m_renderer.sharedMaterial = m_fontAsset.material;
                    m_sharedMaterial = m_fontAsset.material; 
                }
                else
                {
                    m_sharedMaterial = m_renderer.sharedMaterial;
                }

                //m_sharedMaterial.SetFloat("_CullMode", 0);
                m_sharedMaterial.SetFloat(ShaderUtilities.ShaderTag_ZTestMode, 4);

                // Check if we are using the SDF Surface Shader
                if (m_sharedMaterial.passCount == 1)
                {
                    m_renderer.receiveShadows = false;
                    m_renderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                }

            }

            m_padding = GetPaddingForMaterial();
            //m_alignmentPadding = ShaderUtilities.GetFontExtent(m_sharedMaterial);
            m_isMaskingEnabled = ShaderUtilities.IsMaskingEnabled(m_sharedMaterial);


            // Find and cache Underline & Ellipsis characters.
            GetSpecialCharacters(m_fontAsset);


            //m_sharedMaterials.Add(m_sharedMaterial);
            //m_sharedMaterialHashCode = TMP_TextUtilities.GetSimpleHashCode(m_sharedMaterial.name);
            // Hide Material Editor Component
            //m_renderer.sharedMaterial.hideFlags = HideFlags.None;
        }


        void UpdateEnvMapMatrix()
        {
            if (!m_sharedMaterial.HasProperty(ShaderUtilities.ID_EnvMap) || m_sharedMaterial.GetTexture(ShaderUtilities.ID_EnvMap) == null)
                return;

            //Debug.Log("Updating Env Matrix...");
            Vector3 rotation = m_sharedMaterial.GetVector(ShaderUtilities.ID_EnvMatrixRotation);
            m_EnvMapMatrix = Matrix4x4.TRS(Vector3.zero, Quaternion.Euler(rotation), Vector3.one);

            m_sharedMaterial.SetMatrix(ShaderUtilities.ID_EnvMatrix, m_EnvMapMatrix);
        }


        //
        void SetMask(MaskingTypes maskType)
        {
            switch(maskType)
            {
                case MaskingTypes.MaskOff:
                    m_sharedMaterial.DisableKeyword(ShaderUtilities.Keyword_MASK_SOFT);
                    m_sharedMaterial.DisableKeyword(ShaderUtilities.Keyword_MASK_HARD);
                    m_sharedMaterial.DisableKeyword(ShaderUtilities.Keyword_MASK_TEX);
                    break;
                case MaskingTypes.MaskSoft:
                    m_sharedMaterial.EnableKeyword(ShaderUtilities.Keyword_MASK_SOFT);
                    m_sharedMaterial.DisableKeyword(ShaderUtilities.Keyword_MASK_HARD);
                    m_sharedMaterial.DisableKeyword(ShaderUtilities.Keyword_MASK_TEX);
                    break;
                case MaskingTypes.MaskHard:
                    m_sharedMaterial.EnableKeyword(ShaderUtilities.Keyword_MASK_HARD);
                    m_sharedMaterial.DisableKeyword(ShaderUtilities.Keyword_MASK_SOFT);
                    m_sharedMaterial.DisableKeyword(ShaderUtilities.Keyword_MASK_TEX);
                    break;
                //case MaskingTypes.MaskTex:
                //    m_sharedMaterial.EnableKeyword(ShaderUtilities.Keyword_MASK_TEX);
                //    m_sharedMaterial.DisableKeyword(ShaderUtilities.Keyword_MASK_HARD);
                //    m_sharedMaterial.DisableKeyword(ShaderUtilities.Keyword_MASK_SOFT);
                //    break;
            }
        }


        // Method used to set the masking coordinates
        void SetMaskCoordinates(Vector4 coords)
        {
            m_sharedMaterial.SetVector(ShaderUtilities.ID_ClipRect, coords);
        }

        // Method used to set the masking coordinates
        void SetMaskCoordinates(Vector4 coords, float softX, float softY)
        {
            m_sharedMaterial.SetVector(ShaderUtilities.ID_ClipRect, coords);
            m_sharedMaterial.SetFloat(ShaderUtilities.ID_MaskSoftnessX, softX);
            m_sharedMaterial.SetFloat(ShaderUtilities.ID_MaskSoftnessY, softY);
        }



        // Enable Masking in the Shader
        void EnableMasking()
        {
            if (m_sharedMaterial.HasProperty(ShaderUtilities.ID_ClipRect))
            {
                m_sharedMaterial.EnableKeyword(ShaderUtilities.Keyword_MASK_SOFT);
                m_sharedMaterial.DisableKeyword(ShaderUtilities.Keyword_MASK_HARD);
                m_sharedMaterial.DisableKeyword(ShaderUtilities.Keyword_MASK_TEX);

                m_isMaskingEnabled = true;
                UpdateMask();
            }
        }


        // Enable Masking in the Shader
        void DisableMasking()
        {
            if (m_sharedMaterial.HasProperty(ShaderUtilities.ID_ClipRect))
            {
                m_sharedMaterial.DisableKeyword(ShaderUtilities.Keyword_MASK_SOFT);
                m_sharedMaterial.DisableKeyword(ShaderUtilities.Keyword_MASK_HARD);
                m_sharedMaterial.DisableKeyword(ShaderUtilities.Keyword_MASK_TEX);

                m_isMaskingEnabled = false;
                UpdateMask();
            }
        }


        void UpdateMask()
        {
            //Debug.Log("UpdateMask() called.");
            
            if (!m_isMaskingEnabled)
            {
                // Release Masking Material

                // Re-assign Base Material

                return;
            }
            
            if (m_isMaskingEnabled && m_fontMaterial == null)
            {
                CreateMaterialInstance();
            }

            
            /*
            if (!m_isMaskingEnabled)
            {
                //Debug.Log("Masking is not enabled.");
                if (m_maskingPropertyBlock != null)
                {
                    m_renderer.SetPropertyBlock(null);
                    //havePropertiesChanged = true;
                }
                return;
            }
            //else
            //    Debug.Log("Updating Masking...");
            */
             
            // Compute Masking Coordinates & Softness
            //float softnessX = Mathf.Min(Mathf.Min(m_textContainer.margins.x, m_textContainer.margins.z), m_sharedMaterial.GetFloat(ShaderUtilities.ID_MaskSoftnessX));
            //float softnessY = Mathf.Min(Mathf.Min(m_textContainer.margins.y, m_textContainer.margins.w), m_sharedMaterial.GetFloat(ShaderUtilities.ID_MaskSoftnessY));

            //softnessX = softnessX > 0 ? softnessX : 0;
            //softnessY = softnessY > 0 ? softnessY : 0;
           
            //float width = (m_textContainer.width - Mathf.Max(m_textContainer.margins.x, 0) - Mathf.Max(m_textContainer.margins.z, 0)) / 2 + softnessX;
            //float height =  (m_textContainer.height - Mathf.Max(m_textContainer.margins.y, 0) - Mathf.Max(m_textContainer.margins.w, 0)) / 2 + softnessY;
          
            //Vector2 center = new Vector2((0.5f - m_textContainer.pivot.x) * m_textContainer.width + (Mathf.Max(m_textContainer.margins.x, 0) - Mathf.Max(m_textContainer.margins.z, 0)) / 2, (0.5f - m_textContainer.pivot.y) * m_textContainer.height + (- Mathf.Max(m_textContainer.margins.y, 0) + Mathf.Max(m_textContainer.margins.w, 0)) / 2);                           
            //Vector4 mask = new Vector4(center.x, center.y, width, height);


            //m_fontMaterial.SetVector(ShaderUtilities.ID_ClipRect, mask);
            //m_fontMaterial.SetFloat(ShaderUtilities.ID_MaskSoftnessX, softnessX);
            //m_fontMaterial.SetFloat(ShaderUtilities.ID_MaskSoftnessY, softnessY);

            /*                     
            if(m_maskingPropertyBlock == null)
            {                
                m_maskingPropertyBlock = new MaterialPropertyBlock();
         
                //m_maskingPropertyBlock.AddFloat(ShaderUtilities.ID_VertexOffsetX,  m_sharedMaterial.GetFloat(ShaderUtilities.ID_VertexOffsetX));
                //m_maskingPropertyBlock.AddFloat(ShaderUtilities.ID_VertexOffsetY,  m_sharedMaterial.GetFloat(ShaderUtilities.ID_VertexOffsetY));
                //Debug.Log("Creating new MaterialPropertyBlock.");
            }

            //Debug.Log("Updating Material Property Block.");
            //m_maskingPropertyBlock.Clear();
            m_maskingPropertyBlock.AddFloat(ShaderUtilities.ID_MaskID, m_renderer.GetInstanceID());
            m_maskingPropertyBlock.AddVector(ShaderUtilities.ID_MaskCoord, mask);
            m_maskingPropertyBlock.AddFloat(ShaderUtilities.ID_MaskSoftnessX, softnessX);
            m_maskingPropertyBlock.AddFloat(ShaderUtilities.ID_MaskSoftnessY, softnessY);
           
            m_renderer.SetPropertyBlock(m_maskingPropertyBlock);
            */
        }


        // Function called internally when a new material is assigned via the fontMaterial property.
        protected override Material GetMaterial(Material mat)
        {
            // Check in case Object is disabled. If so, we don't have a valid reference to the Renderer.
            // This can occur when the Duplicate Material Context menu is used on an inactive object.
            //if (m_renderer == null)
            //    m_renderer = GetComponent<Renderer>();

            // Create Instance Material only if the new material is not the same instance previously used.
            if (m_fontMaterial == null || m_fontMaterial.GetInstanceID() != mat.GetInstanceID())
                m_fontMaterial = CreateMaterialInstance(mat);

            m_sharedMaterial = m_fontMaterial;

            m_padding = GetPaddingForMaterial();

            SetVerticesDirty();
            SetMaterialDirty();

            return m_sharedMaterial;
        }


        /// <summary>
        /// Method returning instances of the materials used by the text object.
        /// </summary>
        /// <returns></returns>
        protected override Material[] GetMaterials(Material[] mats)
        {
            int materialCount = m_textInfo.materialCount;

            if (m_fontMaterials == null)
                m_fontMaterials = new Material[materialCount];
            else if (m_fontMaterials.Length != materialCount)
                TMP_TextInfo.Resize(ref m_fontMaterials, materialCount, false);

            // Get instances of the materials
            for (int i = 0; i < materialCount; i++)
            {
                if (i == 0)
                    m_fontMaterials[i] = fontMaterial;
                else
                    m_fontMaterials[i] = m_subTextObjects[i].material;
            }

            m_fontSharedMaterials = m_fontMaterials;

            return m_fontMaterials;
        }


        // Function called internally when a new shared material is assigned via the fontSharedMaterial property.
        protected override void SetSharedMaterial(Material mat)
        {
            // Check in case Object is disabled. If so, we don't have a valid reference to the Renderer.
            // This can occur when the Duplicate Material Context menu is used on an inactive object.
            //if (m_renderer == null)
            //    m_renderer = GetComponent<Renderer>();

            m_sharedMaterial = mat;

            m_padding = GetPaddingForMaterial();

            SetMaterialDirty();
        }


        /// <summary>
        /// Method returning an array containing the materials used by the text object.
        /// </summary>
        /// <returns></returns>
        protected override Material[] GetSharedMaterials()
        {
            int materialCount = m_textInfo.materialCount;

            if (m_fontSharedMaterials == null)
                m_fontSharedMaterials = new Material[materialCount];
            else if (m_fontSharedMaterials.Length != materialCount)
                TMP_TextInfo.Resize(ref m_fontSharedMaterials, materialCount, false);

            for (int i = 0; i < materialCount; i++)
            {
                if (i == 0)
                    m_fontSharedMaterials[i] = m_sharedMaterial;
                else
                    m_fontSharedMaterials[i] = m_subTextObjects[i].sharedMaterial;
            }

            return m_fontSharedMaterials;
        }


        /// <summary>
        /// Method used to assign new materials to the text and sub text objects.
        /// </summary>
        protected override void SetSharedMaterials(Material[] materials)
        {
            int materialCount = m_textInfo.materialCount;

            // Check allocation of the fontSharedMaterials array.
            if (m_fontSharedMaterials == null)
                m_fontSharedMaterials = new Material[materialCount];
            else if (m_fontSharedMaterials.Length != materialCount)
                TMP_TextInfo.Resize(ref m_fontSharedMaterials, materialCount, false);

            // Only assign as many materials as the text object contains.
            for (int i = 0; i < materialCount; i++)
            {
                Texture mat_MainTex = materials[i].GetTexture(ShaderUtilities.ID_MainTex);

                if (i == 0)
                {
                    // Only assign new material if the font atlas textures match.
                    if ( mat_MainTex == null || mat_MainTex.GetInstanceID() != m_sharedMaterial.GetTexture(ShaderUtilities.ID_MainTex).GetInstanceID())
                        continue;

                    m_sharedMaterial = m_fontSharedMaterials[i] = materials[i];
                    m_padding = GetPaddingForMaterial(m_sharedMaterial);
                }
                else
                {
                    // Only assign new material if the font atlas textures match.
                    if (mat_MainTex == null || mat_MainTex.GetInstanceID() != m_subTextObjects[i].sharedMaterial.GetTexture(ShaderUtilities.ID_MainTex).GetInstanceID())
                        continue;

                    // Only assign a new material if none were specified in the text input.
                    if (m_subTextObjects[i].isDefaultMaterial)
                        m_subTextObjects[i].sharedMaterial = m_fontSharedMaterials[i] = materials[i];
                }
            }
        }


        // This function will create an instance of the Font Material.
        protected override void SetOutlineThickness(float thickness)
        {
            thickness = Mathf.Clamp01(thickness);
            m_renderer.material.SetFloat(ShaderUtilities.ID_OutlineWidth, thickness);

            if (m_fontMaterial == null)
                m_fontMaterial = m_renderer.material;

            m_fontMaterial = m_renderer.material;
            m_sharedMaterial = m_fontMaterial;
            m_padding = GetPaddingForMaterial();
        }


        // This function will create an instance of the Font Material.
        protected override void SetFaceColor(Color32 color)
        {
            m_renderer.material.SetColor(ShaderUtilities.ID_FaceColor, color);

            if (m_fontMaterial == null)
                m_fontMaterial = m_renderer.material;

            m_sharedMaterial = m_fontMaterial;
        }


        // This function will create an instance of the Font Material.
        protected override void SetOutlineColor(Color32 color)
        {
            m_renderer.material.SetColor(ShaderUtilities.ID_OutlineColor, color);

            if (m_fontMaterial == null)
                m_fontMaterial = m_renderer.material;

            //Debug.Log("Material ID:" + m_fontMaterial.GetInstanceID());
            m_sharedMaterial = m_fontMaterial;
        }


        // Function used to create an instance of the material
        void CreateMaterialInstance()
        {
            Material mat = new Material(m_sharedMaterial);
            mat.shaderKeywords = m_sharedMaterial.shaderKeywords;

            //mat.hideFlags = HideFlags.DontSave;
            mat.name += " Instance";

            m_fontMaterial = mat;
        }


        // Sets the Render Queue and Ztest mode 
        protected override void SetShaderDepth()
        {
            if (m_isOverlay)
            {
                // Changing these properties results in an instance of the material
                m_sharedMaterial.SetFloat(ShaderUtilities.ShaderTag_ZTestMode, 0);
                //m_renderer.material.SetFloat("_ZTestMode", 8);
                m_renderer.material.renderQueue = 4000;

                m_sharedMaterial = m_renderer.material;
                //Debug.Log("Text set to Overlay mode.");
            }
            else
            {
                // Should this use an instanced material?
                m_sharedMaterial.SetFloat(ShaderUtilities.ShaderTag_ZTestMode, 4);
                m_renderer.material.renderQueue = -1;
                
                m_sharedMaterial = m_renderer.material;
                //Debug.Log("Text set to Normal mode.");
            }
        }


        // Sets the Culling mode of the material
        protected override void SetCulling()
        {
            if (m_isCullingEnabled)
            {
                m_renderer.material.SetFloat("_CullMode", 2);

                for (int i = 1; i < m_subTextObjects.Length && m_subTextObjects[i] != null; i++)
                {
                    Renderer renderer = m_subTextObjects[i].renderer;

                    if (renderer != null)
                    {
                        renderer.material.SetFloat(ShaderUtilities.ShaderTag_CullMode, 2);
                    }
                }
            }
            else
            {
                m_renderer.material.SetFloat("_CullMode", 0);

                for (int i = 1; i < m_subTextObjects.Length && m_subTextObjects[i] != null; i++)
                {
                    Renderer renderer = m_subTextObjects[i].renderer;

                    if (renderer != null)
                    {
                        renderer.material.SetFloat(ShaderUtilities.ShaderTag_CullMode, 0);
                    }
                }
            }
        }


        // Set Perspective Correction Mode based on whether Camera is Orthographic or Perspective
        void SetPerspectiveCorrection()
        {
            if (m_isOrthographic)
                m_sharedMaterial.SetFloat(ShaderUtilities.ID_PerspectiveFilter, 0.0f);
            else
                m_sharedMaterial.SetFloat(ShaderUtilities.ID_PerspectiveFilter, 0.875f);
        }


        /// <summary>
        /// Get the padding value for the currently assigned material.
        /// </summary>
        /// <returns></returns>
        protected override float GetPaddingForMaterial(Material mat)
        {
            m_padding = ShaderUtilities.GetPadding(mat, m_enableExtraPadding, m_isUsingBold);
            m_isMaskingEnabled = ShaderUtilities.IsMaskingEnabled(m_sharedMaterial);
            m_isSDFShader = mat.HasProperty(ShaderUtilities.ID_WeightNormal);

            return m_padding;
        }


        /// <summary>
        /// Get the padding value for the currently assigned material.
        /// </summary>
        /// <returns></returns>
        protected override float GetPaddingForMaterial()
        {
            ShaderUtilities.GetShaderPropertyIDs();

            if (m_sharedMaterial == null) return 0;

            m_padding = ShaderUtilities.GetPadding(m_sharedMaterial, m_enableExtraPadding, m_isUsingBold);
            m_isMaskingEnabled = ShaderUtilities.IsMaskingEnabled(m_sharedMaterial);
            m_isSDFShader = m_sharedMaterial.HasProperty(ShaderUtilities.ID_WeightNormal);

            return m_padding;
        }


        // This function parses through the Char[] to determine how many characters will be visible. It then makes sure the arrays are large enough for all those characters.
        protected override int SetArraySizes(UnicodeChar[] chars)
        {
            //Debug.Log("*** SetArraySizes() ***");

            int spriteCount = 0;

            m_totalCharacterCount = 0;
            m_isUsingBold = false;
            m_isParsingText = false;
            tag_NoParsing = false;
            m_FontStyleInternal = m_fontStyle;

            m_FontWeightInternal = (m_FontStyleInternal & FontStyles.Bold) == FontStyles.Bold ? FontWeight.Bold : m_fontWeight;
            m_FontWeightStack.SetDefault(m_FontWeightInternal);

            m_currentFontAsset = m_fontAsset;
            m_currentMaterial = m_sharedMaterial;
            m_currentMaterialIndex = 0;

            m_materialReferenceStack.SetDefault(new MaterialReference(m_currentMaterialIndex, m_currentFontAsset, null, m_currentMaterial, m_padding));

            m_materialReferenceIndexLookup.Clear();
            MaterialReference.AddMaterialReference(m_currentMaterial, m_currentFontAsset, m_materialReferences, m_materialReferenceIndexLookup);

            if (m_textInfo == null) m_textInfo = new TMP_TextInfo();
            m_textElementType = TMP_TextElementType.Character;

            // Clear Linked Text object if we have one.
            if (m_linkedTextComponent != null)
            {
                m_linkedTextComponent.text = string.Empty;
                m_linkedTextComponent.ForceMeshUpdate();
            }

            // Parsing XML tags in the text
            for (int i = 0; i < chars.Length && chars[i].unicode != 0; i++)
            {
                //Make sure the characterInfo array can hold the next text element.
                if (m_textInfo.characterInfo == null || m_totalCharacterCount >= m_textInfo.characterInfo.Length)
                    TMP_TextInfo.Resize(ref m_textInfo.characterInfo, m_totalCharacterCount + 1, true);

                int unicode = (int)chars[i].unicode;

                // PARSE XML TAGS
                #region PARSE XML TAGS
                if (m_isRichText && unicode == 60) // if Char '<'
                {
                    int prev_MaterialIndex = m_currentMaterialIndex;
                    int endTagIndex;

                    // Check if Tag is Valid
                    if (ValidateHtmlTag(chars, i + 1, out endTagIndex))
                    {
                        int tagStartIndex = chars[i].stringIndex;
                        i = endTagIndex;

                        if ((m_FontStyleInternal & FontStyles.Bold) == FontStyles.Bold) m_isUsingBold = true;

                        if (m_textElementType == TMP_TextElementType.Sprite)
                        {
                            m_materialReferences[m_currentMaterialIndex].referenceCount += 1;

                            m_textInfo.characterInfo[m_totalCharacterCount].character = (char)(57344 + m_spriteIndex);
                            m_textInfo.characterInfo[m_totalCharacterCount].spriteIndex = m_spriteIndex;
                            m_textInfo.characterInfo[m_totalCharacterCount].fontAsset = m_currentFontAsset;
                            m_textInfo.characterInfo[m_totalCharacterCount].spriteAsset = m_currentSpriteAsset;
                            m_textInfo.characterInfo[m_totalCharacterCount].materialReferenceIndex = m_currentMaterialIndex;
                            m_textInfo.characterInfo[m_totalCharacterCount].textElement = m_currentSpriteAsset.spriteCharacterTable[m_spriteIndex];
                            m_textInfo.characterInfo[m_totalCharacterCount].elementType = m_textElementType;
                            m_textInfo.characterInfo[m_totalCharacterCount].index = tagStartIndex;
                            m_textInfo.characterInfo[m_totalCharacterCount].stringLength = chars[i].stringIndex - tagStartIndex + 1;

                            // Restore element type and material index to previous values.
                            m_textElementType = TMP_TextElementType.Character;
                            m_currentMaterialIndex = prev_MaterialIndex;

                            spriteCount += 1;
                            m_totalCharacterCount += 1;
                        }

                        continue;
                    }
                }
                #endregion

                bool isUsingAlternativeTypeface = false;
                bool isUsingFallbackOrAlternativeTypeface = false;

                TMP_Character character;
                TMP_FontAsset tempFontAsset;
                TMP_FontAsset prev_fontAsset = m_currentFontAsset;
                Material prev_material = m_currentMaterial;
                int prev_materialIndex = m_currentMaterialIndex;


                // Handle Font Styles like LowerCase, UpperCase and SmallCaps.
                #region Handling of LowerCase, UpperCase and SmallCaps Font Styles
                if (m_textElementType == TMP_TextElementType.Character)
                {
                    if ((m_FontStyleInternal & FontStyles.UpperCase) == FontStyles.UpperCase)
                    {
                        // If this character is lowercase, switch to uppercase.
                        if (char.IsLower((char)unicode))
                            unicode = char.ToUpper((char)unicode);

                    }
                    else if ((m_FontStyleInternal & FontStyles.LowerCase) == FontStyles.LowerCase)
                    {
                        // If this character is uppercase, switch to lowercase.
                        if (char.IsUpper((char)unicode))
                            unicode = char.ToLower((char)unicode);
                    }
                    else if ((m_FontStyleInternal & FontStyles.SmallCaps) == FontStyles.SmallCaps)
                    {
                        // Only convert lowercase characters to uppercase.
                        if (char.IsLower((char)unicode))
                            unicode = char.ToUpper((char)unicode);
                    }
                }
                #endregion


                // Lookup the Glyph data for each character and cache it.
                #region LOOKUP GLYPH
                character = TMP_FontAssetUtilities.GetCharacterFromFontAsset((uint)unicode, m_currentFontAsset, false, m_FontStyleInternal, m_FontWeightInternal, out isUsingAlternativeTypeface, out tempFontAsset);

                // Search for the glyph in the list of fallback assigned to the primary font asset.
                if (character == null)
                {
                    if (m_currentFontAsset.fallbackFontAssetTable != null && m_currentFontAsset.fallbackFontAssetTable.Count > 0)
                        character = TMP_FontAssetUtilities.GetCharacterFromFontAssets((uint)unicode, m_currentFontAsset.fallbackFontAssetTable, true, m_FontStyleInternal, m_FontWeightInternal, out isUsingAlternativeTypeface, out tempFontAsset);
                }

                // Search for the glyph in the Sprite Asset assigned to the text object.
                if (character == null)
                {
                    TMP_SpriteAsset spriteAsset = this.spriteAsset;

                    if (spriteAsset != null)
                    {
                        int spriteIndex = -1;

                        // Check Default Sprite Asset and its Fallbacks
                        spriteAsset = TMP_SpriteAsset.SearchForSpriteByUnicode(spriteAsset, (uint)unicode, true, out spriteIndex);

                        if (spriteIndex != -1)
                        {
                            m_textElementType = TMP_TextElementType.Sprite;
                            m_textInfo.characterInfo[m_totalCharacterCount].elementType = m_textElementType;

                            m_currentMaterialIndex = MaterialReference.AddMaterialReference(spriteAsset.material, spriteAsset, m_materialReferences, m_materialReferenceIndexLookup);
                            m_materialReferences[m_currentMaterialIndex].referenceCount += 1;

                            m_textInfo.characterInfo[m_totalCharacterCount].character = (char)unicode;
                            m_textInfo.characterInfo[m_totalCharacterCount].spriteIndex = spriteIndex;
                            m_textInfo.characterInfo[m_totalCharacterCount].fontAsset = m_currentFontAsset;
                            m_textInfo.characterInfo[m_totalCharacterCount].spriteAsset = spriteAsset;
                            m_textInfo.characterInfo[m_totalCharacterCount].textElement = spriteAsset.spriteCharacterTable[m_spriteIndex];
                            m_textInfo.characterInfo[m_totalCharacterCount].materialReferenceIndex = m_currentMaterialIndex;
                            m_textInfo.characterInfo[m_totalCharacterCount].index = chars[i].stringIndex;
                            m_textInfo.characterInfo[m_totalCharacterCount].stringLength = chars[i].length;

                            // Restore element type and material index to previous values.
                            m_textElementType = TMP_TextElementType.Character;
                            m_currentMaterialIndex = prev_materialIndex;

                            spriteCount += 1;
                            m_totalCharacterCount += 1;

                            continue;
                        }
                    }
                }

                // Search for the glyph in the list of fallback assigned in the TMP Settings (General Fallbacks).
                if (character == null)
                {
                    if (TMP_Settings.fallbackFontAssets != null && TMP_Settings.fallbackFontAssets.Count > 0)
                        character = TMP_FontAssetUtilities.GetCharacterFromFontAssets((uint)unicode, TMP_Settings.fallbackFontAssets, true, m_FontStyleInternal, m_FontWeightInternal, out isUsingAlternativeTypeface, out tempFontAsset);
                }

                // Search for the glyph in the Default Font Asset assigned in the TMP Settings file.
                if (character == null)
                {
                    if (TMP_Settings.defaultFontAsset != null)
                        character = TMP_FontAssetUtilities.GetCharacterFromFontAsset((uint)unicode, TMP_Settings.defaultFontAsset, true, m_FontStyleInternal, m_FontWeightInternal, out isUsingAlternativeTypeface, out tempFontAsset);
                }

                // TODO: Add support for using Sprite Assets like a special Emoji only Sprite Asset when UTF16 or UTF32 glyphs are requested.
                // This would kind of mirror native Emoji support.
                if (character == null)
                {
                    TMP_SpriteAsset spriteAsset = TMP_Settings.defaultSpriteAsset;

                    if (spriteAsset != null)
                    {
                        int spriteIndex = -1;

                        // Check Default Sprite Asset and its Fallbacks
                        spriteAsset = TMP_SpriteAsset.SearchForSpriteByUnicode(spriteAsset, (uint)unicode, true, out spriteIndex);

                        if (spriteIndex != -1)
                        {
                            m_textElementType = TMP_TextElementType.Sprite;
                            m_textInfo.characterInfo[m_totalCharacterCount].elementType = m_textElementType;

                            m_currentMaterialIndex = MaterialReference.AddMaterialReference(spriteAsset.material, spriteAsset, m_materialReferences, m_materialReferenceIndexLookup);
                            m_materialReferences[m_currentMaterialIndex].referenceCount += 1;

                            m_textInfo.characterInfo[m_totalCharacterCount].character = (char)unicode;
                            m_textInfo.characterInfo[m_totalCharacterCount].spriteIndex = spriteIndex;
                            m_textInfo.characterInfo[m_totalCharacterCount].fontAsset = m_currentFontAsset;
                            m_textInfo.characterInfo[m_totalCharacterCount].spriteAsset = spriteAsset;
                            m_textInfo.characterInfo[m_totalCharacterCount].textElement = spriteAsset.spriteCharacterTable[m_spriteIndex];
                            m_textInfo.characterInfo[m_totalCharacterCount].materialReferenceIndex = m_currentMaterialIndex;
                            m_textInfo.characterInfo[m_totalCharacterCount].index = chars[i].stringIndex;
                            m_textInfo.characterInfo[m_totalCharacterCount].stringLength = chars[i].length;

                            // Restore element type and material index to previous values.
                            m_textElementType = TMP_TextElementType.Character;
                            m_currentMaterialIndex = prev_materialIndex;

                            spriteCount += 1;
                            m_totalCharacterCount += 1;

                            continue;
                        }

                    }
                }

                //Check if Lowercase or Uppercase variant of the character is available.
                // Not sure this is necessary anyone as it is very unlikely with recursive search through fallback fonts.
                //if (glyph == null)
                //{
                //    if (char.IsLower((char)c))
                //    {
                //        if (m_currentFontAsset.characterDictionary.TryGetValue(char.ToUpper((char)c), out glyph))
                //            c = chars[i] = char.ToUpper((char)c);
                //    }
                //    else if (char.IsUpper((char)c))
                //    {
                //        if (m_currentFontAsset.characterDictionary.TryGetValue(char.ToLower((char)c), out glyph))
                //            c = chars[i] = char.ToLower((char)c);
                //    }
                //}

                // Replace missing glyph by the Square (9633) glyph or possibly the Space (32) glyph.
                if (character == null)
                {
                    // Save the original unicode character
                    int srcGlyph = unicode;

                    // Try replacing the missing glyph character by TMP Settings Missing Glyph or Square (9633) character.
                    unicode = chars[i].unicode = TMP_Settings.missingGlyphCharacter == 0 ? 9633 : TMP_Settings.missingGlyphCharacter;

                    // Check for the missing glyph character in the currently assigned font asset and its fallbacks
                    character = TMP_FontAssetUtilities.GetCharacterFromFontAsset((uint)unicode, m_currentFontAsset, true, m_FontStyleInternal, m_FontWeightInternal, out isUsingAlternativeTypeface, out tempFontAsset);

                    if (character == null)
                    {
                        // Search for the missing glyph character in the TMP Settings Fallback list.
                        if (TMP_Settings.fallbackFontAssets != null && TMP_Settings.fallbackFontAssets.Count > 0)
                            character = TMP_FontAssetUtilities.GetCharacterFromFontAssets((uint)unicode, TMP_Settings.fallbackFontAssets, true, m_FontStyleInternal, m_FontWeightInternal, out isUsingAlternativeTypeface, out tempFontAsset);
                    }

                    if (character == null)
                    {
                        // Search for the missing glyph in the TMP Settings Default Font Asset.
                        if (TMP_Settings.defaultFontAsset != null)
                            character = TMP_FontAssetUtilities.GetCharacterFromFontAsset((uint)unicode, TMP_Settings.defaultFontAsset, true, m_FontStyleInternal, m_FontWeightInternal, out isUsingAlternativeTypeface, out tempFontAsset);
                    }

                    if (character == null)
                    {
                        // Use Space (32) Glyph from the currently assigned font asset.
                        unicode = chars[i].unicode = 32;
                        character = TMP_FontAssetUtilities.GetCharacterFromFontAsset((uint)unicode, m_currentFontAsset, true, m_FontStyleInternal, m_FontWeightInternal, out isUsingAlternativeTypeface, out tempFontAsset);
                        if (!TMP_Settings.warningsDisabled)
                            Debug.LogWarning("Character with ASCII value of " + srcGlyph + " was not found in the Font Asset Glyph Table. It was replaced by a space.", this);
                    }
                }

                // Determine if the font asset is still the current font asset or a fallback.
                if (tempFontAsset != null)
                {
                    if (tempFontAsset.GetInstanceID() != m_currentFontAsset.GetInstanceID())
                    {
                        isUsingFallbackOrAlternativeTypeface = true;
                        m_currentFontAsset = tempFontAsset;
                    }
                }
                #endregion

                m_textInfo.characterInfo[m_totalCharacterCount].elementType = TMP_TextElementType.Character;
                m_textInfo.characterInfo[m_totalCharacterCount].textElement = character;
                m_textInfo.characterInfo[m_totalCharacterCount].isUsingAlternateTypeface = isUsingAlternativeTypeface;
                m_textInfo.characterInfo[m_totalCharacterCount].character = (char)unicode;
                m_textInfo.characterInfo[m_totalCharacterCount].fontAsset = m_currentFontAsset;
                m_textInfo.characterInfo[m_totalCharacterCount].index = chars[i].stringIndex;
                m_textInfo.characterInfo[m_totalCharacterCount].stringLength = chars[i].length;

                if (isUsingFallbackOrAlternativeTypeface)
                {
                    // Create Fallback material instance matching current material preset if necessary
                    if (TMP_Settings.matchMaterialPreset)
                        m_currentMaterial = TMP_MaterialManager.GetFallbackMaterial(m_currentMaterial, m_currentFontAsset.material);
                    else
                        m_currentMaterial = m_currentFontAsset.material;

                    m_currentMaterialIndex = MaterialReference.AddMaterialReference(m_currentMaterial, m_currentFontAsset, m_materialReferences, m_materialReferenceIndexLookup);
                }

                if (!char.IsWhiteSpace((char)unicode) && unicode != 0x200B)
                {
                    // Limit the mesh of the main text object to 65535 vertices and use sub objects for the overflow.
                    if (m_materialReferences[m_currentMaterialIndex].referenceCount < 16383)
                        m_materialReferences[m_currentMaterialIndex].referenceCount += 1;
                    else
                    {
                        m_currentMaterialIndex = MaterialReference.AddMaterialReference(new Material(m_currentMaterial), m_currentFontAsset, m_materialReferences, m_materialReferenceIndexLookup);
                        m_materialReferences[m_currentMaterialIndex].referenceCount += 1;
                    }
                }

                m_textInfo.characterInfo[m_totalCharacterCount].material = m_currentMaterial;
                m_textInfo.characterInfo[m_totalCharacterCount].materialReferenceIndex = m_currentMaterialIndex;
                m_materialReferences[m_currentMaterialIndex].isFallbackMaterial = isUsingFallbackOrAlternativeTypeface;

                // Restore previous font asset and material if fallback font was used.
                if (isUsingFallbackOrAlternativeTypeface)
                {
                    m_materialReferences[m_currentMaterialIndex].fallbackMaterial = prev_material;
                    m_currentFontAsset = prev_fontAsset;
                    m_currentMaterial = prev_material;
                    m_currentMaterialIndex = prev_materialIndex;
                }

                m_totalCharacterCount += 1;
            }

            // Early return if we are calculating the preferred values.
            if (m_isCalculatingPreferredValues)
            {
                m_isCalculatingPreferredValues = false;
                m_isInputParsingRequired = true;
                return m_totalCharacterCount;
            }

            // Save material and sprite count.
            m_textInfo.spriteCount = spriteCount;
            int materialCount = m_textInfo.materialCount = m_materialReferenceIndexLookup.Count;

            // Check if we need to resize the MeshInfo array for handling different materials.
            if (materialCount > m_textInfo.meshInfo.Length)
                TMP_TextInfo.Resize(ref m_textInfo.meshInfo, materialCount, false);

            // Resize SubTextObject array if necessary
            if (materialCount > m_subTextObjects.Length)
                TMP_TextInfo.Resize(ref m_subTextObjects, Mathf.NextPowerOfTwo(materialCount + 1));

            // Resize CharacterInfo[] if allocations are excessive
            if (m_textInfo.characterInfo.Length - m_totalCharacterCount > 256)
                TMP_TextInfo.Resize(ref m_textInfo.characterInfo, Mathf.Max(m_totalCharacterCount + 1, 256), true);


            // Iterate through the material references to set the mesh buffer allocations
            for (int i = 0; i < materialCount; i++)
            {
                // Add new sub text object for each material reference
                if (i > 0)
                {
                    if (m_subTextObjects[i] == null)
                    {
                        m_subTextObjects[i] = TMP_SubMesh.AddSubTextObject(this, m_materialReferences[i]);

                        // Not sure this is necessary
                        m_textInfo.meshInfo[i].vertices = null;
                    }
                    //else if (m_subTextObjects[i].gameObject.activeInHierarchy == false)
                    //    m_subTextObjects[i].gameObject.SetActive(true);

                    // Check if the material has changed.
                    if (m_subTextObjects[i].sharedMaterial == null || m_subTextObjects[i].sharedMaterial.GetInstanceID() != m_materialReferences[i].material.GetInstanceID())
                    {
                        bool isDefaultMaterial = m_materialReferences[i].isDefaultMaterial;

                        m_subTextObjects[i].isDefaultMaterial = isDefaultMaterial;

                        // Assign new material if we are not using the default material or if the font asset has changed.
                        if (!isDefaultMaterial || m_subTextObjects[i].sharedMaterial == null || m_subTextObjects[i].sharedMaterial.GetTexture(ShaderUtilities.ID_MainTex).GetInstanceID() != m_materialReferences[i].material.GetTexture(ShaderUtilities.ID_MainTex).GetInstanceID())
                        {
                            m_subTextObjects[i].sharedMaterial = m_materialReferences[i].material;
                            m_subTextObjects[i].fontAsset = m_materialReferences[i].fontAsset;
                            m_subTextObjects[i].spriteAsset = m_materialReferences[i].spriteAsset;
                        }
                    }

                    // Check if we need to use a Fallback Material
                    if (m_materialReferences[i].isFallbackMaterial)
                    {
                        m_subTextObjects[i].fallbackMaterial = m_materialReferences[i].material;
                        m_subTextObjects[i].fallbackSourceMaterial = m_materialReferences[i].fallbackMaterial;
                    }

                }

                int referenceCount = m_materialReferences[i].referenceCount;

                // Check to make sure our buffers allocations can accommodate the required text elements.
                if (m_textInfo.meshInfo[i].vertices == null || m_textInfo.meshInfo[i].vertices.Length < referenceCount * (!m_isVolumetricText ? 4 : 8))
                {
                    if (m_textInfo.meshInfo[i].vertices == null)
                    {
                        if (i == 0)
                            m_textInfo.meshInfo[i] = new TMP_MeshInfo(m_mesh, referenceCount + 1, m_isVolumetricText);
                        else
                            m_textInfo.meshInfo[i] = new TMP_MeshInfo(m_subTextObjects[i].mesh, referenceCount + 1, m_isVolumetricText);
                    }
                    else
                        m_textInfo.meshInfo[i].ResizeMeshInfo(referenceCount > 1024 ? referenceCount + 256 : Mathf.NextPowerOfTwo(referenceCount + 1), m_isVolumetricText);
                }
                else if (m_VertexBufferAutoSizeReduction && referenceCount > 0 && m_textInfo.meshInfo[i].vertices.Length - referenceCount * (!m_isVolumetricText ? 4 : 8) > 1024)
                {
                    // Resize vertex buffers if allocations are excessive.
                    //Debug.Log("Reducing the size of the vertex buffers.");
                    m_textInfo.meshInfo[i].ResizeMeshInfo(referenceCount > 1024 ? referenceCount + 256 : Mathf.NextPowerOfTwo(referenceCount + 1), m_isVolumetricText);
                }
            }

            //TMP_MaterialManager.CleanupFallbackMaterials();

            // Clean up unused SubMeshes
            for (int i = materialCount; i < m_subTextObjects.Length && m_subTextObjects[i] != null; i++)
            {
                if (i < m_textInfo.meshInfo.Length)
                    m_textInfo.meshInfo[i].ClearUnusedVertices(0, true);

                //m_subTextObjects[i].gameObject.SetActive(false);
            }

            return m_totalCharacterCount;
        }


        // Added to sort handle the potential issue with OnWillRenderObject() not getting called when objects are not visible by camera.
        //void OnBecameInvisible()
        //{
        //    if (m_mesh != null)
        //        m_mesh.bounds = new Bounds(transform.position, new Vector3(1000, 1000, 0));
        //}


        /// <summary>
        /// Update the margin width and height
        /// </summary>
        public override void ComputeMarginSize()
        {
            if (this.rectTransform != null)
            {
                //Debug.Log("*** ComputeMarginSize() *** Current RectTransform's Width is " + m_rectTransform.rect.width + " and Height is " + m_rectTransform.rect.height); // + " and size delta is "  + m_rectTransform.sizeDelta);

                m_marginWidth = m_rectTransform.rect.width - m_margin.x - m_margin.z;
                m_marginHeight = m_rectTransform.rect.height - m_margin.y - m_margin.w;

                // Update the corners of the RectTransform
                m_RectTransformCorners = GetTextContainerLocalCorners();
            }
        }


        protected override void OnDidApplyAnimationProperties()
        {
            //Debug.Log("*** OnDidApplyAnimationProperties() ***");
            m_havePropertiesChanged = true;
            isMaskUpdateRequired = true;

            SetVerticesDirty();
        }


        protected override void OnTransformParentChanged()
        {
            //Debug.Log("*** OnTransformParentChanged() ***");
            //ComputeMarginSize();

            SetVerticesDirty();
            SetLayoutDirty();
        }


        protected override void OnRectTransformDimensionsChange()
        {
            //Debug.Log("*** OnRectTransformDimensionsChange() ***");
            ComputeMarginSize();

            SetVerticesDirty();
            SetLayoutDirty();
        }


        /// <summary>
        /// Function used as a replacement for LateUpdate to check if the transform or scale of the text object has changed.
        /// </summary>
        internal override void InternalUpdate()
        {
            // We need to update the SDF scale or possibly regenerate the text object if lossy scale has changed.
            if (m_havePropertiesChanged == false)
            {
                float lossyScaleY = m_rectTransform.lossyScale.y;
                if (lossyScaleY != m_previousLossyScaleY && m_text != string.Empty && m_text != null)
                {
                    float scaleDelta = lossyScaleY / m_previousLossyScaleY;

                    UpdateSDFScale(scaleDelta);

                    m_previousLossyScaleY = lossyScaleY;
                }
            }

            // Added to handle legacy animation mode.
            if (m_isUsingLegacyAnimationComponent)
            {
                //if (m_havePropertiesChanged)
                m_havePropertiesChanged = true;
                OnPreRenderObject();
            }
        }


        /// <summary>
        /// Function called when the text needs to be updated.
        /// </summary>
        void OnPreRenderObject()
        {
            //Debug.Log("*** OnPreRenderObject() ***");

            if (!m_isAwake || (this.IsActive() == false && m_ignoreActiveState == false)) return;

            // Debug Variables
            loopCountA = 0;
            //loopCountB = 0;
            //loopCountC = 0;
            //loopCountD = 0;
            //loopCountE = 0;

            if (m_havePropertiesChanged || m_isLayoutDirty)
            {
                //Debug.Log("Properties have changed!"); // Assigned Material is:" + m_sharedMaterial); // New Text is: " + m_text + ".");

                if (isMaskUpdateRequired)
                {
                    UpdateMask();
                    isMaskUpdateRequired = false;
                }

                // Update mesh padding if necessary.
                if (checkPaddingRequired)
                    UpdateMeshPadding();

                // Reparse the text if the input has changed or text was truncated.
                if (m_isInputParsingRequired || m_isTextTruncated)
                    ParseInputText();

                // Reset Font min / max used with Auto-sizing
                if (m_enableAutoSizing)
                    m_fontSize = Mathf.Clamp(m_fontSizeBase, m_fontSizeMin, m_fontSizeMax);

                m_maxFontSize = m_fontSizeMax;
                m_minFontSize = m_fontSizeMin;
                m_lineSpacingDelta = 0;
                m_charWidthAdjDelta = 0;
                //m_recursiveCount = 0;

                m_isCharacterWrappingEnabled = false;
                m_isTextTruncated = false;

                m_havePropertiesChanged = false;
                m_isLayoutDirty = false;
                m_ignoreActiveState = false;

                GenerateTextMesh();
            }
        }


        /// <summary>
        /// This is the main function that is responsible for creating / displaying the text.
        /// </summary>
        protected override void GenerateTextMesh()
        {
            //Debug.Log("***** GenerateTextMesh() *****"); // ***** Frame: " + Time.frameCount); // + ". Point Size: " + m_fontSize + ". Margins are (W) " + m_marginWidth + "  (H) " + m_marginHeight); // ". Iteration Count: " + loopCountA + ".  Min: " + m_minFontSize + "  Max: " + m_maxFontSize + "  Delta: " + (m_maxFontSize - m_minFontSize) + "  Font size is " + m_fontSize); //called for Object with ID " + GetInstanceID()); // Assigned Material is " + m_uiRenderer.GetMaterial().name); // IncludeForMasking " + this.m_IncludeForMasking); // and text is " + m_text);

            // Early exit if no font asset was assigned. This should not be needed since LiberationSans SDF will be assigned by default.
            if (m_fontAsset == null || m_fontAsset.characterLookupTable == null)
            {
                Debug.LogWarning("Can't Generate Mesh! No Font Asset has been assigned to Object ID: " + this.GetInstanceID());
                return;
            }

            // Clear TextInfo
            if (m_textInfo != null)
                m_textInfo.Clear();

            // Early exit if we don't have any Text to generate.
            if (m_TextParsingBuffer == null || m_TextParsingBuffer.Length == 0 || m_TextParsingBuffer[0].unicode == (char)0)
            {
                // Clear mesh and upload changes to the mesh.
                ClearMesh(true);

                m_preferredWidth = 0;
                m_preferredHeight = 0;

                // Event indicating the text has been regenerated.
                TMPro_EventManager.ON_TEXT_CHANGED(this);

                return;
            }

            m_currentFontAsset = m_fontAsset;
            m_currentMaterial = m_sharedMaterial;
            m_currentMaterialIndex = 0;
            m_materialReferenceStack.SetDefault(new MaterialReference(m_currentMaterialIndex, m_currentFontAsset, null, m_currentMaterial, m_padding));

            m_currentSpriteAsset = m_spriteAsset;

            // Stop  all Sprite Animations
            if (m_spriteAnimator != null)
                m_spriteAnimator.StopAllAnimations();

            // Total character count is computed when the text is parsed.
            int totalCharacterCount = m_totalCharacterCount;

            // Calculate the scale of the font based on selected font size and sampling point size.
            // baseScale is calculated using the font asset assigned to the text object.
            float baseScale = m_fontScale = (m_fontSize / m_fontAsset.faceInfo.pointSize * m_fontAsset.faceInfo.scale * (m_isOrthographic ? 1 : 0.1f));
            float currentElementScale = baseScale;
            m_fontScaleMultiplier = 1;

            m_currentFontSize = m_fontSize;
            m_sizeStack.SetDefault(m_currentFontSize);
            float fontSizeDelta = 0;

            int charCode = 0; // Holds the character code of the currently being processed character.

            m_FontStyleInternal = m_fontStyle; // Set the default style.
            m_FontWeightInternal = (m_FontStyleInternal & FontStyles.Bold) == FontStyles.Bold ? FontWeight.Bold : m_fontWeight;
            m_FontWeightStack.SetDefault(m_FontWeightInternal);
            m_fontStyleStack.Clear();

            m_lineJustification = m_textAlignment; // Sets the line justification mode to match editor alignment.
            m_lineJustificationStack.SetDefault(m_lineJustification);

            float padding = 0;
            float style_padding = 0; // Extra padding required to accommodate Bold style.
            float bold_xAdvance_multiplier = 1; // Used to increase spacing between character when style is bold.

            m_baselineOffset = 0; // Used by subscript characters.
            m_baselineOffsetStack.Clear();

            // Underline
            bool beginUnderline = false;
            Vector3 underline_start = Vector3.zero; // Used to track where underline starts & ends.
            Vector3 underline_end = Vector3.zero;

            // Strike-through
            bool beginStrikethrough = false;
            Vector3 strikethrough_start = Vector3.zero;
            Vector3 strikethrough_end = Vector3.zero;

            // Text Highlight
            bool beginHighlight = false;
            Vector3 highlight_start = Vector3.zero;
            Vector3 highlight_end = Vector3.zero;

            m_fontColor32 = m_fontColor;
            Color32 vertexColor;
            m_htmlColor = m_fontColor32;
            m_underlineColor = m_htmlColor;
            m_strikethroughColor = m_htmlColor;

            m_colorStack.SetDefault(m_htmlColor);
            m_underlineColorStack.SetDefault(m_htmlColor);
            m_strikethroughColorStack.SetDefault(m_htmlColor);
            m_highlightColorStack.SetDefault(m_htmlColor);

            m_colorGradientPreset = null;
            m_colorGradientStack.SetDefault(null);

            // Clear the Style stack.
            //m_styleStack.Clear();

            // Clear the Action stack.
            m_actionStack.Clear();

            m_isFXMatrixSet = false;

            m_lineOffset = 0; // Amount of space between lines (font line spacing + m_linespacing).
            m_lineHeight = TMP_Math.FLOAT_UNSET;
            float lineGap = m_currentFontAsset.faceInfo.lineHeight - (m_currentFontAsset.faceInfo.ascentLine - m_currentFontAsset.faceInfo.descentLine);

            m_cSpacing = 0; // Amount of space added between characters as a result of the use of the <cspace> tag.
            m_monoSpacing = 0;
            float lineOffsetDelta = 0;
            m_xAdvance = 0; // Used to track the position of each character.

            tag_LineIndent = 0; // Used for indentation of text.
            tag_Indent = 0;
            m_indentStack.SetDefault(0);
            tag_NoParsing = false;
            //m_isIgnoringAlignment = false;

            m_characterCount = 0; // Total characters in the char[]

            // Tracking of line information
            m_firstCharacterOfLine = 0;
            m_lastCharacterOfLine = 0;
            m_firstVisibleCharacterOfLine = 0;
            m_lastVisibleCharacterOfLine = 0;
            m_maxLineAscender = k_LargeNegativeFloat;
            m_maxLineDescender = k_LargePositiveFloat;
            m_lineNumber = 0;
            m_lineVisibleCharacterCount = 0;
            bool isStartOfNewLine = true;
            m_firstOverflowCharacterIndex = -1;

            m_pageNumber = 0;
            int pageToDisplay = Mathf.Clamp(m_pageToDisplay - 1, 0, m_textInfo.pageInfo.Length - 1);
            int previousPageOverflowChar = 0;

            int ellipsisIndex = 0;

            Vector4 margins = m_margin;
            float marginWidth = m_marginWidth;
            float marginHeight = m_marginHeight;
            m_marginLeft = 0;
            m_marginRight = 0;
            m_width = -1;
            float width = marginWidth + 0.0001f - m_marginLeft - m_marginRight;

            // Need to initialize these Extents structures
            m_meshExtents.min = k_LargePositiveVector2;
            m_meshExtents.max = k_LargeNegativeVector2;

            // Initialize lineInfo
            m_textInfo.ClearLineInfo();

            // Tracking of the highest Ascender
            m_maxCapHeight = 0;
            m_maxAscender = 0;
            m_maxDescender = 0;
            float pageAscender = 0;
            float maxVisibleDescender = 0;
            bool isMaxVisibleDescenderSet = false;
            m_isNewPage = false;

            // Initialize struct to track states of word wrapping
            bool isFirstWord = true;
            m_isNonBreakingSpace = false;
            bool ignoreNonBreakingSpace = false;
            bool isLastBreakingChar = false;
            float linebreakingWidth = 0;
            int wrappingIndex = 0;

            // Save character and line state before we begin layout.
            SaveWordWrappingState(ref m_SavedWordWrapState, -1, -1);
            SaveWordWrappingState(ref m_SavedLineState, -1, -1);

            loopCountA += 1;

            // Parse through Character buffer to read HTML tags and begin creating mesh.
            for (int i = 0; i < m_TextParsingBuffer.Length && m_TextParsingBuffer[i].unicode != 0; i++)
            {
                charCode = m_TextParsingBuffer[i].unicode;

                // Parse Rich Text Tag
                #region Parse Rich Text Tag
                if (m_isRichText && charCode == 60)  // '<'
                {
                    m_isParsingText = true;
                    m_textElementType = TMP_TextElementType.Character;
                    int endTagIndex;

                    // Check if Tag is valid. If valid, skip to the end of the validated tag.
                    if (ValidateHtmlTag(m_TextParsingBuffer, i + 1, out endTagIndex))
                    {
                        i = endTagIndex;

                        // Continue to next character or handle the sprite element
                        if (m_textElementType == TMP_TextElementType.Character)
                            continue;
                    }
                }
                else
                {
                    m_textElementType = m_textInfo.characterInfo[m_characterCount].elementType;
                    m_currentMaterialIndex = m_textInfo.characterInfo[m_characterCount].materialReferenceIndex;
                    m_currentFontAsset = m_textInfo.characterInfo[m_characterCount].fontAsset;
                }
                #endregion End Parse Rich Text Tag

                int prev_MaterialIndex = m_currentMaterialIndex;
                bool isUsingAltTypeface = m_textInfo.characterInfo[m_characterCount].isUsingAlternateTypeface;

                m_isParsingText = false;

                // When using Linked text, mark character as ignored and skip to next character.
                if (m_characterCount < m_firstVisibleCharacter)
                {
                    m_textInfo.characterInfo[m_characterCount].isVisible = false;
                    m_textInfo.characterInfo[m_characterCount].character = (char)0x200B;
                    m_characterCount += 1;
                    continue;
                }

                // Handle Font Styles like LowerCase, UpperCase and SmallCaps.
                #region Handling of LowerCase, UpperCase and SmallCaps Font Styles

                float smallCapsMultiplier = 1.0f;

                if (m_textElementType == TMP_TextElementType.Character)
                {
                    if ((m_FontStyleInternal & FontStyles.UpperCase) == FontStyles.UpperCase)
                    {
                        // If this character is lowercase, switch to uppercase.
                        if (char.IsLower((char)charCode))
                            charCode = char.ToUpper((char)charCode);

                    }
                    else if ((m_FontStyleInternal & FontStyles.LowerCase) == FontStyles.LowerCase)
                    {
                        // If this character is uppercase, switch to lowercase.
                        if (char.IsUpper((char)charCode))
                            charCode = char.ToLower((char)charCode);
                    }
                    else if ((m_FontStyleInternal & FontStyles.SmallCaps) == FontStyles.SmallCaps)
                    {
                        if (char.IsLower((char)charCode))
                        {
                            smallCapsMultiplier = 0.8f;
                            charCode = char.ToUpper((char)charCode);
                        }
                    }
                }
                #endregion


                // Look up Character Data from Dictionary and cache it.
                #region Look up Character Data
                if (m_textElementType == TMP_TextElementType.Sprite)
                {
                    // If a sprite is used as a fallback then get a reference to it and set the color to white.
                    m_currentSpriteAsset = m_textInfo.characterInfo[m_characterCount].spriteAsset;
                    m_spriteIndex = m_textInfo.characterInfo[m_characterCount].spriteIndex;

                    TMP_SpriteCharacter sprite = m_currentSpriteAsset.spriteCharacterTable[m_spriteIndex];
                    if (sprite == null) continue;

                    // Sprites are assigned in the E000 Private Area + sprite Index
                    if (charCode == 60)
                        charCode = 57344 + m_spriteIndex;
                    else
                        m_spriteColor = s_colorWhite;

                    // The sprite scale calculations are based on the font asset assigned to the text object.
                    float spriteScale = (m_currentFontSize / m_currentFontAsset.faceInfo.pointSize * m_currentFontAsset.faceInfo.scale * (m_isOrthographic ? 1 : 0.1f));
                    currentElementScale = m_currentFontAsset.faceInfo.ascentLine / sprite.glyph.metrics.height * sprite.scale * sprite.glyph.scale * spriteScale;

                    m_cached_TextElement = sprite;

                    m_textInfo.characterInfo[m_characterCount].elementType = TMP_TextElementType.Sprite;
                    m_textInfo.characterInfo[m_characterCount].scale = spriteScale;
                    m_textInfo.characterInfo[m_characterCount].spriteAsset = m_currentSpriteAsset;
                    m_textInfo.characterInfo[m_characterCount].fontAsset = m_currentFontAsset;
                    m_textInfo.characterInfo[m_characterCount].materialReferenceIndex = m_currentMaterialIndex;

                    m_currentMaterialIndex = prev_MaterialIndex;

                    padding = 0;
                }
                else if (m_textElementType == TMP_TextElementType.Character)
                {
                    m_cached_TextElement = m_textInfo.characterInfo[m_characterCount].textElement;
                    if (m_cached_TextElement == null) continue;

                    m_currentFontAsset = m_textInfo.characterInfo[m_characterCount].fontAsset;
                    m_currentMaterial = m_textInfo.characterInfo[m_characterCount].material;
                    m_currentMaterialIndex = m_textInfo.characterInfo[m_characterCount].materialReferenceIndex;

                    // Re-calculate font scale as the font asset may have changed.
                    m_fontScale = m_currentFontSize * smallCapsMultiplier / m_currentFontAsset.faceInfo.pointSize * m_currentFontAsset.faceInfo.scale * (m_isOrthographic ? 1 : 0.1f);

                    currentElementScale = m_fontScale * m_fontScaleMultiplier * m_cached_TextElement.scale * m_cached_TextElement.glyph.scale;

                    m_textInfo.characterInfo[m_characterCount].elementType = TMP_TextElementType.Character;
                    m_textInfo.characterInfo[m_characterCount].scale = currentElementScale;

                    padding = m_currentMaterialIndex == 0 ? m_padding : m_subTextObjects[m_currentMaterialIndex].padding;
                }
                #endregion


                // Handle Soft Hyphen
                #region Handle Soft Hyphen
                float old_scale = currentElementScale;
                if (charCode == 0xAD)
                {
                    currentElementScale = 0;
                }
                #endregion


                // Store some of the text object's information
                m_textInfo.characterInfo[m_characterCount].character = (char)charCode;
                m_textInfo.characterInfo[m_characterCount].pointSize = m_currentFontSize;
                m_textInfo.characterInfo[m_characterCount].color = m_htmlColor;
                m_textInfo.characterInfo[m_characterCount].underlineColor = m_underlineColor;
                m_textInfo.characterInfo[m_characterCount].strikethroughColor = m_strikethroughColor;
                m_textInfo.characterInfo[m_characterCount].highlightColor = m_highlightColor;
                m_textInfo.characterInfo[m_characterCount].style = m_FontStyleInternal;
                //m_textInfo.characterInfo[m_characterCount].index = m_TextParsingBuffer[i].stringIndex;
                //m_textInfo.characterInfo[m_characterCount].isIgnoringAlignment = m_isIgnoringAlignment;


                // Handle Kerning if Enabled.
                #region Handle Kerning
                TMP_GlyphValueRecord glyphAdjustments = new TMP_GlyphValueRecord();
                float characterSpacingAdjustment = m_characterSpacing;
                if (m_enableKerning)
                {
                    TMP_GlyphPairAdjustmentRecord adjustmentPair;

                    if (m_characterCount < totalCharacterCount - 1)
                    {
                        uint firstGlyphIndex = m_cached_TextElement.glyphIndex;
                        uint secondGlyphIndex = m_textInfo.characterInfo[m_characterCount + 1].textElement.glyphIndex;
                        long key = new GlyphPairKey(firstGlyphIndex, secondGlyphIndex).key;

                        if (m_currentFontAsset.fontFeatureTable.m_GlyphPairAdjustmentRecordLookupDictionary.TryGetValue(key, out adjustmentPair))
                        {
                            glyphAdjustments = adjustmentPair.firstAdjustmentRecord.glyphValueRecord;
                            characterSpacingAdjustment = (adjustmentPair.featureLookupFlags & FontFeatureLookupFlags.IgnoreSpacingAdjustments) == FontFeatureLookupFlags.IgnoreSpacingAdjustments ? 0 : characterSpacingAdjustment;
                        }
                    }

                    if (m_characterCount >= 1)
                    {
                        uint firstGlyphIndex = m_textInfo.characterInfo[m_characterCount - 1].textElement.glyphIndex;
                        uint secondGlyphIndex = m_cached_TextElement.glyphIndex;
                        long key = new GlyphPairKey(firstGlyphIndex, secondGlyphIndex).key;

                        if (m_currentFontAsset.fontFeatureTable.m_GlyphPairAdjustmentRecordLookupDictionary.TryGetValue(key, out adjustmentPair))
                        {
                            glyphAdjustments += adjustmentPair.secondAdjustmentRecord.glyphValueRecord;
                            characterSpacingAdjustment = (adjustmentPair.featureLookupFlags & FontFeatureLookupFlags.IgnoreSpacingAdjustments) == FontFeatureLookupFlags.IgnoreSpacingAdjustments ? 0 : characterSpacingAdjustment;
                        }
                    }
                }
                #endregion


                // Initial Implementation for RTL support.
                #region Handle Right-to-Left
                if (m_isRightToLeft)
                {
                    m_xAdvance -= ((m_cached_TextElement.glyph.metrics.horizontalAdvance * bold_xAdvance_multiplier + characterSpacingAdjustment + m_wordSpacing + m_currentFontAsset.normalSpacingOffset) * currentElementScale + m_cSpacing) * (1 - m_charWidthAdjDelta);

                    if (char.IsWhiteSpace((char)charCode) || charCode == 0x200B)
                        m_xAdvance -= m_wordSpacing * currentElementScale;
                }
                #endregion


                // Handle Mono Spacing
                #region Handle Mono Spacing
                float monoAdvance = 0;
                if (m_monoSpacing != 0)
                {
                    monoAdvance = (m_monoSpacing / 2 - (m_cached_TextElement.glyph.metrics.width / 2 + m_cached_TextElement.glyph.metrics.horizontalBearingX) * currentElementScale) * (1 - m_charWidthAdjDelta);
                    m_xAdvance += monoAdvance;
                }
                #endregion


                // Set Padding based on selected font style
                #region Handle Style Padding
                if (m_textElementType == TMP_TextElementType.Character && !isUsingAltTypeface && (/*(m_fontStyle & FontStyles.Bold) == FontStyles.Bold ||*/ (m_FontStyleInternal & FontStyles.Bold) == FontStyles.Bold)) // Checks for any combination of Bold Style.
                {
                    if (m_currentMaterial.HasProperty(ShaderUtilities.ID_GradientScale))
                    {
                        float gradientScale = m_currentMaterial.GetFloat(ShaderUtilities.ID_GradientScale);
                        style_padding = m_currentFontAsset.boldStyle / 4.0f * gradientScale * m_currentMaterial.GetFloat(ShaderUtilities.ID_ScaleRatio_A);

                        // Clamp overall padding to Gradient Scale size.
                        if (style_padding + padding > gradientScale)
                            padding = gradientScale - style_padding;
                    }
                    else
                        style_padding = 0;

                    bold_xAdvance_multiplier = 1 + m_currentFontAsset.boldSpacing * 0.01f;
                }
                else
                {
                    if (m_currentMaterial.HasProperty(ShaderUtilities.ID_GradientScale))
                    {
                        float gradientScale = m_currentMaterial.GetFloat(ShaderUtilities.ID_GradientScale);
                        style_padding = m_currentFontAsset.normalStyle / 4.0f * gradientScale * m_currentMaterial.GetFloat(ShaderUtilities.ID_ScaleRatio_A);

                        // Clamp overall padding to Gradient Scale size.
                        if (style_padding + padding > gradientScale)
                            padding = gradientScale - style_padding;
                    }
                    else
                        style_padding = 0;

                    bold_xAdvance_multiplier = 1.0f;
                }
                #endregion Handle Style Padding


                // Determine the position of the vertices of the Character or Sprite.
                #region Calculate Vertices Position
                float fontBaseLineOffset = m_currentFontAsset.faceInfo.baseline * m_fontScale * m_fontScaleMultiplier * m_currentFontAsset.faceInfo.scale;
                Vector3 top_left;
                top_left.x = m_xAdvance + ((m_cached_TextElement.glyph.metrics.horizontalBearingX - padding - style_padding + glyphAdjustments.xPlacement) * currentElementScale * (1 - m_charWidthAdjDelta));
                top_left.y = fontBaseLineOffset + (m_cached_TextElement.glyph.metrics.horizontalBearingY + padding + glyphAdjustments.yPlacement) * currentElementScale - m_lineOffset + m_baselineOffset;
                top_left.z = 0;

                Vector3 bottom_left;
                bottom_left.x = top_left.x;
                bottom_left.y = top_left.y - ((m_cached_TextElement.glyph.metrics.height + padding * 2) * currentElementScale);
                bottom_left.z = 0;

                Vector3 top_right;
                top_right.x = bottom_left.x + ((m_cached_TextElement.glyph.metrics.width + padding * 2 + style_padding * 2) * currentElementScale * (1 - m_charWidthAdjDelta));
                top_right.y = top_left.y;
                top_right.z = 0;

                Vector3 bottom_right;
                bottom_right.x = top_right.x;
                bottom_right.y = bottom_left.y;
                bottom_right.z = 0;
                #endregion


                // Check if we need to Shear the rectangles for Italic styles
                #region Handle Italic & Shearing
                if (m_textElementType == TMP_TextElementType.Character && !isUsingAltTypeface && (/*(m_fontStyle & FontStyles.Italic) == FontStyles.Italic ||*/ (m_FontStyleInternal & FontStyles.Italic) == FontStyles.Italic))
                {
                    // Shift Top vertices forward by half (Shear Value * height of character) and Bottom vertices back by same amount. 
                    float shear_value = m_currentFontAsset.italicStyle * 0.01f;
                    Vector3 topShear = new Vector3(shear_value * ((m_cached_TextElement.glyph.metrics.horizontalBearingY + padding + style_padding) * currentElementScale), 0, 0);
                    Vector3 bottomShear = new Vector3(shear_value * (((m_cached_TextElement.glyph.metrics.horizontalBearingY - m_cached_TextElement.glyph.metrics.height - padding - style_padding)) * currentElementScale), 0, 0);

                    top_left = top_left + topShear;
                    bottom_left = bottom_left + bottomShear;
                    top_right = top_right + topShear;
                    bottom_right = bottom_right + bottomShear;
                }
                #endregion Handle Italics & Shearing


                // Handle Character Rotation
                #region Handle Character Rotation
                if (m_isFXMatrixSet)
                {
                    // Apply scale matrix when simulating Condensed text.
                    if (m_FXMatrix.lossyScale.x != 1)
                    {
                        //top_left = m_FXMatrix.MultiplyPoint3x4(top_left);
                        //bottom_left = m_FXMatrix.MultiplyPoint3x4(bottom_left);
                        //top_right = m_FXMatrix.MultiplyPoint3x4(top_right);
                        //bottom_right = m_FXMatrix.MultiplyPoint3x4(bottom_right);
                    }

                    Vector3 positionOffset = (top_right + bottom_left) / 2;

                    top_left = m_FXMatrix.MultiplyPoint3x4(top_left - positionOffset) + positionOffset;
                    bottom_left = m_FXMatrix.MultiplyPoint3x4(bottom_left - positionOffset) + positionOffset;
                    top_right = m_FXMatrix.MultiplyPoint3x4(top_right - positionOffset) + positionOffset;
                    bottom_right = m_FXMatrix.MultiplyPoint3x4(bottom_right - positionOffset) + positionOffset;
                }
                #endregion


                // Store vertex information for the character or sprite.
                m_textInfo.characterInfo[m_characterCount].bottomLeft = bottom_left;
                m_textInfo.characterInfo[m_characterCount].topLeft = top_left;
                m_textInfo.characterInfo[m_characterCount].topRight = top_right;
                m_textInfo.characterInfo[m_characterCount].bottomRight = bottom_right;

                m_textInfo.characterInfo[m_characterCount].origin = m_xAdvance;
                m_textInfo.characterInfo[m_characterCount].baseLine = fontBaseLineOffset - m_lineOffset + m_baselineOffset;
                m_textInfo.characterInfo[m_characterCount].aspectRatio = (top_right.x - bottom_left.x) / (top_left.y - bottom_left.y);


                // Compute and save text element Ascender and maximum line Ascender.
                float elementAscender = m_currentFontAsset.faceInfo.ascentLine * (m_textElementType == TMP_TextElementType.Character ? currentElementScale / smallCapsMultiplier : m_textInfo.characterInfo[m_characterCount].scale) + m_baselineOffset;
                m_textInfo.characterInfo[m_characterCount].ascender = elementAscender - m_lineOffset;
                m_maxLineAscender = elementAscender > m_maxLineAscender ? elementAscender : m_maxLineAscender;

                // Compute and save text element Descender and maximum line Descender.
                float elementDescender = m_currentFontAsset.faceInfo.descentLine * (m_textElementType == TMP_TextElementType.Character ? currentElementScale / smallCapsMultiplier : m_textInfo.characterInfo[m_characterCount].scale) + m_baselineOffset;
                float elementDescenderII = m_textInfo.characterInfo[m_characterCount].descender = elementDescender - m_lineOffset;
                m_maxLineDescender = elementDescender < m_maxLineDescender ? elementDescender : m_maxLineDescender;

                // Adjust maxLineAscender and maxLineDescender if style is superscript or subscript
                if ((m_FontStyleInternal & FontStyles.Subscript) == FontStyles.Subscript || (m_FontStyleInternal & FontStyles.Superscript) == FontStyles.Superscript)
                {
                    float baseAscender = (elementAscender - m_baselineOffset) / m_currentFontAsset.faceInfo.subscriptSize;
                    elementAscender = m_maxLineAscender;
                    m_maxLineAscender = baseAscender > m_maxLineAscender ? baseAscender : m_maxLineAscender;

                    float baseDescender = (elementDescender - m_baselineOffset) / m_currentFontAsset.faceInfo.subscriptSize;
                    elementDescender = m_maxLineDescender;
                    m_maxLineDescender = baseDescender < m_maxLineDescender ? baseDescender : m_maxLineDescender;
                }

                if (m_lineNumber == 0 || m_isNewPage)
                {
                    m_maxAscender = m_maxAscender > elementAscender ? m_maxAscender : elementAscender;
                    m_maxCapHeight = Mathf.Max(m_maxCapHeight, m_currentFontAsset.faceInfo.capLine * currentElementScale / smallCapsMultiplier);
                }
                if (m_lineOffset == 0) pageAscender = pageAscender > elementAscender ? pageAscender : elementAscender;


                // Set Characters to not visible by default.
                m_textInfo.characterInfo[m_characterCount].isVisible = false;

                // Setup Mesh for visible text elements. ie. not a SPACE / LINEFEED / CARRIAGE RETURN.
                #region Handle Visible Characters
                if (charCode == 9 || charCode == 0xA0 || charCode == 0x2007 || (!char.IsWhiteSpace((char)charCode) && charCode != 0x200B) || m_textElementType == TMP_TextElementType.Sprite)
                {
                    m_textInfo.characterInfo[m_characterCount].isVisible = true;

                    #region Experimental Margin Shaper
                    //Vector2 shapedMargins;
                    //if (marginShaper)
                    //{
                    //    shapedMargins = m_marginShaper.GetShapedMargins(m_textInfo.characterInfo[m_characterCount].baseLine);
                    //    if (shapedMargins.x < margins.x)
                    //    {
                    //        shapedMargins.x = m_marginLeft;
                    //    }
                    //    else
                    //    {
                    //        shapedMargins.x += m_marginLeft - margins.x;
                    //    }
                    //    if (shapedMargins.y < margins.z)
                    //    {
                    //        shapedMargins.y = m_marginRight;
                    //    }
                    //    else
                    //    {
                    //        shapedMargins.y += m_marginRight - margins.z;
                    //    }
                    //}
                    //else
                    //{
                    //    shapedMargins.x = m_marginLeft;
                    //    shapedMargins.y = m_marginRight;
                    //}
                    //width = marginWidth + 0.0001f - shapedMargins.x - shapedMargins.y;
                    //if (m_width != -1 && m_width < width)
                    //{
                    //    width = m_width;
                    //}
                    //m_textInfo.lineInfo[m_lineNumber].marginLeft = shapedMargins.x;
                    #endregion

                    width = m_width != -1 ? Mathf.Min(marginWidth + 0.0001f - m_marginLeft - m_marginRight, m_width) : marginWidth + 0.0001f - m_marginLeft - m_marginRight;
                    m_textInfo.lineInfo[m_lineNumber].marginLeft = m_marginLeft;

                    bool isJustifiedOrFlush = ((_HorizontalAlignmentOptions)m_lineJustification & _HorizontalAlignmentOptions.Flush) == _HorizontalAlignmentOptions.Flush || ((_HorizontalAlignmentOptions)m_lineJustification & _HorizontalAlignmentOptions.Justified) == _HorizontalAlignmentOptions.Justified;

                    // Calculate the line breaking width of the text.
                    linebreakingWidth = Mathf.Abs(m_xAdvance) + (!m_isRightToLeft ? m_cached_TextElement.glyph.metrics.horizontalAdvance : 0) * (1 - m_charWidthAdjDelta) * (charCode != 0xAD ? currentElementScale : old_scale);

                    // Check if Character exceeds the width of the Text Container
                    #region Handle Line Breaking, Text Auto-Sizing and Horizontal Overflow
                    if (linebreakingWidth > width * (isJustifiedOrFlush ? 1.05f : 1.0f))
                    {
                        ellipsisIndex = m_characterCount - 1; // Last safely rendered character

                        // Word Wrapping
                        #region Handle Word Wrapping
                        if (enableWordWrapping && m_characterCount != m_firstCharacterOfLine)
                        {
                            // Check if word wrapping is still possible
                            #region Line Breaking Check
                            if (wrappingIndex == m_SavedWordWrapState.previous_WordBreak || isFirstWord)
                            {
                                // Word wrapping is no longer possible. Shrink size of text if auto-sizing is enabled.
                                if (m_enableAutoSizing && m_fontSize > m_fontSizeMin)
                                {
                                    // Handle Character Width Adjustments
                                    #region Character Width Adjustments
                                    if (m_charWidthAdjDelta < m_charWidthMaxAdj / 100)
                                    {
                                        loopCountA = 0;
                                        m_charWidthAdjDelta += 0.01f;
                                        GenerateTextMesh();
                                        return;
                                    }
                                    #endregion

                                    // Adjust Point Size
                                    m_maxFontSize = m_fontSize;

                                    m_fontSize -= Mathf.Max((m_fontSize - m_minFontSize) / 2, 0.05f);
                                    m_fontSize = (int)(Mathf.Max(m_fontSize, m_fontSizeMin) * 20 + 0.5f) / 20f;

                                    if (loopCountA > 20) return; // Added to debug
                                    GenerateTextMesh();
                                    return;
                                }

                                // Word wrapping is no longer possible, now breaking up individual words.
                                if (m_isCharacterWrappingEnabled == false)
                                {
                                    if (ignoreNonBreakingSpace == false)
                                        ignoreNonBreakingSpace = true;
                                    else
                                        m_isCharacterWrappingEnabled = true;
                                }
                                else
                                    isLastBreakingChar = true;

                                //m_recursiveCount += 1;
                                //if (m_recursiveCount > 20)
                                //{
                                //    Debug.Log("Recursive count exceeded!");
                                //    continue;
                                //}
                            }
                            #endregion

                            // Restore to previously stored state of last valid (space character or linefeed)
                            i = RestoreWordWrappingState(ref m_SavedWordWrapState);
                            wrappingIndex = i;  // Used to detect when line length can no longer be reduced.

                            // Handling for Soft Hyphen
                            if (m_TextParsingBuffer[i].unicode == 0xAD) // && !m_isCharacterWrappingEnabled) // && ellipsisIndex != i && !m_isCharacterWrappingEnabled)
                            {
                                m_isTextTruncated = true;
                                m_TextParsingBuffer[i].unicode = 0x2D;
                                GenerateTextMesh();
                                return;
                            }

                            //Debug.Log("Last Visible Character of line # " + m_lineNumber + " is [" + m_textInfo.characterInfo[m_lastVisibleCharacterOfLine].character + " Character Count: " + m_characterCount + " Last visible: " + m_lastVisibleCharacterOfLine);

                            // Check if Line Spacing of previous line needs to be adjusted.
                            if (m_lineNumber > 0 && !TMP_Math.Approximately(m_maxLineAscender, m_startOfLineAscender) && m_lineHeight == TMP_Math.FLOAT_UNSET && !m_isNewPage)
                            {
                                //Debug.Log("(Line Break - Adjusting Line Spacing on line #" + m_lineNumber);
                                float offsetDelta = m_maxLineAscender - m_startOfLineAscender;
                                AdjustLineOffset(m_firstCharacterOfLine, m_characterCount, offsetDelta);
                                m_lineOffset += offsetDelta;
                                m_SavedWordWrapState.lineOffset = m_lineOffset;
                                m_SavedWordWrapState.previousLineAscender = m_maxLineAscender;

                                // TODO - Add check for character exceeding vertical bounds
                            }
                            m_isNewPage = false;

                            // Calculate lineAscender & make sure if last character is superscript or subscript that we check that as well.
                            float lineAscender = m_maxLineAscender - m_lineOffset;
                            float lineDescender = m_maxLineDescender - m_lineOffset;


                            // Update maxDescender and maxVisibleDescender
                            m_maxDescender = m_maxDescender < lineDescender ? m_maxDescender : lineDescender;
                            if (!isMaxVisibleDescenderSet)
                                maxVisibleDescender = m_maxDescender;

                            if (m_useMaxVisibleDescender && (m_characterCount >= m_maxVisibleCharacters || m_lineNumber >= m_maxVisibleLines))
                                isMaxVisibleDescenderSet = true;

                            // Track & Store lineInfo for the new line
                            m_textInfo.lineInfo[m_lineNumber].firstCharacterIndex = m_firstCharacterOfLine;
                            m_textInfo.lineInfo[m_lineNumber].firstVisibleCharacterIndex = m_firstVisibleCharacterOfLine = m_firstCharacterOfLine > m_firstVisibleCharacterOfLine ? m_firstCharacterOfLine : m_firstVisibleCharacterOfLine;
                            m_textInfo.lineInfo[m_lineNumber].lastCharacterIndex = m_lastCharacterOfLine = m_characterCount - 1 > 0 ? m_characterCount - 1 : 0;
                            m_textInfo.lineInfo[m_lineNumber].lastVisibleCharacterIndex = m_lastVisibleCharacterOfLine = m_lastVisibleCharacterOfLine < m_firstVisibleCharacterOfLine ? m_firstVisibleCharacterOfLine : m_lastVisibleCharacterOfLine;

                            m_textInfo.lineInfo[m_lineNumber].characterCount = m_textInfo.lineInfo[m_lineNumber].lastCharacterIndex - m_textInfo.lineInfo[m_lineNumber].firstCharacterIndex + 1;
                            m_textInfo.lineInfo[m_lineNumber].visibleCharacterCount = m_lineVisibleCharacterCount;
                            m_textInfo.lineInfo[m_lineNumber].lineExtents.min = new Vector2(m_textInfo.characterInfo[m_firstVisibleCharacterOfLine].bottomLeft.x, lineDescender);
                            m_textInfo.lineInfo[m_lineNumber].lineExtents.max = new Vector2(m_textInfo.characterInfo[m_lastVisibleCharacterOfLine].topRight.x, lineAscender);
                            m_textInfo.lineInfo[m_lineNumber].length = m_textInfo.lineInfo[m_lineNumber].lineExtents.max.x;
                            m_textInfo.lineInfo[m_lineNumber].width = width;

                            //m_textInfo.lineInfo[m_lineNumber].alignment = m_lineJustification;

                            m_textInfo.lineInfo[m_lineNumber].maxAdvance = m_textInfo.characterInfo[m_lastVisibleCharacterOfLine].xAdvance - (characterSpacingAdjustment + m_currentFontAsset.normalSpacingOffset) * currentElementScale - m_cSpacing;

                            m_textInfo.lineInfo[m_lineNumber].baseline = 0 - m_lineOffset;
                            m_textInfo.lineInfo[m_lineNumber].ascender = lineAscender;
                            m_textInfo.lineInfo[m_lineNumber].descender = lineDescender;
                            m_textInfo.lineInfo[m_lineNumber].lineHeight = lineAscender - lineDescender + lineGap * baseScale;

                            m_firstCharacterOfLine = m_characterCount; // Store first character of the next line.
                            m_lineVisibleCharacterCount = 0;

                            // Store the state of the line before starting on the new line.
                            SaveWordWrappingState(ref m_SavedLineState, i, m_characterCount - 1);

                            m_lineNumber += 1;
                            isStartOfNewLine = true;
                            isFirstWord = true;

                            // Check to make sure Array is large enough to hold a new line.
                            if (m_lineNumber >= m_textInfo.lineInfo.Length)
                                ResizeLineExtents(m_lineNumber);

                            // Apply Line Spacing based on scale of the last character of the line.
                            if (m_lineHeight == TMP_Math.FLOAT_UNSET)
                            {
                                float ascender = m_textInfo.characterInfo[m_characterCount].ascender - m_textInfo.characterInfo[m_characterCount].baseLine;
                                lineOffsetDelta = 0 - m_maxLineDescender + ascender + (lineGap + m_lineSpacing + m_lineSpacingDelta) * baseScale;
                                m_lineOffset += lineOffsetDelta;

                                m_startOfLineAscender = ascender;
                            }
                            else
                                m_lineOffset += m_lineHeight + m_lineSpacing * baseScale;

                            m_maxLineAscender = k_LargeNegativeFloat;
                            m_maxLineDescender = k_LargePositiveFloat;

                            m_xAdvance = 0 + tag_Indent;

                            continue;
                        }
                        #endregion End Word Wrapping


                        // Text Auto-Sizing (text exceeding Width of container. 
                        #region Handle Text Auto-Sizing
                        if (m_enableAutoSizing && m_fontSize > m_fontSizeMin)
                        {
                            // Handle Character Width Adjustments
                            #region Character Width Adjustments
                            if (m_charWidthAdjDelta < m_charWidthMaxAdj / 100)
                            {
                                loopCountA = 0;
                                m_charWidthAdjDelta += 0.01f;
                                GenerateTextMesh();
                                return;
                            }
                            #endregion

                            // Adjust Point Size
                            m_maxFontSize = m_fontSize;

                            m_fontSize -= Mathf.Max((m_fontSize - m_minFontSize) / 2, 0.05f);
                            m_fontSize = (int)(Mathf.Max(m_fontSize, m_fontSizeMin) * 20 + 0.5f) / 20f;

                            //m_recursiveCount = 0;
                            if (loopCountA > 20) return; // Added to debug 
                            GenerateTextMesh();
                            return;
                        }
                        #endregion End Text Auto-Sizing


                        // Handle Text Overflow
                        #region Handle Text Overflow
                        switch (m_overflowMode)
                        {
                            case TextOverflowModes.Overflow:
                                if (m_isMaskingEnabled)
                                    DisableMasking();

                                break;
                            case TextOverflowModes.Ellipsis:
                                if (m_isMaskingEnabled)
                                    DisableMasking();

                                m_isTextTruncated = true;

                                if (m_characterCount < 1)
                                {
                                    m_textInfo.characterInfo[m_characterCount].isVisible = false;
                                    //m_visibleCharacterCount = 0;
                                    break;
                                }

                                m_TextParsingBuffer[i - 1].unicode = 8230;
                                m_TextParsingBuffer[i].unicode = (char)0;

                                if (m_cached_Ellipsis_Character != null)
                                {
                                    m_textInfo.characterInfo[ellipsisIndex].character = (char)8230;
                                    m_textInfo.characterInfo[ellipsisIndex].textElement = m_cached_Ellipsis_Character;
                                    m_textInfo.characterInfo[ellipsisIndex].fontAsset = m_materialReferences[0].fontAsset;
                                    m_textInfo.characterInfo[ellipsisIndex].material = m_materialReferences[0].material;
                                    m_textInfo.characterInfo[ellipsisIndex].materialReferenceIndex = 0;
                                }
                                else
                                {
                                    Debug.LogWarning("Unable to use Ellipsis character since it wasn't found in the current Font Asset [" + m_fontAsset.name + "]. Consider regenerating this font asset to include the Ellipsis character (u+2026).\nNote: Warnings can be disabled in the TMP Settings file.", this);
                                }

                                m_totalCharacterCount = ellipsisIndex + 1;

                                GenerateTextMesh();
                                return;
                            //case TextOverflowModes.Masking:
                            //    if (!m_isMaskingEnabled)
                            //        EnableMasking();
                            //    break;
                            //case TextOverflowModes.ScrollRect:
                            //    if (!m_isMaskingEnabled)
                            //        EnableMasking();
                            //    break;
                            case TextOverflowModes.Truncate:
                                if (m_isMaskingEnabled)
                                    DisableMasking();

                                m_textInfo.characterInfo[m_characterCount].isVisible = false;
                                break;
                            case TextOverflowModes.Linked:
                                //m_textInfo.characterInfo[m_characterCount].isVisible = false;

                                //if (m_linkedTextComponent != null)
                                //{
                                //    m_linkedTextComponent.text = text;
                                //    m_linkedTextComponent.firstVisibleCharacter = m_characterCount;
                                //    m_linkedTextComponent.ForceMeshUpdate();
                                //}
                                break;
                        }
                        #endregion End Text Overflow

                    }
                    #endregion End Check for Characters Exceeding Width of Text Container


                    // Special handling of characters that are not ignored at the end of a line.
                    if (charCode == 9 || charCode == 0xA0 || charCode == 0x2007)
                    {
                        m_textInfo.characterInfo[m_characterCount].isVisible = false;
                        m_lastVisibleCharacterOfLine = m_characterCount;
                        m_textInfo.lineInfo[m_lineNumber].spaceCount += 1;
                        m_textInfo.spaceCount += 1;

                        if (charCode == 0xA0)
                            m_textInfo.lineInfo[m_lineNumber].controlCharacterCount += 1;
                    }
                    else
                    {
                        // Determine Vertex Color
                        if (m_overrideHtmlColors)
                            vertexColor = m_fontColor32;
                        else
                            vertexColor = m_htmlColor;

                        // Store Character & Sprite Vertex Information
                        if (m_textElementType == TMP_TextElementType.Character)
                        {
                            // Save Character Vertex Data
                            SaveGlyphVertexInfo(padding, style_padding, vertexColor);
                        }
                        else if (m_textElementType == TMP_TextElementType.Sprite)
                        {
                            SaveSpriteVertexInfo(vertexColor);
                        }
                    }


                    // Increase visible count for Characters.
                    if (m_textInfo.characterInfo[m_characterCount].isVisible && charCode != 0xAD)
                    {
                        if (isStartOfNewLine) { isStartOfNewLine = false; m_firstVisibleCharacterOfLine = m_characterCount; }

                        m_lineVisibleCharacterCount += 1;
                        m_lastVisibleCharacterOfLine = m_characterCount;
                    }
                }
                else
                {   // This is a Space, Tab, LineFeed or Carriage Return

                    // Track # of spaces per line which is used for line justification.
                    if ((charCode == 10 || char.IsSeparator((char)charCode)) && charCode != 0xAD && charCode != 0x200B && charCode != 0x2060)
                    {
                        m_textInfo.lineInfo[m_lineNumber].spaceCount += 1;
                        m_textInfo.spaceCount += 1;
                    }
                }
                #endregion Handle Visible Characters


                // Check if Line Spacing of previous line needs to be adjusted.
                #region Adjust Line Spacing
                if (m_lineNumber > 0 && !TMP_Math.Approximately(m_maxLineAscender, m_startOfLineAscender) && m_lineHeight == TMP_Math.FLOAT_UNSET && !m_isNewPage)
                {
                    //Debug.Log("Inline - Adjusting Line Spacing on line #" + m_lineNumber);
                    //float gap = 0; // Compute gap.

                    float offsetDelta = m_maxLineAscender - m_startOfLineAscender;
                    AdjustLineOffset(m_firstCharacterOfLine, m_characterCount, offsetDelta);
                    elementDescenderII -= offsetDelta;
                    m_lineOffset += offsetDelta;

                    m_startOfLineAscender += offsetDelta;
                    m_SavedWordWrapState.lineOffset = m_lineOffset;
                    m_SavedWordWrapState.previousLineAscender = m_startOfLineAscender;
                }
                #endregion


                // Store Rectangle positions for each Character.
                #region Store Character Data
                m_textInfo.characterInfo[m_characterCount].lineNumber = m_lineNumber;
                m_textInfo.characterInfo[m_characterCount].pageNumber = m_pageNumber;

                if (charCode != 10 && charCode != 13 && charCode != 8230 || m_textInfo.lineInfo[m_lineNumber].characterCount == 1)
                    m_textInfo.lineInfo[m_lineNumber].alignment = m_lineJustification;
                #endregion Store Character Data


                // Check if text Exceeds the vertical bounds of the margin area.
                #region Check Vertical Bounds & Auto-Sizing
                if (m_maxAscender - elementDescenderII > marginHeight + 0.0001f)
                {
                    // Handle Line spacing adjustments
                    #region Line Spacing Adjustments
                    if (m_enableAutoSizing && m_lineSpacingDelta > m_lineSpacingMax && m_lineNumber > 0)
                    {
                        loopCountA = 0;

                        m_lineSpacingDelta -= 1;
                        GenerateTextMesh();
                        return;
                    }
                    #endregion


                    // Handle Text Auto-sizing resulting from text exceeding vertical bounds.
                    #region Text Auto-Sizing (Text greater than vertical bounds)
                    if (m_enableAutoSizing && m_fontSize > m_fontSizeMin)
                    {
                        m_maxFontSize = m_fontSize;

                        m_fontSize -= Mathf.Max((m_fontSize - m_minFontSize) / 2, 0.05f);
                        m_fontSize = (int)(Mathf.Max(m_fontSize, m_fontSizeMin) * 20 + 0.5f) / 20f;

                        //m_recursiveCount = 0;
                        if (loopCountA > 20) return; // Added to debug 
                        GenerateTextMesh();
                        return;
                    }
                    #endregion Text Auto-Sizing

                    // Set isTextOverflowing and firstOverflowCharacterIndex
                    if (m_firstOverflowCharacterIndex == -1)
                        m_firstOverflowCharacterIndex = m_characterCount;

                    // Handle Text Overflow
                    #region Text Overflow
                    switch (m_overflowMode)
                    {
                        case TextOverflowModes.Overflow:
                            if (m_isMaskingEnabled)
                                DisableMasking();

                            break;
                        case TextOverflowModes.Ellipsis:
                            if (m_isMaskingEnabled)
                                DisableMasking();

                            if (m_lineNumber > 0)
                            {
                                m_TextParsingBuffer[m_textInfo.characterInfo[ellipsisIndex].index].unicode = 8230;
                                m_TextParsingBuffer[m_textInfo.characterInfo[ellipsisIndex].index + 1].unicode = (char)0;

                                if (m_cached_Ellipsis_Character != null)
                                {
                                    m_textInfo.characterInfo[ellipsisIndex].character = (char)8230;
                                    m_textInfo.characterInfo[ellipsisIndex].textElement = m_cached_Ellipsis_Character;
                                    m_textInfo.characterInfo[ellipsisIndex].fontAsset = m_materialReferences[0].fontAsset;
                                    m_textInfo.characterInfo[ellipsisIndex].material = m_materialReferences[0].material;
                                    m_textInfo.characterInfo[ellipsisIndex].materialReferenceIndex = 0;
                                }
                                else
                                {
                                    Debug.LogWarning("Unable to use Ellipsis character since it wasn't found in the current Font Asset [" + m_fontAsset.name + "]. Consider regenerating this font asset to include the Ellipsis character (u+2026).\nNote: Warnings can be disabled in the TMP Settings file.", this);
                                }

                                m_totalCharacterCount = ellipsisIndex + 1;

                                GenerateTextMesh();
                                m_isTextTruncated = true;
                                return;
                            }
                            else
                            {
                                ClearMesh(false);
                                return;
                            }
                        //case TextOverflowModes.Masking:
                        //    if (!m_isMaskingEnabled)
                        //        EnableMasking();
                        //    break;
                        //case TextOverflowModes.ScrollRect:
                        //    if (!m_isMaskingEnabled)
                        //        EnableMasking();
                        //    break;
                        case TextOverflowModes.Truncate:
                            if (m_isMaskingEnabled)
                                DisableMasking();

                            // TODO : Optimize 
                            if (m_lineNumber > 0)
                            {
                                m_TextParsingBuffer[m_textInfo.characterInfo[ellipsisIndex].index + 1].unicode = (char)0;

                                m_totalCharacterCount = ellipsisIndex + 1;

                                GenerateTextMesh();
                                m_isTextTruncated = true;
                                return;
                            }
                            else
                            {
                                ClearMesh(false);
                                return;
                            }
                        case TextOverflowModes.Page:
                            if (m_isMaskingEnabled)
                                DisableMasking();

                            // Ignore Page Break, Linefeed or carriage return
                            if (charCode == 13 || charCode == 10)
                                break;

                            // Return if the first character doesn't fit.
                            if (i == 0)
                            {
                                ClearMesh();
                                return;
                            }
                            else if (previousPageOverflowChar == i)
                            {
                                m_TextParsingBuffer[i].unicode = 0;
                                m_isTextTruncated = true;
                            }

                            previousPageOverflowChar = i;

                            // Go back to previous line and re-layout 
                            i = RestoreWordWrappingState(ref m_SavedLineState);

                            m_isNewPage = true;
                            m_xAdvance = 0 + tag_Indent;
                            m_lineOffset = 0;
                            m_maxAscender = 0;
                            pageAscender = 0;
                            m_lineNumber += 1;
                            m_pageNumber += 1;
                            continue;
                        case TextOverflowModes.Linked:
                            if (m_linkedTextComponent != null)
                            {
                                m_linkedTextComponent.text = text;
                                m_linkedTextComponent.firstVisibleCharacter = m_characterCount;
                                m_linkedTextComponent.ForceMeshUpdate();
                            }

                            // Truncate remaining text
                            if (m_lineNumber > 0)
                            {
                                m_TextParsingBuffer[i].unicode = (char)0;

                                m_totalCharacterCount = m_characterCount;

                                // TODO : Optimize as we should be able to end the layout phase here without having to do another pass.
                                GenerateTextMesh();
                                m_isTextTruncated = true;
                                return;
                            }
                            else
                            {
                                ClearMesh(true);
                                return;
                            }
                    }
                    #endregion End Text Overflow

                }
                #endregion Check Vertical Bounds


                // Handle xAdvance & Tabulation Stops. Tab stops at every 25% of Font Size.
                #region XAdvance, Tabulation & Stops
                if (charCode == 9)
                {
                    float tabSize = m_currentFontAsset.faceInfo.tabWidth * m_currentFontAsset.tabSize * currentElementScale;
                    float tabs = Mathf.Ceil(m_xAdvance / tabSize) * tabSize;
                    m_xAdvance = tabs > m_xAdvance ? tabs : m_xAdvance + tabSize;
                }
                else if (m_monoSpacing != 0)
                {
                    m_xAdvance += (m_monoSpacing - monoAdvance + ((characterSpacingAdjustment + m_currentFontAsset.normalSpacingOffset) * currentElementScale) + m_cSpacing) * (1 - m_charWidthAdjDelta);

                    if (char.IsWhiteSpace((char)charCode) || charCode == 0x200B)
                        m_xAdvance += m_wordSpacing * currentElementScale;
                }
                else if (!m_isRightToLeft)
                {
                    float scaleFXMultiplier = 1;
                    if (m_isFXMatrixSet) scaleFXMultiplier = m_FXMatrix.lossyScale.x;

                    m_xAdvance += ((m_cached_TextElement.glyph.metrics.horizontalAdvance * scaleFXMultiplier * bold_xAdvance_multiplier + characterSpacingAdjustment + m_currentFontAsset.normalSpacingOffset + glyphAdjustments.xAdvance) * currentElementScale + m_cSpacing) * (1 - m_charWidthAdjDelta);

                    if (char.IsWhiteSpace((char)charCode) || charCode == 0x200B)
                        m_xAdvance += m_wordSpacing * currentElementScale;
                }
                else
                {
                    m_xAdvance -= glyphAdjustments.xAdvance * currentElementScale;
                }


                // Store xAdvance information
                m_textInfo.characterInfo[m_characterCount].xAdvance = m_xAdvance;

                #endregion Tabulation & Stops


                // Handle Carriage Return
                #region Carriage Return
                if (charCode == 13)
                {
                    m_xAdvance = 0 + tag_Indent;
                }
                #endregion Carriage Return


                // Handle Line Spacing Adjustments + Word Wrapping & special case for last line.
                #region Check for Line Feed and Last Character
                if (charCode == 10 || m_characterCount == totalCharacterCount - 1)
                {
                    // Check if Line Spacing of previous line needs to be adjusted.
                    if (m_lineNumber > 0 && !TMP_Math.Approximately(m_maxLineAscender, m_startOfLineAscender) && m_lineHeight == TMP_Math.FLOAT_UNSET && !m_isNewPage)
                    {
                        //Debug.Log("Line Feed - Adjusting Line Spacing on line #" + m_lineNumber);
                        float offsetDelta = m_maxLineAscender - m_startOfLineAscender;
                        AdjustLineOffset(m_firstCharacterOfLine, m_characterCount, offsetDelta);
                        elementDescenderII -= offsetDelta;
                        m_lineOffset += offsetDelta;
                    }
                    m_isNewPage = false;

                    // Calculate lineAscender & make sure if last character is superscript or subscript that we check that as well.
                    float lineAscender = m_maxLineAscender - m_lineOffset;
                    float lineDescender = m_maxLineDescender - m_lineOffset;

                    // Update maxDescender and maxVisibleDescender
                    m_maxDescender = m_maxDescender < lineDescender ? m_maxDescender : lineDescender;
                    if (!isMaxVisibleDescenderSet)
                        maxVisibleDescender = m_maxDescender;

                    if (m_useMaxVisibleDescender && (m_characterCount >= m_maxVisibleCharacters || m_lineNumber >= m_maxVisibleLines))
                        isMaxVisibleDescenderSet = true;

                    // Save Line Information
                    m_textInfo.lineInfo[m_lineNumber].firstCharacterIndex = m_firstCharacterOfLine;
                    m_textInfo.lineInfo[m_lineNumber].firstVisibleCharacterIndex = m_firstVisibleCharacterOfLine = m_firstCharacterOfLine > m_firstVisibleCharacterOfLine ? m_firstCharacterOfLine : m_firstVisibleCharacterOfLine;
                    m_textInfo.lineInfo[m_lineNumber].lastCharacterIndex = m_lastCharacterOfLine = m_characterCount;
                    m_textInfo.lineInfo[m_lineNumber].lastVisibleCharacterIndex = m_lastVisibleCharacterOfLine = m_lastVisibleCharacterOfLine < m_firstVisibleCharacterOfLine ? m_firstVisibleCharacterOfLine : m_lastVisibleCharacterOfLine;

                    m_textInfo.lineInfo[m_lineNumber].characterCount = m_textInfo.lineInfo[m_lineNumber].lastCharacterIndex - m_textInfo.lineInfo[m_lineNumber].firstCharacterIndex + 1;
                    m_textInfo.lineInfo[m_lineNumber].visibleCharacterCount = m_lineVisibleCharacterCount;
                    m_textInfo.lineInfo[m_lineNumber].lineExtents.min = new Vector2(m_textInfo.characterInfo[m_firstVisibleCharacterOfLine].bottomLeft.x, lineDescender);
                    m_textInfo.lineInfo[m_lineNumber].lineExtents.max = new Vector2(m_textInfo.characterInfo[m_lastVisibleCharacterOfLine].topRight.x, lineAscender);
                    m_textInfo.lineInfo[m_lineNumber].length = m_textInfo.lineInfo[m_lineNumber].lineExtents.max.x - (padding * currentElementScale);
                    m_textInfo.lineInfo[m_lineNumber].width = width;

                    if (m_textInfo.lineInfo[m_lineNumber].characterCount == 1)
                        m_textInfo.lineInfo[m_lineNumber].alignment = m_lineJustification;

                    if (m_textInfo.characterInfo[m_lastVisibleCharacterOfLine].isVisible)
                        m_textInfo.lineInfo[m_lineNumber].maxAdvance = m_textInfo.characterInfo[m_lastVisibleCharacterOfLine].xAdvance - (characterSpacingAdjustment + m_currentFontAsset.normalSpacingOffset) * currentElementScale - m_cSpacing;
                    else
                        m_textInfo.lineInfo[m_lineNumber].maxAdvance = m_textInfo.characterInfo[m_lastCharacterOfLine].xAdvance - (characterSpacingAdjustment + m_currentFontAsset.normalSpacingOffset) * currentElementScale - m_cSpacing;

                    m_textInfo.lineInfo[m_lineNumber].baseline = 0 - m_lineOffset;
                    m_textInfo.lineInfo[m_lineNumber].ascender = lineAscender;
                    m_textInfo.lineInfo[m_lineNumber].descender = lineDescender;
                    m_textInfo.lineInfo[m_lineNumber].lineHeight = lineAscender - lineDescender + lineGap * baseScale;

                    m_firstCharacterOfLine = m_characterCount + 1;
                    m_lineVisibleCharacterCount = 0;

                    // Add new line if not last line or character.
                    if (charCode == 10)
                    {
                        // Store the state of the line before starting on the new line.
                        SaveWordWrappingState(ref m_SavedLineState, i, m_characterCount);
                        // Store the state of the last Character before the new line.
                        SaveWordWrappingState(ref m_SavedWordWrapState, i, m_characterCount);

                        m_lineNumber += 1;
                        isStartOfNewLine = true;
                        ignoreNonBreakingSpace = false;
                        isFirstWord = true;

                        // Check to make sure Array is large enough to hold a new line.
                        if (m_lineNumber >= m_textInfo.lineInfo.Length)
                            ResizeLineExtents(m_lineNumber);

                        // Apply Line Spacing
                        if (m_lineHeight == TMP_Math.FLOAT_UNSET)
                        {
                            lineOffsetDelta = 0 - m_maxLineDescender + elementAscender + (lineGap + m_lineSpacing + m_paragraphSpacing + m_lineSpacingDelta) * baseScale;
                            m_lineOffset += lineOffsetDelta;
                        }
                        else
                            m_lineOffset += m_lineHeight + (m_lineSpacing + m_paragraphSpacing) * baseScale;

                        m_maxLineAscender = k_LargeNegativeFloat;
                        m_maxLineDescender = k_LargePositiveFloat;
                        m_startOfLineAscender = elementAscender;

                        m_xAdvance = 0 + tag_LineIndent + tag_Indent;

                        ellipsisIndex = m_characterCount - 1;

                        m_characterCount += 1;
                        continue;
                    }
                }
                #endregion Check for Linefeed or Last Character


                // Store Rectangle positions for each Character.
                #region Save CharacterInfo for the current character.
                // Determine the bounds of the Mesh.
                if (m_textInfo.characterInfo[m_characterCount].isVisible)
                {
                    m_meshExtents.min.x = Mathf.Min(m_meshExtents.min.x, m_textInfo.characterInfo[m_characterCount].bottomLeft.x);
                    m_meshExtents.min.y = Mathf.Min(m_meshExtents.min.y, m_textInfo.characterInfo[m_characterCount].bottomLeft.y);

                    m_meshExtents.max.x = Mathf.Max(m_meshExtents.max.x, m_textInfo.characterInfo[m_characterCount].topRight.x);
                    m_meshExtents.max.y = Mathf.Max(m_meshExtents.max.y, m_textInfo.characterInfo[m_characterCount].topRight.y);

                    //m_meshExtents.min = new Vector2(Mathf.Min(m_meshExtents.min.x, m_textInfo.characterInfo[m_characterCount].bottomLeft.x), Mathf.Min(m_meshExtents.min.y, m_textInfo.characterInfo[m_characterCount].bottomLeft.y));
                    //m_meshExtents.max = new Vector2(Mathf.Max(m_meshExtents.max.x, m_textInfo.characterInfo[m_characterCount].topRight.x), Mathf.Max(m_meshExtents.max.y, m_textInfo.characterInfo[m_characterCount].topRight.y));
                }


                // Save pageInfo Data
                if (m_overflowMode == TextOverflowModes.Page && charCode != 13 && charCode != 10) // && m_pageNumber < 16)
                {
                    // Check if we need to increase allocations for the pageInfo array.
                    if (m_pageNumber + 1 > m_textInfo.pageInfo.Length)
                        TMP_TextInfo.Resize(ref m_textInfo.pageInfo, m_pageNumber + 1, true);

                    m_textInfo.pageInfo[m_pageNumber].ascender = pageAscender;
                    m_textInfo.pageInfo[m_pageNumber].descender = elementDescender < m_textInfo.pageInfo[m_pageNumber].descender ? elementDescender : m_textInfo.pageInfo[m_pageNumber].descender;

                    if (m_pageNumber == 0 && m_characterCount == 0)
                        m_textInfo.pageInfo[m_pageNumber].firstCharacterIndex = m_characterCount;
                    else if (m_characterCount > 0 && m_pageNumber != m_textInfo.characterInfo[m_characterCount - 1].pageNumber)
                    {
                        m_textInfo.pageInfo[m_pageNumber - 1].lastCharacterIndex = m_characterCount - 1;
                        m_textInfo.pageInfo[m_pageNumber].firstCharacterIndex = m_characterCount;
                    }
                    else if (m_characterCount == totalCharacterCount - 1)
                        m_textInfo.pageInfo[m_pageNumber].lastCharacterIndex = m_characterCount;
                }
                #endregion Saving CharacterInfo


                // Save State of Mesh Creation for handling of Word Wrapping
                #region Save Word Wrapping State
                if (m_enableWordWrapping || m_overflowMode == TextOverflowModes.Truncate || m_overflowMode == TextOverflowModes.Ellipsis)
                {
                    if ((char.IsWhiteSpace((char)charCode) || charCode == 0x200B || charCode == 0x2D || charCode == 0xAD) && (!m_isNonBreakingSpace || ignoreNonBreakingSpace) && charCode != 0xA0 && charCode != 0x2007 && charCode != 0x2011 && charCode != 0x202F && charCode != 0x2060)
                    {
                        // We store the state of numerous variables for the most recent Space, LineFeed or Carriage Return to enable them to be restored 
                        // for Word Wrapping.
                        SaveWordWrappingState(ref m_SavedWordWrapState, i, m_characterCount);
                        m_isCharacterWrappingEnabled = false;
                        isFirstWord = false;
                    }
                    // Handling for East Asian languages
                    else if ((  charCode > 0x1100 && charCode < 0x11ff || /* Hangul Jamo */
                                charCode > 0x2E80 && charCode < 0x9FFF || /* CJK */
                                charCode > 0xA960 && charCode < 0xA97F || /* Hangul Jame Extended-A */
                                charCode > 0xAC00 && charCode < 0xD7FF || /* Hangul Syllables */
                                charCode > 0xF900 && charCode < 0xFAFF || /* CJK Compatibility Ideographs */
                                charCode > 0xFE30 && charCode < 0xFE4F || /* CJK Compatibility Forms */
                                charCode > 0xFF00 && charCode < 0xFFEF)   /* CJK Halfwidth */
                                && !m_isNonBreakingSpace)
                    {
                        if (isFirstWord || isLastBreakingChar || TMP_Settings.linebreakingRules.leadingCharacters.ContainsKey(charCode) == false &&
                            (m_characterCount < totalCharacterCount - 1 &&
                            TMP_Settings.linebreakingRules.followingCharacters.ContainsKey(m_textInfo.characterInfo[m_characterCount + 1].character) == false))
                        {
                            SaveWordWrappingState(ref m_SavedWordWrapState, i, m_characterCount);
                            m_isCharacterWrappingEnabled = false;
                            isFirstWord = false;
                        }
                    }
                    else if ((isFirstWord || m_isCharacterWrappingEnabled == true || isLastBreakingChar))
                        SaveWordWrappingState(ref m_SavedWordWrapState, i, m_characterCount);
                }
                #endregion Save Word Wrapping State

                m_characterCount += 1;
            }

            // Check Auto Sizing and increase font size to fill text container.
            #region Check Auto-Sizing (Upper Font Size Bounds)
            fontSizeDelta = m_maxFontSize - m_minFontSize;
            if (!m_isCharacterWrappingEnabled && m_enableAutoSizing && fontSizeDelta > 0.051f && m_fontSize < m_fontSizeMax)
            {
                m_minFontSize = m_fontSize;
                m_fontSize += Mathf.Max((m_maxFontSize - m_fontSize) / 2, 0.05f);
                m_fontSize = (int)(Mathf.Min(m_fontSize, m_fontSizeMax) * 20 + 0.5f) / 20f;

                //Debug.Log(m_fontSize);

                if (loopCountA > 20) return; // Added to debug
                GenerateTextMesh();
                return;
            }
            #endregion End Auto-sizing Check


            m_isCharacterWrappingEnabled = false;


            //Debug.Log("Iteration Count: " + loopCountA + ". Final Point Size: " + m_fontSize); // + "  B: " + loopCountB + "  C: " + loopCountC + "  D: " + loopCountD);



            // If there are no visible characters... no need to continue
            if (m_characterCount == 0) // && m_visibleSpriteCount == 0)
            {
                ClearMesh(true);

                // Event indicating the text has been regenerated.
                TMPro_EventManager.ON_TEXT_CHANGED(this);
                return;
            }


            // *** PHASE II of Text Generation ***
            int last_vert_index = m_materialReferences[0].referenceCount * (!m_isVolumetricText ? 4 : 8);

            // Partial clear of the vertices array to mark unused vertices as degenerate.
            m_textInfo.meshInfo[0].Clear(false);

            // Handle Text Alignment
            #region Text Vertical Alignment
            Vector3 anchorOffset = Vector3.zero;
            Vector3[] corners = m_RectTransformCorners; // GetTextContainerLocalCorners();

            // Handle Vertical Text Alignment
            switch (m_textAlignment)
            {
                // Top Vertically
                case TextAlignmentOptions.Top:
                case TextAlignmentOptions.TopLeft:
                case TextAlignmentOptions.TopRight:
                case TextAlignmentOptions.TopJustified:
                case TextAlignmentOptions.TopFlush:
                case TextAlignmentOptions.TopGeoAligned:
                    if (m_overflowMode != TextOverflowModes.Page)
                        anchorOffset = corners[1] + new Vector3(0 + margins.x, 0 - m_maxAscender - margins.y, 0);
                    else
                        anchorOffset = corners[1] + new Vector3(0 + margins.x, 0 - m_textInfo.pageInfo[pageToDisplay].ascender - margins.y, 0);
                    break;

                // Middle Vertically
                case TextAlignmentOptions.Left:
                case TextAlignmentOptions.Right:
                case TextAlignmentOptions.Center:
                case TextAlignmentOptions.Justified:
                case TextAlignmentOptions.Flush:
                case TextAlignmentOptions.CenterGeoAligned:
                    if (m_overflowMode != TextOverflowModes.Page)
                        anchorOffset = (corners[0] + corners[1]) / 2 + new Vector3(0 + margins.x, 0 - (m_maxAscender + margins.y + maxVisibleDescender - margins.w) / 2, 0);
                    else
                        anchorOffset = (corners[0] + corners[1]) / 2 + new Vector3(0 + margins.x, 0 - (m_textInfo.pageInfo[pageToDisplay].ascender + margins.y + m_textInfo.pageInfo[pageToDisplay].descender - margins.w) / 2, 0);
                    break;

                // Bottom Vertically
                case TextAlignmentOptions.Bottom:
                case TextAlignmentOptions.BottomLeft:
                case TextAlignmentOptions.BottomRight:
                case TextAlignmentOptions.BottomJustified:
                case TextAlignmentOptions.BottomFlush:
                case TextAlignmentOptions.BottomGeoAligned:
                    if (m_overflowMode != TextOverflowModes.Page)
                        anchorOffset = corners[0] + new Vector3(0 + margins.x, 0 - maxVisibleDescender + margins.w, 0);
                    else
                        anchorOffset = corners[0] + new Vector3(0 + margins.x, 0 - m_textInfo.pageInfo[pageToDisplay].descender + margins.w, 0);
                    break;

                // Baseline Vertically
                case TextAlignmentOptions.Baseline:
                case TextAlignmentOptions.BaselineLeft:
                case TextAlignmentOptions.BaselineRight:
                case TextAlignmentOptions.BaselineJustified:
                case TextAlignmentOptions.BaselineFlush:
                case TextAlignmentOptions.BaselineGeoAligned:
                    anchorOffset = (corners[0] + corners[1]) / 2 + new Vector3(0 + margins.x, 0, 0);
                    break;

                // Midline Vertically 
                case TextAlignmentOptions.MidlineLeft:
                case TextAlignmentOptions.Midline:
                case TextAlignmentOptions.MidlineRight:
                case TextAlignmentOptions.MidlineJustified:
                case TextAlignmentOptions.MidlineFlush:
                case TextAlignmentOptions.MidlineGeoAligned:
                    anchorOffset = (corners[0] + corners[1]) / 2 + new Vector3(0 + margins.x, 0 - (m_meshExtents.max.y + margins.y + m_meshExtents.min.y - margins.w) / 2, 0);
                    break;

                // Capline Vertically 
                case TextAlignmentOptions.CaplineLeft:
                case TextAlignmentOptions.Capline:
                case TextAlignmentOptions.CaplineRight:
                case TextAlignmentOptions.CaplineJustified:
                case TextAlignmentOptions.CaplineFlush:
                case TextAlignmentOptions.CaplineGeoAligned:
                    anchorOffset = (corners[0] + corners[1]) / 2 + new Vector3(0 + margins.x, 0 - (m_maxCapHeight - margins.y - margins.w) / 2, 0);
                    break;
            }
            #endregion


            // Initialization for Second Pass
            Vector3 justificationOffset = Vector3.zero;
            Vector3 offset = Vector3.zero;
            int vert_index_X4 = 0;
            int sprite_index_X4 = 0;

            int wordCount = 0;
            int lineCount = 0;
            int lastLine = 0;
            bool isFirstSeperator = false;

            bool isStartOfWord = false;
            int wordFirstChar = 0;
            int wordLastChar = 0;

            // Second Pass : Line Justification, UV Mapping, Character & Line Visibility & more.
            float lossyScale = m_previousLossyScaleY = this.transform.lossyScale.y;

            Color32 underlineColor = Color.white;
            Color32 strikethroughColor = Color.white;
            Color32 highlightColor = new Color32(255, 255, 0, 64);
            float xScale = 0;
            float xScaleMax = 0;
            float underlineStartScale = 0;
            float underlineEndScale = 0;
            float underlineMaxScale = 0;
            float underlineBaseLine = k_LargePositiveFloat;
            int lastPage = 0;

            float strikethroughPointSize = 0;
            float strikethroughScale = 0;
            float strikethroughBaseline = 0;

            TMP_CharacterInfo[] characterInfos = m_textInfo.characterInfo;
            #region Handle Line Justification & UV Mapping & Character Visibility & More
            for (int i = 0; i < m_characterCount; i++)
            {
                TMP_FontAsset currentFontAsset = characterInfos[i].fontAsset;

                char currentCharacter = characterInfos[i].character;

                int currentLine = characterInfos[i].lineNumber;
                TMP_LineInfo lineInfo = m_textInfo.lineInfo[currentLine];
                lineCount = currentLine + 1;

                TextAlignmentOptions lineAlignment = lineInfo.alignment;

                // Process Line Justification
                #region Handle Line Justification
                switch (lineAlignment)
                {
                    case TextAlignmentOptions.TopLeft:
                    case TextAlignmentOptions.Left:
                    case TextAlignmentOptions.BottomLeft:
                    case TextAlignmentOptions.BaselineLeft:
                    case TextAlignmentOptions.MidlineLeft:
                    case TextAlignmentOptions.CaplineLeft:
                            if (!m_isRightToLeft)
                                justificationOffset = new Vector3(0 + lineInfo.marginLeft, 0, 0);
                            else
                                justificationOffset = new Vector3(0 - lineInfo.maxAdvance, 0, 0);
                            break;

                    case TextAlignmentOptions.Top:
                    case TextAlignmentOptions.Center:
                    case TextAlignmentOptions.Bottom:
                    case TextAlignmentOptions.Baseline:
                    case TextAlignmentOptions.Midline:
                    case TextAlignmentOptions.Capline:
                        justificationOffset = new Vector3(lineInfo.marginLeft + lineInfo.width / 2 - lineInfo.maxAdvance / 2, 0, 0);
                        break;

                    case TextAlignmentOptions.TopGeoAligned:
                    case TextAlignmentOptions.CenterGeoAligned:
                    case TextAlignmentOptions.BottomGeoAligned:
                    case TextAlignmentOptions.BaselineGeoAligned:
                    case TextAlignmentOptions.MidlineGeoAligned:
                    case TextAlignmentOptions.CaplineGeoAligned:
                        justificationOffset = new Vector3(lineInfo.marginLeft + lineInfo.width / 2 - (lineInfo.lineExtents.min.x + lineInfo.lineExtents.max.x) / 2, 0, 0);
                        break;

                    case TextAlignmentOptions.TopRight:
                    case TextAlignmentOptions.Right:
                    case TextAlignmentOptions.BottomRight:
                    case TextAlignmentOptions.BaselineRight:
                    case TextAlignmentOptions.MidlineRight:
                    case TextAlignmentOptions.CaplineRight:
                        if (!m_isRightToLeft)
                            justificationOffset = new Vector3(lineInfo.marginLeft + lineInfo.width - lineInfo.maxAdvance, 0, 0);
                        else
                            justificationOffset = new Vector3(lineInfo.marginLeft + lineInfo.width, 0, 0);
                        break;

                    case TextAlignmentOptions.TopJustified:
                    case TextAlignmentOptions.Justified:
                    case TextAlignmentOptions.BottomJustified:
                    case TextAlignmentOptions.BaselineJustified:
                    case TextAlignmentOptions.MidlineJustified:
                    case TextAlignmentOptions.CaplineJustified:
                    case TextAlignmentOptions.TopFlush:
                    case TextAlignmentOptions.Flush:
                    case TextAlignmentOptions.BottomFlush:
                    case TextAlignmentOptions.BaselineFlush:
                    case TextAlignmentOptions.MidlineFlush:
                    case TextAlignmentOptions.CaplineFlush:
                        // Skip Zero Width Characters
                        if (currentCharacter == 0xAD || currentCharacter == 0x200B || currentCharacter == 0x2060) break;

                        char lastCharOfCurrentLine = characterInfos[lineInfo.lastCharacterIndex].character;
                        bool isFlush = ((_HorizontalAlignmentOptions)lineAlignment & _HorizontalAlignmentOptions.Flush) == _HorizontalAlignmentOptions.Flush;

                        // In Justified mode, all lines are justified except the last one.
                        // In Flush mode, all lines are justified.
                        if (char.IsControl(lastCharOfCurrentLine) == false && currentLine < m_lineNumber || isFlush || lineInfo.maxAdvance > lineInfo.width)
                        {
                            // First character of each line.
                            if (currentLine != lastLine || i == 0 || i == m_firstVisibleCharacter)
                            {
                                if (!m_isRightToLeft)
                                    justificationOffset = new Vector3(lineInfo.marginLeft, 0, 0);
                                else
                                    justificationOffset = new Vector3(lineInfo.marginLeft + lineInfo.width, 0, 0);

                                if (char.IsSeparator(currentCharacter))
                                    isFirstSeperator = true;
                                else
                                    isFirstSeperator = false;
                            }
                            else
                            {
                                float gap = !m_isRightToLeft ? lineInfo.width - lineInfo.maxAdvance : lineInfo.width + lineInfo.maxAdvance;

                                int visibleCount = lineInfo.visibleCharacterCount - 1 + lineInfo.controlCharacterCount;

                                // Get the number of spaces for each line ignoring the last character if it is not visible (ie. a space or linefeed).
                                int spaces = (characterInfos[lineInfo.lastCharacterIndex].isVisible ? lineInfo.spaceCount : lineInfo.spaceCount - 1) - lineInfo.controlCharacterCount;

                                if (isFirstSeperator) { spaces -= 1; visibleCount += 1; }

                                float ratio = spaces > 0 ? m_wordWrappingRatios : 1;

                                if (spaces < 1) spaces = 1;

                                if (currentCharacter != 0xA0 && (currentCharacter == 9 || char.IsSeparator((char)currentCharacter)))
                                {
                                    if (!m_isRightToLeft)
                                        justificationOffset += new Vector3(gap * (1 - ratio) / spaces, 0, 0);
                                    else
                                        justificationOffset -= new Vector3(gap * (1 - ratio) / spaces, 0, 0);
                                }
                                else
                                {
                                    if (!m_isRightToLeft)
                                        justificationOffset += new Vector3(gap * ratio / visibleCount, 0, 0);
                                    else
                                        justificationOffset -= new Vector3(gap * ratio / visibleCount, 0, 0);
                                }
                            }
                        }
                        else
                        {
                            if (!m_isRightToLeft)
                                justificationOffset = new Vector3(lineInfo.marginLeft, 0, 0); // Keep last line left justified.
                            else
                                justificationOffset = new Vector3(lineInfo.marginLeft + lineInfo.width, 0, 0); // Keep last line right justified.
                        }
                        //Debug.Log("Char [" + (char)charCode + "] Code:" + charCode + "  Line # " + currentLine + "  Offset:" + justificationOffset + "  # Spaces:" + lineInfo.spaceCount + "  # Characters:" + lineInfo.characterCount);
                        break;
                }
                #endregion End Text Justification

                offset = anchorOffset + justificationOffset;

                // Handle UV2 mapping options and packing of scale information into UV2.
                #region Handling of UV2 mapping & Scale packing
                bool isCharacterVisible = characterInfos[i].isVisible;
                if (isCharacterVisible)
                {
                    TMP_TextElementType elementType = characterInfos[i].elementType;
                    switch (elementType)
                    {
                        // CHARACTERS
                        case TMP_TextElementType.Character:
                            Extents lineExtents = lineInfo.lineExtents;
                            float uvOffset = (m_uvLineOffset * currentLine) % 1; // + m_uvOffset.x;

                            // Setup UV2 based on Character Mapping Options Selected
                            #region Handle UV Mapping Options
                            switch (m_horizontalMapping)
                            {
                                case TextureMappingOptions.Character:
                                    characterInfos[i].vertex_BL.uv2.x = 0; //+ m_uvOffset.x;
                                    characterInfos[i].vertex_TL.uv2.x = 0; //+ m_uvOffset.x;
                                    characterInfos[i].vertex_TR.uv2.x = 1; //+ m_uvOffset.x;
                                    characterInfos[i].vertex_BR.uv2.x = 1; //+ m_uvOffset.x;
                                    break;

                                case TextureMappingOptions.Line:
                                    if (m_textAlignment != TextAlignmentOptions.Justified)
                                    {
                                        characterInfos[i].vertex_BL.uv2.x = (characterInfos[i].vertex_BL.position.x - lineExtents.min.x) / (lineExtents.max.x - lineExtents.min.x) + uvOffset;
                                        characterInfos[i].vertex_TL.uv2.x = (characterInfos[i].vertex_TL.position.x - lineExtents.min.x) / (lineExtents.max.x - lineExtents.min.x) + uvOffset;
                                        characterInfos[i].vertex_TR.uv2.x = (characterInfos[i].vertex_TR.position.x - lineExtents.min.x) / (lineExtents.max.x - lineExtents.min.x) + uvOffset;
                                        characterInfos[i].vertex_BR.uv2.x = (characterInfos[i].vertex_BR.position.x - lineExtents.min.x) / (lineExtents.max.x - lineExtents.min.x) + uvOffset;
                                        break;
                                    }
                                    else // Special Case if Justified is used in Line Mode.
                                    {
                                        characterInfos[i].vertex_BL.uv2.x = (characterInfos[i].vertex_BL.position.x + justificationOffset.x - m_meshExtents.min.x) / (m_meshExtents.max.x - m_meshExtents.min.x) + uvOffset;
                                        characterInfos[i].vertex_TL.uv2.x = (characterInfos[i].vertex_TL.position.x + justificationOffset.x - m_meshExtents.min.x) / (m_meshExtents.max.x - m_meshExtents.min.x) + uvOffset;
                                        characterInfos[i].vertex_TR.uv2.x = (characterInfos[i].vertex_TR.position.x + justificationOffset.x - m_meshExtents.min.x) / (m_meshExtents.max.x - m_meshExtents.min.x) + uvOffset;
                                        characterInfos[i].vertex_BR.uv2.x = (characterInfos[i].vertex_BR.position.x + justificationOffset.x - m_meshExtents.min.x) / (m_meshExtents.max.x - m_meshExtents.min.x) + uvOffset;
                                        break;
                                    }

                                case TextureMappingOptions.Paragraph:
                                    characterInfos[i].vertex_BL.uv2.x = (characterInfos[i].vertex_BL.position.x + justificationOffset.x - m_meshExtents.min.x) / (m_meshExtents.max.x - m_meshExtents.min.x) + uvOffset;
                                    characterInfos[i].vertex_TL.uv2.x = (characterInfos[i].vertex_TL.position.x + justificationOffset.x - m_meshExtents.min.x) / (m_meshExtents.max.x - m_meshExtents.min.x) + uvOffset;
                                    characterInfos[i].vertex_TR.uv2.x = (characterInfos[i].vertex_TR.position.x + justificationOffset.x - m_meshExtents.min.x) / (m_meshExtents.max.x - m_meshExtents.min.x) + uvOffset;
                                    characterInfos[i].vertex_BR.uv2.x = (characterInfos[i].vertex_BR.position.x + justificationOffset.x - m_meshExtents.min.x) / (m_meshExtents.max.x - m_meshExtents.min.x) + uvOffset;
                                    break;

                                case TextureMappingOptions.MatchAspect:

                                    switch (m_verticalMapping)
                                    {
                                        case TextureMappingOptions.Character:
                                            characterInfos[i].vertex_BL.uv2.y = 0; //+ m_uvOffset.y;
                                            characterInfos[i].vertex_TL.uv2.y = 1; //+ m_uvOffset.y;
                                            characterInfos[i].vertex_TR.uv2.y = 0; //+ m_uvOffset.y;
                                            characterInfos[i].vertex_BR.uv2.y = 1; //+ m_uvOffset.y;
                                            break;

                                        case TextureMappingOptions.Line:
                                            characterInfos[i].vertex_BL.uv2.y = (characterInfos[i].vertex_BL.position.y - lineExtents.min.y) / (lineExtents.max.y - lineExtents.min.y) + uvOffset;
                                            characterInfos[i].vertex_TL.uv2.y = (characterInfos[i].vertex_TL.position.y - lineExtents.min.y) / (lineExtents.max.y - lineExtents.min.y) + uvOffset;
                                            characterInfos[i].vertex_TR.uv2.y = characterInfos[i].vertex_BL.uv2.y;
                                            characterInfos[i].vertex_BR.uv2.y = characterInfos[i].vertex_TL.uv2.y;
                                            break;

                                        case TextureMappingOptions.Paragraph:
                                            characterInfos[i].vertex_BL.uv2.y = (characterInfos[i].vertex_BL.position.y - m_meshExtents.min.y) / (m_meshExtents.max.y - m_meshExtents.min.y) + uvOffset;
                                            characterInfos[i].vertex_TL.uv2.y = (characterInfos[i].vertex_TL.position.y - m_meshExtents.min.y) / (m_meshExtents.max.y - m_meshExtents.min.y) + uvOffset;
                                            characterInfos[i].vertex_TR.uv2.y = characterInfos[i].vertex_BL.uv2.y;
                                            characterInfos[i].vertex_BR.uv2.y = characterInfos[i].vertex_TL.uv2.y;
                                            break;

                                        case TextureMappingOptions.MatchAspect:
                                            Debug.Log("ERROR: Cannot Match both Vertical & Horizontal.");
                                            break;
                                    }

                                    //float xDelta = 1 - (_uv2s[vert_index + 0].y * textMeshCharacterInfo[i].AspectRatio); // Left aligned
                                    float xDelta = (1 - ((characterInfos[i].vertex_BL.uv2.y + characterInfos[i].vertex_TL.uv2.y) * characterInfos[i].aspectRatio)) / 2; // Center of Rectangle

                                    characterInfos[i].vertex_BL.uv2.x = (characterInfos[i].vertex_BL.uv2.y * characterInfos[i].aspectRatio) + xDelta + uvOffset;
                                    characterInfos[i].vertex_TL.uv2.x = characterInfos[i].vertex_BL.uv2.x;
                                    characterInfos[i].vertex_TR.uv2.x = (characterInfos[i].vertex_TL.uv2.y * characterInfos[i].aspectRatio) + xDelta + uvOffset;
                                    characterInfos[i].vertex_BR.uv2.x = characterInfos[i].vertex_TR.uv2.x;
                                    break;
                            }

                            switch (m_verticalMapping)
                            {
                                case TextureMappingOptions.Character:
                                    characterInfos[i].vertex_BL.uv2.y = 0; //+ m_uvOffset.y;
                                    characterInfos[i].vertex_TL.uv2.y = 1; //+ m_uvOffset.y;
                                    characterInfos[i].vertex_TR.uv2.y = 1; //+ m_uvOffset.y;
                                    characterInfos[i].vertex_BR.uv2.y = 0; //+ m_uvOffset.y;
                                    break;

                                case TextureMappingOptions.Line:
                                    characterInfos[i].vertex_BL.uv2.y = (characterInfos[i].vertex_BL.position.y - lineInfo.descender) / (lineInfo.ascender - lineInfo.descender); // + m_uvOffset.y;
                                    characterInfos[i].vertex_TL.uv2.y = (characterInfos[i].vertex_TL.position.y - lineInfo.descender) / (lineInfo.ascender - lineInfo.descender); // + m_uvOffset.y;
                                    characterInfos[i].vertex_TR.uv2.y = characterInfos[i].vertex_TL.uv2.y;
                                    characterInfos[i].vertex_BR.uv2.y = characterInfos[i].vertex_BL.uv2.y;
                                    break;

                                case TextureMappingOptions.Paragraph:
                                    characterInfos[i].vertex_BL.uv2.y = (characterInfos[i].vertex_BL.position.y - m_meshExtents.min.y) / (m_meshExtents.max.y - m_meshExtents.min.y); // + m_uvOffset.y;
                                    characterInfos[i].vertex_TL.uv2.y = (characterInfos[i].vertex_TL.position.y - m_meshExtents.min.y) / (m_meshExtents.max.y - m_meshExtents.min.y); // + m_uvOffset.y;
                                    characterInfos[i].vertex_TR.uv2.y = characterInfos[i].vertex_TL.uv2.y;
                                    characterInfos[i].vertex_BR.uv2.y = characterInfos[i].vertex_BL.uv2.y;
                                    break;

                                case TextureMappingOptions.MatchAspect:
                                    float yDelta = (1 - ((characterInfos[i].vertex_BL.uv2.x + characterInfos[i].vertex_TR.uv2.x) / characterInfos[i].aspectRatio)) / 2; // Center of Rectangle

                                    characterInfos[i].vertex_BL.uv2.y = yDelta + (characterInfos[i].vertex_BL.uv2.x / characterInfos[i].aspectRatio); // + m_uvOffset.y;
                                    characterInfos[i].vertex_TL.uv2.y = yDelta + (characterInfos[i].vertex_TR.uv2.x / characterInfos[i].aspectRatio); // + m_uvOffset.y;
                                    characterInfos[i].vertex_BR.uv2.y = characterInfos[i].vertex_BL.uv2.y;
                                    characterInfos[i].vertex_TR.uv2.y = characterInfos[i].vertex_TL.uv2.y;
                                    break;
                            }
                            #endregion

                            // Pack UV's so that we can pass Xscale needed for Shader to maintain 1:1 ratio.
                            #region Pack Scale into UV2
                            xScale = characterInfos[i].scale * Mathf.Abs(lossyScale) * (1 - m_charWidthAdjDelta);
                            if (!characterInfos[i].isUsingAlternateTypeface && (characterInfos[i].style & FontStyles.Bold) == FontStyles.Bold) xScale *= -1;

                            //int isBold = (m_textInfo.characterInfo[i].style & FontStyles.Bold) == FontStyles.Bold ? 1 : 0;
                            //Vector2 vertexData = new Vector2(isBold, xScale);
                            //characterInfos[i].vertex_BL.uv4 = vertexData;
                            //characterInfos[i].vertex_TL.uv4 = vertexData;
                            //characterInfos[i].vertex_TR.uv4 = vertexData;
                            //characterInfos[i].vertex_BR.uv4 = vertexData;

                            float x0 = characterInfos[i].vertex_BL.uv2.x;
                            float y0 = characterInfos[i].vertex_BL.uv2.y;
                            float x1 = characterInfos[i].vertex_TR.uv2.x;
                            float y1 = characterInfos[i].vertex_TR.uv2.y;

                            float dx = (int)x0;
                            float dy = (int)y0;

                            x0 = x0 - dx;
                            x1 = x1 - dx;
                            y0 = y0 - dy;
                            y1 = y1 - dy;

                            // Optimization to avoid having a vector2 returned from the Pack UV function.
                            characterInfos[i].vertex_BL.uv2.x = PackUV(x0, y0); characterInfos[i].vertex_BL.uv2.y = xScale;
                            characterInfos[i].vertex_TL.uv2.x = PackUV(x0, y1); characterInfos[i].vertex_TL.uv2.y = xScale;
                            characterInfos[i].vertex_TR.uv2.x = PackUV(x1, y1); characterInfos[i].vertex_TR.uv2.y = xScale;
                            characterInfos[i].vertex_BR.uv2.x = PackUV(x1, y0); characterInfos[i].vertex_BR.uv2.y = xScale;
                            #endregion
                            break;
                        
                        // SPRITES
                        case TMP_TextElementType.Sprite:
                            // Nothing right now
                            break;
                    }

                    // Handle maxVisibleCharacters, maxVisibleLines and Overflow Page Mode.
                    #region Handle maxVisibleCharacters / maxVisibleLines / Page Mode
                    if (i < m_maxVisibleCharacters && wordCount < m_maxVisibleWords && currentLine < m_maxVisibleLines && m_overflowMode != TextOverflowModes.Page)
                    {
                        characterInfos[i].vertex_BL.position += offset;
                        characterInfos[i].vertex_TL.position += offset;
                        characterInfos[i].vertex_TR.position += offset;
                        characterInfos[i].vertex_BR.position += offset;
                    }
                    else if (i < m_maxVisibleCharacters && wordCount < m_maxVisibleWords && currentLine < m_maxVisibleLines && m_overflowMode == TextOverflowModes.Page && characterInfos[i].pageNumber == pageToDisplay)
                    {
                        characterInfos[i].vertex_BL.position += offset;
                        characterInfos[i].vertex_TL.position += offset;
                        characterInfos[i].vertex_TR.position += offset;
                        characterInfos[i].vertex_BR.position += offset;
                    }
                    else
                    {
                        characterInfos[i].vertex_BL.position = Vector3.zero;
                        characterInfos[i].vertex_TL.position = Vector3.zero;
                        characterInfos[i].vertex_TR.position = Vector3.zero;
                        characterInfos[i].vertex_BR.position = Vector3.zero;
                        characterInfos[i].isVisible = false;
                    }
                    #endregion


                    // Fill Vertex Buffers for the various types of element
                    if (elementType == TMP_TextElementType.Character)
                    {
                        FillCharacterVertexBuffers(i, vert_index_X4, m_isVolumetricText);
                    }
                    else if (elementType == TMP_TextElementType.Sprite)
                    {
                        FillSpriteVertexBuffers(i, sprite_index_X4);
                    }
                }
                #endregion

                // Apply Alignment and Justification Offset
                m_textInfo.characterInfo[i].bottomLeft += offset;
                m_textInfo.characterInfo[i].topLeft += offset;
                m_textInfo.characterInfo[i].topRight += offset;
                m_textInfo.characterInfo[i].bottomRight += offset;

                m_textInfo.characterInfo[i].origin += offset.x;
                m_textInfo.characterInfo[i].xAdvance += offset.x;

                m_textInfo.characterInfo[i].ascender += offset.y;
                m_textInfo.characterInfo[i].descender += offset.y;
                m_textInfo.characterInfo[i].baseLine += offset.y;

                // Update MeshExtents
                if (isCharacterVisible)
                {
                    //m_meshExtents.min = new Vector2(Mathf.Min(m_meshExtents.min.x, m_textInfo.characterInfo[i].bottomLeft.x), Mathf.Min(m_meshExtents.min.y, m_textInfo.characterInfo[i].bottomLeft.y));
                    //m_meshExtents.max = new Vector2(Mathf.Max(m_meshExtents.max.x, m_textInfo.characterInfo[i].topRight.x), Mathf.Max(m_meshExtents.max.y, m_textInfo.characterInfo[i].topLeft.y));
                }

                // Need to recompute lineExtent to account for the offset from justification.
                #region Adjust lineExtents resulting from alignment offset
                if (currentLine != lastLine || i == m_characterCount - 1)
                {
                    // Update the previous line's extents
                    if (currentLine != lastLine)
                    {
                        m_textInfo.lineInfo[lastLine].baseline += offset.y;
                        m_textInfo.lineInfo[lastLine].ascender += offset.y;
                        m_textInfo.lineInfo[lastLine].descender += offset.y;

                        m_textInfo.lineInfo[lastLine].lineExtents.min = new Vector2(m_textInfo.characterInfo[m_textInfo.lineInfo[lastLine].firstCharacterIndex].bottomLeft.x, m_textInfo.lineInfo[lastLine].descender);
                        m_textInfo.lineInfo[lastLine].lineExtents.max = new Vector2(m_textInfo.characterInfo[m_textInfo.lineInfo[lastLine].lastVisibleCharacterIndex].topRight.x, m_textInfo.lineInfo[lastLine].ascender);
                    }

                    // Update the current line's extents
                    if (i == m_characterCount - 1)
                    {
                        m_textInfo.lineInfo[currentLine].baseline += offset.y;
                        m_textInfo.lineInfo[currentLine].ascender += offset.y;
                        m_textInfo.lineInfo[currentLine].descender += offset.y;

                        m_textInfo.lineInfo[currentLine].lineExtents.min = new Vector2(m_textInfo.characterInfo[m_textInfo.lineInfo[currentLine].firstCharacterIndex].bottomLeft.x, m_textInfo.lineInfo[currentLine].descender);
                        m_textInfo.lineInfo[currentLine].lineExtents.max = new Vector2(m_textInfo.characterInfo[m_textInfo.lineInfo[currentLine].lastVisibleCharacterIndex].topRight.x, m_textInfo.lineInfo[currentLine].ascender);
                    }
                }
                #endregion


                // Track Word Count per line and for the object
                #region Track Word Count
                if (char.IsLetterOrDigit(currentCharacter) || currentCharacter == 0x2D || currentCharacter == 0xAD || currentCharacter == 0x2010 || currentCharacter == 0x2011)
                {
                    if (isStartOfWord == false)
                    {
                        isStartOfWord = true;
                        wordFirstChar = i;
                    }

                    // If last character is a word
                    if (isStartOfWord && i == m_characterCount - 1)
                    {
                        int size = m_textInfo.wordInfo.Length;
                        int index = m_textInfo.wordCount;

                        if (m_textInfo.wordCount + 1 > size)
                            TMP_TextInfo.Resize(ref m_textInfo.wordInfo, size + 1);

                        wordLastChar = i;

                        m_textInfo.wordInfo[index].firstCharacterIndex = wordFirstChar;
                        m_textInfo.wordInfo[index].lastCharacterIndex = wordLastChar;
                        m_textInfo.wordInfo[index].characterCount = wordLastChar - wordFirstChar + 1;
                        m_textInfo.wordInfo[index].textComponent = this;

                        wordCount += 1;
                        m_textInfo.wordCount += 1;
                        m_textInfo.lineInfo[currentLine].wordCount += 1;
                    }
                }
                else if (isStartOfWord || i == 0 && (!char.IsPunctuation(currentCharacter) || char.IsWhiteSpace(currentCharacter) || currentCharacter == 0x200B || i == m_characterCount - 1))
                {
                    if (i > 0 && i < characterInfos.Length - 1 && i < m_characterCount && (currentCharacter == 39 || currentCharacter == 8217) && char.IsLetterOrDigit(characterInfos[i - 1].character) && char.IsLetterOrDigit(characterInfos[i + 1].character))
                    {

                    }
                    else
                    {
                        wordLastChar = i == m_characterCount - 1 && char.IsLetterOrDigit(currentCharacter) ? i : i - 1;
                        isStartOfWord = false;

                        int size = m_textInfo.wordInfo.Length;
                        int index = m_textInfo.wordCount;

                        if (m_textInfo.wordCount + 1 > size)
                            TMP_TextInfo.Resize(ref m_textInfo.wordInfo, size + 1);

                        m_textInfo.wordInfo[index].firstCharacterIndex = wordFirstChar;
                        m_textInfo.wordInfo[index].lastCharacterIndex = wordLastChar;
                        m_textInfo.wordInfo[index].characterCount = wordLastChar - wordFirstChar + 1;
                        m_textInfo.wordInfo[index].textComponent = this;

                        wordCount += 1;
                        m_textInfo.wordCount += 1;
                        m_textInfo.lineInfo[currentLine].wordCount += 1;
                    }
                }
                #endregion


                // Setup & Handle Underline
                #region Underline
                // NOTE: Need to figure out how underline will be handled with multiple fonts and which font will be used for the underline.
                bool isUnderline = (m_textInfo.characterInfo[i].style & FontStyles.Underline) == FontStyles.Underline;
                if (isUnderline)
                {
                    bool isUnderlineVisible = true;
                    int currentPage = m_textInfo.characterInfo[i].pageNumber;

                    if (i > m_maxVisibleCharacters || currentLine > m_maxVisibleLines || (m_overflowMode == TextOverflowModes.Page && currentPage + 1 != m_pageToDisplay))
                        isUnderlineVisible = false;

                    // We only use the scale of visible characters.
                    if (!char.IsWhiteSpace(currentCharacter) && currentCharacter != 0x200B)
                    {
                        underlineMaxScale = Mathf.Max(underlineMaxScale, m_textInfo.characterInfo[i].scale);
                        xScaleMax = Mathf.Max(xScaleMax, Mathf.Abs(xScale));
                        underlineBaseLine = Mathf.Min(currentPage == lastPage ? underlineBaseLine : k_LargePositiveFloat, m_textInfo.characterInfo[i].baseLine + font.faceInfo.underlineOffset * underlineMaxScale);
                        lastPage = currentPage; // Need to track pages to ensure we reset baseline for the new pages.
                    }

                    if (beginUnderline == false && isUnderlineVisible == true && i <= lineInfo.lastVisibleCharacterIndex && currentCharacter != 10 && currentCharacter != 13)
                    {
                        if (i == lineInfo.lastVisibleCharacterIndex && char.IsSeparator(currentCharacter))
                        { }
                        else
                        {
                            beginUnderline = true;
                            underlineStartScale = m_textInfo.characterInfo[i].scale;
                            if (underlineMaxScale == 0)
                            {
                                underlineMaxScale = underlineStartScale;
                                xScaleMax = xScale;
                            }
                            underline_start = new Vector3(m_textInfo.characterInfo[i].bottomLeft.x, underlineBaseLine, 0);
                            underlineColor = m_textInfo.characterInfo[i].underlineColor;
                        }
                    }

                    // End Underline if text only contains one character.
                    if (beginUnderline && m_characterCount == 1)
                    {
                        beginUnderline = false;
                        underline_end = new Vector3(m_textInfo.characterInfo[i].topRight.x, underlineBaseLine, 0);
                        underlineEndScale = m_textInfo.characterInfo[i].scale;

                        DrawUnderlineMesh(underline_start, underline_end, ref last_vert_index, underlineStartScale, underlineEndScale, underlineMaxScale, xScaleMax, underlineColor);
                        underlineMaxScale = 0;
                        xScaleMax = 0;
                        underlineBaseLine = k_LargePositiveFloat;
                    }
                    else if (beginUnderline && (i == lineInfo.lastCharacterIndex || i >= lineInfo.lastVisibleCharacterIndex))
                    {
                        // Terminate underline at previous visible character if space or carriage return.
                        if (char.IsWhiteSpace(currentCharacter) || currentCharacter == 0x200B)
                        {
                            int lastVisibleCharacterIndex = lineInfo.lastVisibleCharacterIndex;
                            underline_end = new Vector3(m_textInfo.characterInfo[lastVisibleCharacterIndex].topRight.x, underlineBaseLine, 0);
                            underlineEndScale = m_textInfo.characterInfo[lastVisibleCharacterIndex].scale;
                        }
                        else
                        {   // End underline if last character of the line.
                            underline_end = new Vector3(m_textInfo.characterInfo[i].topRight.x, underlineBaseLine, 0);
                            underlineEndScale = m_textInfo.characterInfo[i].scale;
                        }

                        beginUnderline = false;
                        DrawUnderlineMesh(underline_start, underline_end, ref last_vert_index, underlineStartScale, underlineEndScale, underlineMaxScale, xScaleMax, underlineColor);
                        underlineMaxScale = 0;
                        xScaleMax = 0;
                        underlineBaseLine = k_LargePositiveFloat;
                    }
                    else if (beginUnderline && !isUnderlineVisible)
                    {
                        beginUnderline = false;
                        underline_end = new Vector3(m_textInfo.characterInfo[i - 1].topRight.x, underlineBaseLine, 0);
                        underlineEndScale = m_textInfo.characterInfo[i - 1].scale;

                        DrawUnderlineMesh(underline_start, underline_end, ref last_vert_index, underlineStartScale, underlineEndScale, underlineMaxScale, xScaleMax, underlineColor);
                        underlineMaxScale = 0;
                        xScaleMax = 0;
                        underlineBaseLine = k_LargePositiveFloat;
                    }
                    else if (beginUnderline && i < m_characterCount - 1 && !underlineColor.Compare(m_textInfo.characterInfo[i + 1].underlineColor))
                    {
                        // End underline if underline color has changed.
                        beginUnderline = false;
                        underline_end = new Vector3(m_textInfo.characterInfo[i].topRight.x, underlineBaseLine, 0);
                        underlineEndScale = m_textInfo.characterInfo[i].scale;

                        DrawUnderlineMesh(underline_start, underline_end, ref last_vert_index, underlineStartScale, underlineEndScale, underlineMaxScale, xScaleMax, underlineColor);
                        underlineMaxScale = 0;
                        xScaleMax = 0;
                        underlineBaseLine = k_LargePositiveFloat;
                    }
                }
                else
                {
                    // End Underline
                    if (beginUnderline == true)
                    {
                        beginUnderline = false;
                        underline_end = new Vector3(m_textInfo.characterInfo[i - 1].topRight.x, underlineBaseLine, 0);
                        underlineEndScale = m_textInfo.characterInfo[i - 1].scale;

                        DrawUnderlineMesh(underline_start, underline_end, ref last_vert_index, underlineStartScale, underlineEndScale, underlineMaxScale, xScaleMax, underlineColor);
                        underlineMaxScale = 0;
                        xScaleMax = 0;
                        underlineBaseLine = k_LargePositiveFloat;
                    }
                }
                #endregion


                // Setup & Handle Strikethrough
                #region Strikethrough
                // NOTE: Need to figure out how underline will be handled with multiple fonts and which font will be used for the underline.
                bool isStrikethrough = (m_textInfo.characterInfo[i].style & FontStyles.Strikethrough) == FontStyles.Strikethrough;
                float strikethroughOffset = currentFontAsset.faceInfo.strikethroughOffset;

                if (isStrikethrough)
                {
                    bool isStrikeThroughVisible = true;

                    if (i > m_maxVisibleCharacters || currentLine > m_maxVisibleLines || (m_overflowMode == TextOverflowModes.Page && m_textInfo.characterInfo[i].pageNumber + 1 != m_pageToDisplay))
                        isStrikeThroughVisible = false;

                    if (beginStrikethrough == false && isStrikeThroughVisible && i <= lineInfo.lastVisibleCharacterIndex && currentCharacter != 10 && currentCharacter != 13)
                    {
                        if (i == lineInfo.lastVisibleCharacterIndex && char.IsSeparator(currentCharacter))
                        { }
                        else
                        {
                            beginStrikethrough = true;
                            strikethroughPointSize = m_textInfo.characterInfo[i].pointSize;
                            strikethroughScale = m_textInfo.characterInfo[i].scale;
                            strikethrough_start = new Vector3(m_textInfo.characterInfo[i].bottomLeft.x, m_textInfo.characterInfo[i].baseLine + strikethroughOffset * strikethroughScale, 0);
                            strikethroughColor = m_textInfo.characterInfo[i].strikethroughColor;
                            strikethroughBaseline = m_textInfo.characterInfo[i].baseLine;
                            //Debug.Log("Char [" + currentCharacter + "] Start Strikethrough POS: " + strikethrough_start);
                        }
                    }

                    // End Strikethrough if text only contains one character.
                    if (beginStrikethrough && m_characterCount == 1)
                    {
                        beginStrikethrough = false;
                        strikethrough_end = new Vector3(m_textInfo.characterInfo[i].topRight.x, m_textInfo.characterInfo[i].baseLine + strikethroughOffset * strikethroughScale, 0);

                        DrawUnderlineMesh(strikethrough_start, strikethrough_end, ref last_vert_index, strikethroughScale, strikethroughScale, strikethroughScale, xScale, strikethroughColor);
                    }
                    else if (beginStrikethrough && i == lineInfo.lastCharacterIndex)
                    {
                        // Terminate Strikethrough at previous visible character if space or carriage return.
                        if (char.IsWhiteSpace(currentCharacter) || currentCharacter == 0x200B)
                        {
                            int lastVisibleCharacterIndex = lineInfo.lastVisibleCharacterIndex;
                            strikethrough_end = new Vector3(m_textInfo.characterInfo[lastVisibleCharacterIndex].topRight.x, m_textInfo.characterInfo[lastVisibleCharacterIndex].baseLine + strikethroughOffset * strikethroughScale, 0);
                        }
                        else
                        {
                            // Terminate Strikethrough at last character of line.
                            strikethrough_end = new Vector3(m_textInfo.characterInfo[i].topRight.x, m_textInfo.characterInfo[i].baseLine + strikethroughOffset * strikethroughScale, 0);
                        }

                        beginStrikethrough = false;
                        DrawUnderlineMesh(strikethrough_start, strikethrough_end, ref last_vert_index, strikethroughScale, strikethroughScale, strikethroughScale, xScale, strikethroughColor);
                    }
                    else if (beginStrikethrough && i < m_characterCount && (m_textInfo.characterInfo[i + 1].pointSize != strikethroughPointSize || !TMP_Math.Approximately(m_textInfo.characterInfo[i + 1].baseLine + offset.y, strikethroughBaseline)))
                    {
                        // Terminate Strikethrough if scale changes.
                        beginStrikethrough = false;

                        int lastVisibleCharacterIndex = lineInfo.lastVisibleCharacterIndex;
                        if (i > lastVisibleCharacterIndex)
                            strikethrough_end = new Vector3(m_textInfo.characterInfo[lastVisibleCharacterIndex].topRight.x, m_textInfo.characterInfo[lastVisibleCharacterIndex].baseLine + strikethroughOffset * strikethroughScale, 0);
                        else
                            strikethrough_end = new Vector3(m_textInfo.characterInfo[i].topRight.x, m_textInfo.characterInfo[i].baseLine + strikethroughOffset * strikethroughScale, 0);

                        DrawUnderlineMesh(strikethrough_start, strikethrough_end, ref last_vert_index, strikethroughScale, strikethroughScale, strikethroughScale, xScale, strikethroughColor);
                        //Debug.Log("Char [" + currentCharacter + "] at Index: " + i + "  End Strikethrough POS: " + strikethrough_end + "  Baseline: " + m_textInfo.characterInfo[i].baseLine.ToString("f3"));
                    }
                    else if (beginStrikethrough && i < m_characterCount && currentFontAsset.GetInstanceID() != characterInfos[i + 1].fontAsset.GetInstanceID())
                    {
                        // Terminate Strikethrough if font asset changes.
                        beginStrikethrough = false;
                        strikethrough_end = new Vector3(m_textInfo.characterInfo[i].topRight.x, m_textInfo.characterInfo[i].baseLine + strikethroughOffset * strikethroughScale, 0);

                        DrawUnderlineMesh(strikethrough_start, strikethrough_end, ref last_vert_index, strikethroughScale, strikethroughScale, strikethroughScale, xScale, strikethroughColor);
                    }
                    else if (beginStrikethrough && !isStrikeThroughVisible)
                    {
                        // Terminate Strikethrough if character is not visible.
                        beginStrikethrough = false;
                        strikethrough_end = new Vector3(m_textInfo.characterInfo[i - 1].topRight.x, m_textInfo.characterInfo[i - 1].baseLine + strikethroughOffset * strikethroughScale, 0);

                        DrawUnderlineMesh(strikethrough_start, strikethrough_end, ref last_vert_index, strikethroughScale, strikethroughScale, strikethroughScale, xScale, strikethroughColor);
                    }
                }
                else
                {
                    // End Strikethrough
                    if (beginStrikethrough == true)
                    {
                        beginStrikethrough = false;
                        strikethrough_end = new Vector3(m_textInfo.characterInfo[i - 1].topRight.x, m_textInfo.characterInfo[i - 1].baseLine + strikethroughOffset * strikethroughScale, 0);

                        DrawUnderlineMesh(strikethrough_start, strikethrough_end, ref last_vert_index, strikethroughScale, strikethroughScale, strikethroughScale, xScale, strikethroughColor);
                    }
                }
                #endregion


                // HANDLE TEXT HIGHLIGHTING
                #region Text Highlighting
                bool isHighlight = (m_textInfo.characterInfo[i].style & FontStyles.Highlight) == FontStyles.Highlight;
                if (isHighlight)
                {
                    bool isHighlightVisible = true;
                    int currentPage = m_textInfo.characterInfo[i].pageNumber;

                    if (i > m_maxVisibleCharacters || currentLine > m_maxVisibleLines || (m_overflowMode == TextOverflowModes.Page && currentPage + 1 != m_pageToDisplay))
                        isHighlightVisible = false;

                    if (beginHighlight == false && isHighlightVisible == true && i <= lineInfo.lastVisibleCharacterIndex && currentCharacter != 10 && currentCharacter != 13)
                    {
                        if (i == lineInfo.lastVisibleCharacterIndex && char.IsSeparator(currentCharacter))
                        { }
                        else
                        {
                            beginHighlight = true;
                            highlight_start = k_LargePositiveVector2;
                            highlight_end = k_LargeNegativeVector2;
                            highlightColor = m_textInfo.characterInfo[i].highlightColor;
                        }
                    }

                    if (beginHighlight)
                    {
                        Color32 currentHighlightColor = m_textInfo.characterInfo[i].highlightColor;
                        bool isColorTransition = false;

                        // Handle Highlight color changes
                        if (!highlightColor.Compare(currentHighlightColor))
                        {
                            // End drawing at the start of new highlight color to prevent a gap between highlight sections.
                            highlight_end.x = (highlight_end.x + m_textInfo.characterInfo[i].bottomLeft.x) / 2;

                            highlight_start.y = Mathf.Min(highlight_start.y, m_textInfo.characterInfo[i].descender);
                            highlight_end.y = Mathf.Max(highlight_end.y, m_textInfo.characterInfo[i].ascender);

                            DrawTextHighlight(highlight_start, highlight_end, ref last_vert_index, highlightColor);

                            beginHighlight = true;
                            highlight_start = highlight_end;

                            highlight_end = new Vector3(m_textInfo.characterInfo[i].topRight.x, m_textInfo.characterInfo[i].descender, 0);
                            highlightColor = m_textInfo.characterInfo[i].highlightColor;

                            isColorTransition = true;
                        }

                        if (!isColorTransition)
                        {
                            // Use the Min / Max Extents of the Highlight area to handle different character sizes and fonts.
                            highlight_start.x = Mathf.Min(highlight_start.x, m_textInfo.characterInfo[i].bottomLeft.x);
                            highlight_start.y = Mathf.Min(highlight_start.y, m_textInfo.characterInfo[i].descender);

                            highlight_end.x = Mathf.Max(highlight_end.x, m_textInfo.characterInfo[i].topRight.x);
                            highlight_end.y = Mathf.Max(highlight_end.y, m_textInfo.characterInfo[i].ascender); 
                        }
                    }

                    // End Highlight if text only contains one character.
                    if (beginHighlight && m_characterCount == 1)
                    {
                        beginHighlight = false;

                        DrawTextHighlight(highlight_start, highlight_end, ref last_vert_index, highlightColor);
                    }
                    else if (beginHighlight && (i == lineInfo.lastCharacterIndex || i >= lineInfo.lastVisibleCharacterIndex))
                    {
                        beginHighlight = false;
                        DrawTextHighlight(highlight_start, highlight_end, ref last_vert_index, highlightColor);
                    }
                    else if (beginHighlight && !isHighlightVisible)
                    {
                        beginHighlight = false;
                        DrawTextHighlight(highlight_start, highlight_end, ref last_vert_index, highlightColor);
                    }
                }
                else
                {
                    // End Highlight
                    if (beginHighlight == true)
                    {
                        beginHighlight = false;
                        DrawTextHighlight(highlight_start, highlight_end, ref last_vert_index, highlightColor);
                    }
                }
                #endregion


                lastLine = currentLine;
            }
            #endregion


            // METRICS ABOUT THE TEXT OBJECT
            m_textInfo.characterCount = m_characterCount;
            m_textInfo.spriteCount = m_spriteCount;
            m_textInfo.lineCount = lineCount;
            m_textInfo.wordCount = wordCount != 0 && m_characterCount > 0 ? wordCount : 1;
            m_textInfo.pageCount = m_pageNumber + 1;


            ////Profiler.BeginSample("TMP Generate Text - Phase III");
            // Update Mesh Vertex Data
            if (m_renderMode == TextRenderFlags.Render && IsActive())
            {
                // Clear unused vertices
                //m_textInfo.meshInfo[0].ClearUnusedVertices();

                // Sort the geometry of the text object if needed.
                if (m_geometrySortingOrder != VertexSortingOrder.Normal)
                    m_textInfo.meshInfo[0].SortGeometry(VertexSortingOrder.Reverse);

                // Upload Mesh Data
                m_mesh.MarkDynamic();
                m_mesh.vertices = m_textInfo.meshInfo[0].vertices;
                m_mesh.uv = m_textInfo.meshInfo[0].uvs0;
                m_mesh.uv2 = m_textInfo.meshInfo[0].uvs2;
                //m_mesh.uv4 = m_textInfo.meshInfo[0].uvs4;
                m_mesh.colors32 = m_textInfo.meshInfo[0].colors32;

                // Compute Bounds for the mesh. Manual computation is more efficient then using Mesh.recalcualteBounds.
                m_mesh.RecalculateBounds();
                //m_mesh.bounds = new Bounds(new Vector3((m_meshExtents.max.x + m_meshExtents.min.x) / 2, (m_meshExtents.max.y + m_meshExtents.min.y) / 2, 0) + offset, new Vector3(m_meshExtents.max.x - m_meshExtents.min.x, m_meshExtents.max.y - m_meshExtents.min.y, 0));

                for (int i = 1; i < m_textInfo.materialCount; i++)
                {
                    // Clear unused vertices
                    m_textInfo.meshInfo[i].ClearUnusedVertices();

                    if (m_subTextObjects[i] == null) continue;

                    // Sort the geometry of the sub-text objects if needed.
                    if (m_geometrySortingOrder != VertexSortingOrder.Normal)
                        m_textInfo.meshInfo[i].SortGeometry(VertexSortingOrder.Reverse);

                    m_subTextObjects[i].mesh.vertices = m_textInfo.meshInfo[i].vertices;
                    m_subTextObjects[i].mesh.uv = m_textInfo.meshInfo[i].uvs0;
                    m_subTextObjects[i].mesh.uv2 = m_textInfo.meshInfo[i].uvs2;
                    //m_subTextObjects[i].mesh.uv4 = m_textInfo.meshInfo[i].uvs4;
                    m_subTextObjects[i].mesh.colors32 = m_textInfo.meshInfo[i].colors32;

                    m_subTextObjects[i].mesh.RecalculateBounds();

                    // Update the collider on the sub text object
                    //m_subTextObjects[i].UpdateColliders(m_textInfo.meshInfo[i].vertexCount);
                }
            }

            // Event indicating the text has been regenerated.
            TMPro_EventManager.ON_TEXT_CHANGED(this);

            ////Profiler.EndSample();
            //Debug.Log("Done Rendering Text.");
        }


        /// <summary>
        /// Method to return the local corners of the Text Container or RectTransform.
        /// </summary>
        /// <returns></returns>
        protected override Vector3[] GetTextContainerLocalCorners()
        {
            if (m_rectTransform == null) m_rectTransform = this.rectTransform;

            m_rectTransform.GetLocalCorners(m_RectTransformCorners);

            return m_RectTransformCorners;
        }


        /// <summary>
        /// Method to disable the renderers.
        /// </summary>
        void SetMeshFilters(bool state)
        {
            // Parent text object
            if (m_meshFilter != null)
            {
                if (state)
                    m_meshFilter.sharedMesh = m_mesh;
                else
                    m_meshFilter.sharedMesh = null;
            }

            for (int i = 1; i < m_subTextObjects.Length && m_subTextObjects[i] != null; i++)
            {
                if (m_subTextObjects[i].meshFilter != null)
                {
                    if (state)
                        m_subTextObjects[i].meshFilter.sharedMesh = m_subTextObjects[i].mesh;
                    else
                        m_subTextObjects[i].meshFilter.sharedMesh = null;
                }
            }
        }


        /// <summary>
        /// Method to Enable or Disable child SubMesh objects.
        /// </summary>
        /// <param name="state"></param>
        protected override void SetActiveSubMeshes(bool state)
        {
            for (int i = 1; i < m_subTextObjects.Length && m_subTextObjects[i] != null; i++)
            {
                if (m_subTextObjects[i].enabled != state)
                    m_subTextObjects[i].enabled = state;
            }
        }


        /// <summary>
        /// Destroy Sub Mesh Objects
        /// </summary>
        protected override void ClearSubMeshObjects()
        {
            for (int i = 1; i < m_subTextObjects.Length && m_subTextObjects[i] != null; i++)
            {
                Debug.Log("Destroying Sub Text object[" + i + "].");
                DestroyImmediate(m_subTextObjects[i]);
            }
        }


        /// <summary>
        ///  Method returning the compound bounds of the text object and child sub objects.
        /// </summary>
        /// <returns></returns>
        protected override Bounds GetCompoundBounds()
        {
            Bounds mainBounds = m_mesh.bounds;
            Vector3 min = mainBounds.min;
            Vector3 max = mainBounds.max;

            for (int i = 1; i < m_subTextObjects.Length && m_subTextObjects[i] != null; i++)
            {
                Bounds subBounds = m_subTextObjects[i].mesh.bounds;
                min.x = min.x < subBounds.min.x ? min.x : subBounds.min.x;
                min.y = min.y < subBounds.min.y ? min.y : subBounds.min.y;

                max.x = max.x > subBounds.max.x ? max.x : subBounds.max.x;
                max.y = max.y > subBounds.max.y ? max.y : subBounds.max.y;
            }

            Vector3 center = (min + max) / 2;
            Vector2 size = max - min;
            return new Bounds(center, size);
        }


        /// <summary>
        /// Method to Update Scale in UV2
        /// </summary>
        //void UpdateSDFScale(float lossyScale)
        //{
        //    // TODO: Resolve - Underline / Strikethrough segments not getting their SDF Scale adjusted.

        //    //Debug.Log("*** UpdateSDFScale() ***");

        //    // Iterate through each of the characters.
        //    for (int i = 0; i < m_textInfo.characterCount; i++)
        //    {
        //        // Only update scale for visible characters.
        //        if (m_textInfo.characterInfo[i].isVisible && m_textInfo.characterInfo[i].elementType == TMP_TextElementType.Character)
        //        {
        //            float scale = lossyScale * m_textInfo.characterInfo[i].scale * (1 - m_charWidthAdjDelta);
        //            if (!m_textInfo.characterInfo[i].isUsingAlternateTypeface && (m_textInfo.characterInfo[i].style & FontStyles.Bold) == FontStyles.Bold) scale *= -1;

        //            int index = m_textInfo.characterInfo[i].materialReferenceIndex;
        //            int vertexIndex = m_textInfo.characterInfo[i].vertexIndex;

        //            m_textInfo.meshInfo[index].uvs2[vertexIndex + 0].y = scale;
        //            m_textInfo.meshInfo[index].uvs2[vertexIndex + 1].y = scale;
        //            m_textInfo.meshInfo[index].uvs2[vertexIndex + 2].y = scale;
        //            m_textInfo.meshInfo[index].uvs2[vertexIndex + 3].y = scale;
        //        }
        //    }

        //    // Push the updated uv2 scale information to the meshes.
        //    for (int i = 0; i < m_textInfo.meshInfo.Length; i++)
        //    {
        //        if (i == 0)
        //            m_mesh.uv2 = m_textInfo.meshInfo[0].uvs2;
        //        else
        //            m_subTextObjects[i].mesh.uv2 = m_textInfo.meshInfo[i].uvs2;
        //    }
        //}

        /// <summary>
        /// Method to update the SDF Scale in UV2.
        /// </summary>
        /// <param name="scaleDelta"></param>
        void UpdateSDFScale(float scaleDelta)
        {
            if (scaleDelta == 0 || scaleDelta == float.PositiveInfinity)
            {
                m_havePropertiesChanged = true;
                OnPreRenderObject();
                return;
            }

            for (int materialIndex = 0; materialIndex < m_textInfo.materialCount; materialIndex++)
            {
                TMP_MeshInfo meshInfo = m_textInfo.meshInfo[materialIndex];

                for (int i = 0; i < meshInfo.uvs2.Length; i++)
                {
                    meshInfo.uvs2[i].y *= Mathf.Abs(scaleDelta);
                }
            }

            // Push the updated uv2 scale information to the meshes.
            for (int i = 0; i < m_textInfo.meshInfo.Length; i++)
            {
                if (i == 0)
                    m_mesh.uv2 = m_textInfo.meshInfo[0].uvs2;
                else
                    m_subTextObjects[i].mesh.uv2 = m_textInfo.meshInfo[i].uvs2;
            }
        }


        // Function to offset vertices position to account for line spacing changes.
        protected override void AdjustLineOffset(int startIndex, int endIndex, float offset)
        {
            Vector3 vertexOffset = new Vector3(0, offset, 0);

            for (int i = startIndex; i <= endIndex; i++)
            {
                m_textInfo.characterInfo[i].bottomLeft -= vertexOffset;
                m_textInfo.characterInfo[i].topLeft -= vertexOffset;
                m_textInfo.characterInfo[i].topRight -= vertexOffset;
                m_textInfo.characterInfo[i].bottomRight -= vertexOffset;

                m_textInfo.characterInfo[i].ascender -= vertexOffset.y;
                m_textInfo.characterInfo[i].baseLine -= vertexOffset.y;
                m_textInfo.characterInfo[i].descender -= vertexOffset.y;

                if (m_textInfo.characterInfo[i].isVisible)
                {
                    m_textInfo.characterInfo[i].vertex_BL.position -= vertexOffset;
                    m_textInfo.characterInfo[i].vertex_TL.position -= vertexOffset;
                    m_textInfo.characterInfo[i].vertex_TR.position -= vertexOffset;
                    m_textInfo.characterInfo[i].vertex_BR.position -= vertexOffset;
                }
            }
        }

    }
}
#define TMP_PRESENT

using UnityEngine;
using UnityEngine.TextCore;
using UnityEngine.UI;
using UnityEngine.Events;
using UnityEngine.EventSystems;
using System;
using System.Text;
using System.Collections;
using System.Collections.Generic;


namespace TMPro
{
    public interface ITextElement
    {
        Material sharedMaterial { get; }

        void Rebuild(CanvasUpdate update);
        int GetInstanceID();
    }

    public enum TextAlignmentOptions
    {
        TopLeft = _HorizontalAlignmentOptions.Left | _VerticalAlignmentOptions.Top,
        Top = _HorizontalAlignmentOptions.Center | _VerticalAlignmentOptions.Top,
        TopRight = _HorizontalAlignmentOptions.Right | _VerticalAlignmentOptions.Top,
        TopJustified = _HorizontalAlignmentOptions.Justified | _VerticalAlignmentOptions.Top,
        TopFlush = _HorizontalAlignmentOptions.Flush | _VerticalAlignmentOptions.Top,
        TopGeoAligned = _HorizontalAlignmentOptions.Geometry | _VerticalAlignmentOptions.Top,

        Left = _HorizontalAlignmentOptions.Left | _VerticalAlignmentOptions.Middle,
        Center = _HorizontalAlignmentOptions.Center | _VerticalAlignmentOptions.Middle,
        Right = _HorizontalAlignmentOptions.Right | _VerticalAlignmentOptions.Middle,
        Justified = _HorizontalAlignmentOptions.Justified | _VerticalAlignmentOptions.Middle,
        Flush = _HorizontalAlignmentOptions.Flush | _VerticalAlignmentOptions.Middle,
        CenterGeoAligned = _HorizontalAlignmentOptions.Geometry | _VerticalAlignmentOptions.Middle,

        BottomLeft = _HorizontalAlignmentOptions.Left | _VerticalAlignmentOptions.Bottom,
        Bottom = _HorizontalAlignmentOptions.Center | _VerticalAlignmentOptions.Bottom,
        BottomRight = _HorizontalAlignmentOptions.Right | _VerticalAlignmentOptions.Bottom,
        BottomJustified = _HorizontalAlignmentOptions.Justified | _VerticalAlignmentOptions.Bottom,
        BottomFlush = _HorizontalAlignmentOptions.Flush | _VerticalAlignmentOptions.Bottom,
        BottomGeoAligned = _HorizontalAlignmentOptions.Geometry | _VerticalAlignmentOptions.Bottom,

        BaselineLeft = _HorizontalAlignmentOptions.Left | _VerticalAlignmentOptions.Baseline,
        Baseline = _HorizontalAlignmentOptions.Center | _VerticalAlignmentOptions.Baseline,
        BaselineRight = _HorizontalAlignmentOptions.Right | _VerticalAlignmentOptions.Baseline,
        BaselineJustified = _HorizontalAlignmentOptions.Justified | _VerticalAlignmentOptions.Baseline,
        BaselineFlush = _HorizontalAlignmentOptions.Flush | _VerticalAlignmentOptions.Baseline,
        BaselineGeoAligned = _HorizontalAlignmentOptions.Geometry | _VerticalAlignmentOptions.Baseline,

        MidlineLeft = _HorizontalAlignmentOptions.Left | _VerticalAlignmentOptions.Geometry,
        Midline = _HorizontalAlignmentOptions.Center | _VerticalAlignmentOptions.Geometry,
        MidlineRight = _HorizontalAlignmentOptions.Right | _VerticalAlignmentOptions.Geometry,
        MidlineJustified = _HorizontalAlignmentOptions.Justified | _VerticalAlignmentOptions.Geometry,
        MidlineFlush = _HorizontalAlignmentOptions.Flush | _VerticalAlignmentOptions.Geometry,
        MidlineGeoAligned = _HorizontalAlignmentOptions.Geometry | _VerticalAlignmentOptions.Geometry,

        CaplineLeft = _HorizontalAlignmentOptions.Left | _VerticalAlignmentOptions.Capline,
        Capline = _HorizontalAlignmentOptions.Center | _VerticalAlignmentOptions.Capline,
        CaplineRight = _HorizontalAlignmentOptions.Right | _VerticalAlignmentOptions.Capline,
        CaplineJustified = _HorizontalAlignmentOptions.Justified | _VerticalAlignmentOptions.Capline,
        CaplineFlush = _HorizontalAlignmentOptions.Flush | _VerticalAlignmentOptions.Capline,
        CaplineGeoAligned = _HorizontalAlignmentOptions.Geometry | _VerticalAlignmentOptions.Capline
    };

    /// <summary>
    /// Internal horizontal text alignment options.
    /// </summary>
    public enum _HorizontalAlignmentOptions
    {
        Left = 0x1, Center = 0x2, Right = 0x4, Justified = 0x8, Flush = 0x10, Geometry = 0x20
    }

    /// <summary>
    /// Internal vertical text alignment options.
    /// </summary>
    public enum _VerticalAlignmentOptions
    {
        Top = 0x100, Middle = 0x200, Bottom = 0x400, Baseline = 0x800, Geometry = 0x1000, Capline = 0x2000,
    }


    /// <summary>
    /// Flags controlling what vertex data gets pushed to the mesh.
    /// </summary>
    public enum TextRenderFlags
    {
        DontRender = 0x0,
        Render = 0xFF
    };

    public enum TMP_TextElementType { Character, Sprite };
    public enum MaskingTypes { MaskOff = 0, MaskHard = 1, MaskSoft = 2 }; //, MaskTex = 4 };
    public enum TextOverflowModes { Overflow = 0, Ellipsis = 1, Masking = 2, Truncate = 3, ScrollRect = 4, Page = 5, Linked = 6 };
    public enum MaskingOffsetMode { Percentage = 0, Pixel = 1 };
    public enum TextureMappingOptions { Character = 0, Line = 1, Paragraph = 2, MatchAspect = 3 };

    public enum FontStyles { Normal = 0x0, Bold = 0x1, Italic = 0x2, Underline = 0x4, LowerCase = 0x8, UpperCase = 0x10, SmallCaps = 0x20, Strikethrough = 0x40, Superscript = 0x80, Subscript = 0x100, Highlight = 0x200 };
    public enum FontWeight { Thin = 100, ExtraLight = 200, Light = 300, Regular = 400, Medium = 500, SemiBold = 600, Bold = 700, Heavy = 800, Black = 900 };

    /// <summary>
    /// Base class which contains common properties and functions shared between the TextMeshPro and TextMeshProUGUI component.
    /// </summary>
    public abstract class TMP_Text : MaskableGraphic
    {
        /// <summary>
        /// A string containing the text to be displayed.
        /// </summary>
        public string text
        {
            get { return m_text; }
            set { if (m_text == value) return; m_text = old_text = value; m_inputSource = TextInputSources.String; m_havePropertiesChanged = true; m_isCalculateSizeRequired = true; m_isInputParsingRequired = true; SetVerticesDirty(); SetLayoutDirty(); }
        }
        [SerializeField]
        [TextArea(5, 10)]
        protected string m_text;


        /// <summary>
        /// 
        /// </summary>
        public bool isRightToLeftText
        {
            get { return m_isRightToLeft; }
            set { if (m_isRightToLeft == value) return; m_isRightToLeft = value; m_havePropertiesChanged = true; m_isCalculateSizeRequired = true; m_isInputParsingRequired = true; SetVerticesDirty(); SetLayoutDirty(); }
        }
        [SerializeField]
        protected bool m_isRightToLeft = false;


        /// <summary>
        /// The Font Asset to be assigned to this text object.
        /// </summary>
        public TMP_FontAsset font
        {
            get { return m_fontAsset; }
            set { if (m_fontAsset == value) return; m_fontAsset = value; LoadFontAsset(); m_havePropertiesChanged = true; m_isCalculateSizeRequired = true; m_isInputParsingRequired = true; SetVerticesDirty(); SetLayoutDirty(); }
        }
        [SerializeField]
        protected TMP_FontAsset m_fontAsset;
        protected TMP_FontAsset m_currentFontAsset;
        protected bool m_isSDFShader;


        /// <summary>
        /// The material to be assigned to this text object.
        /// </summary>
        public virtual Material fontSharedMaterial
        {
            get { return m_sharedMaterial; }
            set { if (m_sharedMaterial == value) return; SetSharedMaterial(value); m_havePropertiesChanged = true; m_isInputParsingRequired = true; SetVerticesDirty(); SetMaterialDirty(); }
        }
        [SerializeField]
        protected Material m_sharedMaterial;
        protected Material m_currentMaterial;
        protected MaterialReference[] m_materialReferences = new MaterialReference[32];
        protected Dictionary<int, int> m_materialReferenceIndexLookup = new Dictionary<int, int>();

        protected TMP_RichTextTagStack<MaterialReference> m_materialReferenceStack = new TMP_RichTextTagStack<MaterialReference>(new MaterialReference[16]);
        protected int m_currentMaterialIndex;
        //protected int m_sharedMaterialHashCode;


        /// <summary>
        /// An array containing the materials used by the text object.
        /// </summary>
        public virtual Material[] fontSharedMaterials
        {
            get { return GetSharedMaterials(); }
            set { SetSharedMaterials(value); m_havePropertiesChanged = true; m_isInputParsingRequired = true; SetVerticesDirty(); SetMaterialDirty(); }
        }
        [SerializeField]
        protected Material[] m_fontSharedMaterials;


        /// <summary>
        /// The material to be assigned to this text object. An instance of the material will be assigned to the object's renderer.
        /// </summary>
        public Material fontMaterial
        {
            // Return an Instance of the current material.
            get { return GetMaterial(m_sharedMaterial); }

            // Assign new font material
            set
            {
                if (m_sharedMaterial != null && m_sharedMaterial.GetInstanceID() == value.GetInstanceID()) return;

                m_sharedMaterial = value;

                m_padding = GetPaddingForMaterial();
                m_havePropertiesChanged = true;
                m_isInputParsingRequired = true;

                SetVerticesDirty();
                SetMaterialDirty();
            }
        }
        [SerializeField]
        protected Material m_fontMaterial;


        /// <summary>
        /// The materials to be assigned to this text object. An instance of the materials will be assigned.
        /// </summary>
        public virtual Material[] fontMaterials
        {
            get { return GetMaterials(m_fontSharedMaterials); }

            set { SetSharedMaterials(value); m_havePropertiesChanged = true; m_isInputParsingRequired = true; SetVerticesDirty(); SetMaterialDirty(); }
        }
        [SerializeField]
        protected Material[] m_fontMaterials;

        protected bool m_isMaterialDirty;


        /// <summary>
        /// This is the default vertex color assigned to each vertices. Color tags will override vertex colors unless the overrideColorTags is set.
        /// </summary>
        public override Color color
        {
            get { return m_fontColor; }
            set { if (m_fontColor == value) return; m_havePropertiesChanged = true; m_fontColor = value; SetVerticesDirty(); }
        }
        //[UnityEngine.Serialization.FormerlySerializedAs("m_fontColor")] // Required for backwards compatibility with pre-Unity 4.6 releases.
        [SerializeField]
        protected Color32 m_fontColor32 = Color.white;
        [SerializeField]
        protected Color m_fontColor = Color.white;
        protected static Color32 s_colorWhite = new Color32(255, 255, 255, 255);
        protected Color32 m_underlineColor = s_colorWhite;
        protected Color32 m_strikethroughColor = s_colorWhite;
        protected Color32 m_highlightColor = s_colorWhite;
        protected Vector4 m_highlightPadding = Vector4.zero;
        

        /// <summary>
        /// Sets the vertex color alpha value.
        /// </summary>
        public float alpha
        {
            get { return m_fontColor.a; }
            set { if (m_fontColor.a == value) return; m_fontColor.a = value; m_havePropertiesChanged = true; SetVerticesDirty(); }
        }


        /// <summary>
        /// Determines if Vertex Color Gradient should be used
        /// </summary>
        /// <value><c>true</c> if enable vertex gradient; otherwise, <c>false</c>.</value>
        public bool enableVertexGradient
        {
            get { return m_enableVertexGradient; }
            set { if (m_enableVertexGradient == value) return; m_havePropertiesChanged = true; m_enableVertexGradient = value; SetVerticesDirty(); }
        }
        [SerializeField]
        protected bool m_enableVertexGradient;

        [SerializeField]
        protected ColorMode m_colorMode = ColorMode.FourCornersGradient;
        
        /// <summary>
        /// Sets the vertex colors for each of the 4 vertices of the character quads.
        /// </summary>
        /// <value>The color gradient.</value>
        public VertexGradient colorGradient
        {
            get { return m_fontColorGradient; }
            set { m_havePropertiesChanged = true; m_fontColorGradient = value; SetVerticesDirty(); }
        }
        [SerializeField]
        protected VertexGradient m_fontColorGradient = new VertexGradient(Color.white);


        /// <summary>
        /// Set the vertex colors of the 4 vertices of each character quads.
        /// </summary>
        public TMP_ColorGradient colorGradientPreset
        {
            get { return m_fontColorGradientPreset; }
            set { m_havePropertiesChanged = true; m_fontColorGradientPreset = value; SetVerticesDirty(); }
        }
        [SerializeField]
        protected TMP_ColorGradient m_fontColorGradientPreset;


        /// <summary>
        /// Default Sprite Asset used by the text object.
        /// </summary>
        public TMP_SpriteAsset spriteAsset
        {
            get { return m_spriteAsset; }
            set { m_spriteAsset = value; m_havePropertiesChanged = true; m_isInputParsingRequired = true; m_isCalculateSizeRequired = true; SetVerticesDirty(); SetLayoutDirty(); }
        }
        [SerializeField]
        protected TMP_SpriteAsset m_spriteAsset;


        /// <summary>
        /// Determines whether or not the sprite color is multiplies by the vertex color of the text.
        /// </summary>
        public bool tintAllSprites
        {
            get { return m_tintAllSprites; }
            set { if (m_tintAllSprites == value) return; m_tintAllSprites = value; m_havePropertiesChanged = true; SetVerticesDirty(); }
        }
        [SerializeField]
        protected bool m_tintAllSprites;
        protected bool m_tintSprite;
        protected Color32 m_spriteColor;


        /// <summary>
        /// This overrides the color tags forcing the vertex colors to be the default font color.
        /// </summary>
        public bool overrideColorTags
        {
            get { return m_overrideHtmlColors; }
            set { if (m_overrideHtmlColors == value) return; m_havePropertiesChanged = true; m_overrideHtmlColors = value; SetVerticesDirty(); }
        }
        [SerializeField]
        protected bool m_overrideHtmlColors = false;


        /// <summary>
        /// Sets the color of the _FaceColor property of the assigned material. Changing face color will result in an instance of the material.
        /// </summary>
        public Color32 faceColor
        {
            get
            {
                if (m_sharedMaterial == null) return m_faceColor;

                m_faceColor = m_sharedMaterial.GetColor(ShaderUtilities.ID_FaceColor);
                return m_faceColor;
            }

            set { if (m_faceColor.Compare(value)) return; SetFaceColor(value); m_havePropertiesChanged = true; m_faceColor = value; SetVerticesDirty(); SetMaterialDirty(); }
        }
        [SerializeField]
        protected Color32 m_faceColor = Color.white;


        /// <summary>
        /// Sets the color of the _OutlineColor property of the assigned material. Changing outline color will result in an instance of the material.
        /// </summary>
        public Color32 outlineColor
        {
            get
            {
                if (m_sharedMaterial == null) return m_outlineColor;

                m_outlineColor = m_sharedMaterial.GetColor(ShaderUtilities.ID_OutlineColor);
                return m_outlineColor;
            }

            set { if (m_outlineColor.Compare(value)) return; SetOutlineColor(value); m_havePropertiesChanged = true; m_outlineColor = value; SetVerticesDirty(); }
        }
        [SerializeField]
        protected Color32 m_outlineColor = Color.black;


        /// <summary>
        /// Sets the thickness of the outline of the font. Setting this value will result in an instance of the material.
        /// </summary>
        public float outlineWidth
        {
            get
            {
                if (m_sharedMaterial == null) return m_outlineWidth;

                m_outlineWidth = m_sharedMaterial.GetFloat(ShaderUtilities.ID_OutlineWidth);
                return m_outlineWidth;
            }
            set { if (m_outlineWidth == value) return; SetOutlineThickness(value); m_havePropertiesChanged = true; m_outlineWidth = value; SetVerticesDirty(); }
        }
        protected float m_outlineWidth = 0.0f;


        /// <summary>
        /// The point size of the font.
        /// </summary>
        public float fontSize
        {
            get { return m_fontSize; }
            set { if (m_fontSize == value) return; m_havePropertiesChanged = true; m_isCalculateSizeRequired = true; m_fontSize = value; if (!m_enableAutoSizing) m_fontSizeBase = m_fontSize; SetVerticesDirty(); SetLayoutDirty(); }
        }
        [SerializeField]
        protected float m_fontSize = 36; // Font Size
        protected float m_currentFontSize; // Temporary Font Size affected by tags
        [SerializeField]
        protected float m_fontSizeBase = 36;
        protected TMP_RichTextTagStack<float> m_sizeStack = new TMP_RichTextTagStack<float>(16);


        /// <summary>
        /// The scale of the current text.
        /// </summary>
        public float fontScale
        {
            get { return m_fontScale; }
        }


        /// <summary>
        /// Control the weight of the font if an alternative font asset is assigned for the given weight in the font asset editor.
        /// </summary>
        public FontWeight fontWeight
        {
            get { return m_fontWeight; }
            set { if (m_fontWeight == value) return; m_fontWeight = value; m_havePropertiesChanged = true; m_isCalculateSizeRequired = true; m_isInputParsingRequired = true; SetVerticesDirty(); SetLayoutDirty(); }
        }
        [SerializeField]
        protected FontWeight m_fontWeight = FontWeight.Regular;
        protected FontWeight m_FontWeightInternal = FontWeight.Regular;
        protected TMP_RichTextTagStack<FontWeight> m_FontWeightStack = new TMP_RichTextTagStack<FontWeight>(8);

        /// <summary>
        /// 
        /// </summary>
        public float pixelsPerUnit
        {
            get
            {
                var localCanvas = canvas;
                if (!localCanvas)
                    return 1;
                // For dynamic fonts, ensure we use one pixel per pixel on the screen.
                if (!font)
                    return localCanvas.scaleFactor;
                // For non-dynamic fonts, calculate pixels per unit based on specified font size relative to font object's own font size.
                if (m_currentFontAsset == null || m_currentFontAsset.faceInfo.pointSize <= 0 || m_fontSize <= 0)
                    return 1;
                return m_fontSize / m_currentFontAsset.faceInfo.pointSize;
            }
        }


        /// <summary>
        /// Enable text auto-sizing
        /// </summary>
        public bool enableAutoSizing
        {
            get { return m_enableAutoSizing; }
            set { if (m_enableAutoSizing == value) return; m_enableAutoSizing = value; SetVerticesDirty(); SetLayoutDirty(); }
        }
        [SerializeField]
        protected bool m_enableAutoSizing;
        protected float m_maxFontSize; // Used in conjunction with auto-sizing
        protected float m_minFontSize; // Used in conjunction with auto-sizing


        /// <summary>
        /// Minimum point size of the font when text auto-sizing is enabled.
        /// </summary>
        public float fontSizeMin
        {
            get { return m_fontSizeMin; }
            set { if (m_fontSizeMin == value) return; m_fontSizeMin = value; SetVerticesDirty(); SetLayoutDirty(); }
        }
        [SerializeField]
        protected float m_fontSizeMin = 0; // Text Auto Sizing Min Font Size.


        /// <summary>
        /// Maximum point size of the font when text auto-sizing is enabled.
        /// </summary>
        public float fontSizeMax
        {
            get { return m_fontSizeMax; }
            set { if (m_fontSizeMax == value) return; m_fontSizeMax = value; SetVerticesDirty(); SetLayoutDirty(); }
        }
        [SerializeField]
        protected float m_fontSizeMax = 0; // Text Auto Sizing Max Font Size.


        /// <summary>
        /// The style of the text
        /// </summary>
        public FontStyles fontStyle
        {
            get { return m_fontStyle; }
            set { if (m_fontStyle == value) return; m_fontStyle = value; m_havePropertiesChanged = true; m_isCalculateSizeRequired = true; m_isInputParsingRequired = true; SetVerticesDirty(); SetLayoutDirty(); }
        }
        [SerializeField]
        protected FontStyles m_fontStyle = FontStyles.Normal;
        protected FontStyles m_FontStyleInternal = FontStyles.Normal;
        protected TMP_FontStyleStack m_fontStyleStack;

        /// <summary>
        /// Property used in conjunction with padding calculation for the geometry.
        /// </summary>
        public bool isUsingBold { get { return m_isUsingBold; } }
        protected bool m_isUsingBold = false; // Used to ensure GetPadding & Ratios take into consideration bold characters.


        /// <summary>
        /// Text alignment options
        /// </summary>
        public TextAlignmentOptions alignment
        {
            get { return m_textAlignment; }
            set { if (m_textAlignment == value) return; m_havePropertiesChanged = true; m_textAlignment = value; SetVerticesDirty(); }
        }
        [SerializeField]
        [UnityEngine.Serialization.FormerlySerializedAs("m_lineJustification")]
        protected TextAlignmentOptions m_textAlignment = TextAlignmentOptions.TopLeft;
        protected TextAlignmentOptions m_lineJustification;
        protected TMP_RichTextTagStack<TextAlignmentOptions> m_lineJustificationStack = new TMP_RichTextTagStack<TextAlignmentOptions>(new TextAlignmentOptions[16]);
        protected Vector3[] m_textContainerLocalCorners = new Vector3[4];

        /// <summary>
        /// Use the extents of the text geometry for alignment instead of font metrics.
        /// </summary>
        //public bool alignByGeometry
        //{
        //    get { return m_alignByGeometry; }
        //    set { if (m_alignByGeometry == value) return; m_havePropertiesChanged = true; m_alignByGeometry = value; SetVerticesDirty(); }
        //}
        //[SerializeField]
        //protected bool m_alignByGeometry;


        /// <summary>
        /// The amount of additional spacing between characters.
        /// </summary>
        public float characterSpacing
        {
            get { return m_characterSpacing; }
            set { if (m_characterSpacing == value) return; m_havePropertiesChanged = true; m_isCalculateSizeRequired = true;  m_characterSpacing = value; SetVerticesDirty(); SetLayoutDirty(); }
        }
        [SerializeField]
        protected float m_characterSpacing = 0;
        protected float m_cSpacing = 0;
        protected float m_monoSpacing = 0;

        /// <summary>
        /// The amount of additional spacing between words.
        /// </summary>
        public float wordSpacing
        {
            get { return m_wordSpacing; }
            set { if (m_wordSpacing == value) return; m_havePropertiesChanged = true; m_isCalculateSizeRequired = true; m_wordSpacing = value; SetVerticesDirty(); SetLayoutDirty(); }
        }
        [SerializeField]
        protected float m_wordSpacing = 0;

        /// <summary>
        /// The amount of additional spacing to add between each lines of text.
        /// </summary>
        public float lineSpacing
        {
            get { return m_lineSpacing; }
            set { if (m_lineSpacing == value) return; m_havePropertiesChanged = true; m_isCalculateSizeRequired = true; m_lineSpacing = value; SetVerticesDirty(); SetLayoutDirty(); }
        }
        [SerializeField]
        protected float m_lineSpacing = 0;
        protected float m_lineSpacingDelta = 0; // Used with Text Auto Sizing feature
        protected float m_lineHeight = TMP_Math.FLOAT_UNSET; // Used with the <line-height=xx.x> tag.


        /// <summary>
        /// The amount of potential line spacing adjustment before text auto sizing kicks in.
        /// </summary>
        public float lineSpacingAdjustment
        {
            get { return m_lineSpacingMax; }
            set { if (m_lineSpacingMax == value) return; m_havePropertiesChanged = true; m_isCalculateSizeRequired = true; m_lineSpacingMax = value; SetVerticesDirty(); SetLayoutDirty(); }
        }
        [SerializeField]
        protected float m_lineSpacingMax = 0; // Text Auto Sizing Max Line spacing reduction.
        //protected bool m_forceLineBreak;

        /// <summary>
        /// The amount of additional spacing to add between each lines of text.
        /// </summary>
        public float paragraphSpacing
        {
            get { return m_paragraphSpacing; }
            set { if (m_paragraphSpacing == value) return; m_havePropertiesChanged = true; m_isCalculateSizeRequired = true; m_paragraphSpacing = value; SetVerticesDirty(); SetLayoutDirty(); }
        }
        [SerializeField]
        protected float m_paragraphSpacing = 0;


        /// <summary>
        /// Percentage the width of characters can be adjusted before text auto-sizing begins to reduce the point size.
        /// </summary>
        public float characterWidthAdjustment
        {
            get { return m_charWidthMaxAdj; }
            set { if (m_charWidthMaxAdj == value) return; m_havePropertiesChanged = true; m_isCalculateSizeRequired = true; m_charWidthMaxAdj = value; SetVerticesDirty(); SetLayoutDirty(); }
        }
        [SerializeField]
        protected float m_charWidthMaxAdj = 0f; // Text Auto Sizing Max Character Width reduction.
        protected float m_charWidthAdjDelta = 0;


        /// <summary>
        /// Controls whether or not word wrapping is applied. When disabled, the text will be displayed on a single line.
        /// </summary>
        public bool enableWordWrapping
        {
            get { return m_enableWordWrapping; }
            set { if (m_enableWordWrapping == value) return; m_havePropertiesChanged = true; m_isInputParsingRequired = true; m_isCalculateSizeRequired = true; m_enableWordWrapping = value; SetVerticesDirty(); SetLayoutDirty(); }
        }
        [SerializeField]
        protected bool m_enableWordWrapping = false;
        protected bool m_isCharacterWrappingEnabled = false;
        protected bool m_isNonBreakingSpace = false;
        protected bool m_isIgnoringAlignment;

        /// <summary>
        /// Controls the blending between using character and word spacing to fill-in the space for justified text.
        /// </summary>
        public float wordWrappingRatios
        {
            get { return m_wordWrappingRatios; }
            set { if (m_wordWrappingRatios == value) return; m_wordWrappingRatios = value; m_havePropertiesChanged = true; m_isCalculateSizeRequired = true; SetVerticesDirty(); SetLayoutDirty(); }
        }
        [SerializeField]
        protected float m_wordWrappingRatios = 0.4f; // Controls word wrapping ratios between word or characters.


        /// <summary>
        /// 
        /// </summary>
        //public bool enableAdaptiveJustification
        //{
        //    get { return m_enableAdaptiveJustification; }
        //    set { if (m_enableAdaptiveJustification == value) return;  m_enableAdaptiveJustification = value;  m_havePropertiesChanged = true;  m_isCalculateSizeRequired = true;  SetVerticesDirty(); SetLayoutDirty(); }
        //}
        //[SerializeField]
        //protected bool m_enableAdaptiveJustification;
        //protected float m_adaptiveJustificationThreshold = 10.0f;


        /// <summary>
        /// Controls the Text Overflow Mode
        /// </summary>
        public TextOverflowModes overflowMode
        {
            get { return m_overflowMode; }
            set { if (m_overflowMode == value) return; m_overflowMode = value; m_havePropertiesChanged = true; m_isCalculateSizeRequired = true; SetVerticesDirty(); SetLayoutDirty(); }
        }
        [SerializeField]
        protected TextOverflowModes m_overflowMode = TextOverflowModes.Overflow;


        /// <summary>
        /// Indicates if the text exceeds the vertical bounds of its text container.
        /// </summary>
        public bool isTextOverflowing
        {
            get { if (m_firstOverflowCharacterIndex != -1) return true; return false; }
        }


        /// <summary>
        /// The first character which exceeds the vertical bounds of its text container.
        /// </summary>
        public int firstOverflowCharacterIndex
        {
            get { return m_firstOverflowCharacterIndex; }
        }
        [SerializeField]
        protected int m_firstOverflowCharacterIndex = -1;


        /// <summary>
        /// The linked text component used for flowing the text from one text component to another.
        /// </summary>
        public TMP_Text linkedTextComponent
        {
            get { return m_linkedTextComponent; }

            set
            {
                if (m_linkedTextComponent != value)
                {
                    // Release previously linked text component.
                    if (m_linkedTextComponent != null)
                    {
                        m_linkedTextComponent.overflowMode = TextOverflowModes.Overflow;
                        m_linkedTextComponent.linkedTextComponent = null;
                        m_linkedTextComponent.isLinkedTextComponent = false;
                    }

                    m_linkedTextComponent = value;

                    if (m_linkedTextComponent != null)
                        m_linkedTextComponent.isLinkedTextComponent = true;
                }

                m_havePropertiesChanged = true;
                m_isCalculateSizeRequired = true;
                SetVerticesDirty();
                SetLayoutDirty();
            }
        }
        [SerializeField]
        protected TMP_Text m_linkedTextComponent;


        /// <summary>
        /// Indicates whether this text component is linked to another.
        /// </summary>
        public bool isLinkedTextComponent
        {
            get { return m_isLinkedTextComponent; }

            set
            {
                m_isLinkedTextComponent = value;

                if (m_isLinkedTextComponent == false)
                    m_firstVisibleCharacter = 0;

                m_havePropertiesChanged = true;
                m_isCalculateSizeRequired = true;
                SetVerticesDirty();
                SetLayoutDirty();
            }
        }
        [SerializeField]
        protected bool m_isLinkedTextComponent;


        /// <summary>
        /// Property indicating whether the text is Truncated or using Ellipsis.
        /// </summary>
        public bool isTextTruncated { get { return m_isTextTruncated; } }
        [SerializeField]
        protected bool m_isTextTruncated;


        /// <summary>
        /// Determines if kerning is enabled or disabled.
        /// </summary>
        public bool enableKerning
        {
            get { return m_enableKerning; }
            set { if (m_enableKerning == value) return; m_havePropertiesChanged = true; m_isCalculateSizeRequired = true; m_enableKerning = value; SetVerticesDirty(); SetLayoutDirty(); }
        }
        [SerializeField]
        protected bool m_enableKerning;


        /// <summary>
        /// Adds extra padding around each character. This may be necessary when the displayed text is very small to prevent clipping.
        /// </summary>
        public bool extraPadding
        {
            get { return m_enableExtraPadding; }
            set { if (m_enableExtraPadding == value) return; m_havePropertiesChanged = true; m_enableExtraPadding = value; UpdateMeshPadding(); /* m_isCalculateSizeRequired = true;*/ SetVerticesDirty(); /* SetLayoutDirty();*/ }
        }
        [SerializeField]
        protected bool m_enableExtraPadding = false;
        [SerializeField]
        protected bool checkPaddingRequired;


        /// <summary>
        /// Enables or Disables Rich Text Tags
        /// </summary>
        public bool richText
        {
            get { return m_isRichText; }
            set { if (m_isRichText == value) return; m_isRichText = value; m_havePropertiesChanged = true; m_isCalculateSizeRequired = true; m_isInputParsingRequired = true; SetVerticesDirty(); SetLayoutDirty(); }
        }
        [SerializeField]
        protected bool m_isRichText = true; // Used to enable or disable Rich Text.


        /// <summary>
        /// Enables or Disables parsing of CTRL characters in input text.
        /// </summary>
        public bool parseCtrlCharacters
        {
            get { return m_parseCtrlCharacters; }
            set { if (m_parseCtrlCharacters == value) return; m_parseCtrlCharacters = value; m_havePropertiesChanged = true; m_isCalculateSizeRequired = true; m_isInputParsingRequired = true; SetVerticesDirty(); SetLayoutDirty(); }
        }
        [SerializeField]
        protected bool m_parseCtrlCharacters = true;


        /// <summary>
        /// Sets the RenderQueue along with Ztest to force the text to be drawn last and on top of scene elements.
        /// </summary>
        public bool isOverlay
        {
            get { return m_isOverlay; }
            set { if (m_isOverlay == value) return; m_isOverlay = value; SetShaderDepth(); m_havePropertiesChanged = true; SetVerticesDirty(); }
        }
        protected bool m_isOverlay = false;


        /// <summary>
        /// Sets Perspective Correction to Zero for Orthographic Camera mode & 0.875f for Perspective Camera mode.
        /// </summary>
        public bool isOrthographic
        {
            get { return m_isOrthographic; }
            set { if (m_isOrthographic == value) return; m_havePropertiesChanged = true; m_isOrthographic = value; SetVerticesDirty(); }
        }
        [SerializeField]
        protected bool m_isOrthographic = false;


        /// <summary>
        /// Sets the culling on the shaders. Note changing this value will result in an instance of the material.
        /// </summary>
        public bool enableCulling
        {
            get { return m_isCullingEnabled; }
            set { if (m_isCullingEnabled == value) return; m_isCullingEnabled = value; SetCulling(); m_havePropertiesChanged = true; }
        }
        [SerializeField]
        protected bool m_isCullingEnabled = false;

        /// <summary>
        /// Controls whether or not the text object will be culled when using a 2D Rect Mask.
        /// </summary>
        public bool ignoreRectMaskCulling
        {
            get { return m_ignoreRectMaskCulling; }
            set { if (m_ignoreRectMaskCulling == value) return; m_ignoreRectMaskCulling = value; m_havePropertiesChanged = true; }
        }
        [SerializeField]
        protected bool m_ignoreRectMaskCulling;


        /// <summary>
        /// Forces objects that are not visible to get refreshed.
        /// </summary>
        public bool ignoreVisibility
        {
            get { return m_ignoreCulling; }
            set { if (m_ignoreCulling == value) return; m_havePropertiesChanged = true; m_ignoreCulling = value; }
        }
        [SerializeField]
        protected bool m_ignoreCulling = true; // Not implemented yet.


        /// <summary>
        /// Controls how the face and outline textures will be applied to the text object.
        /// </summary>
        public TextureMappingOptions horizontalMapping
        {
            get { return m_horizontalMapping; }
            set { if (m_horizontalMapping == value) return; m_havePropertiesChanged = true; m_horizontalMapping = value; SetVerticesDirty(); }
        }
        [SerializeField]
        protected TextureMappingOptions m_horizontalMapping = TextureMappingOptions.Character;


        /// <summary>
        /// Controls how the face and outline textures will be applied to the text object.
        /// </summary>
        public TextureMappingOptions verticalMapping
        {
            get { return m_verticalMapping; }
            set { if (m_verticalMapping == value) return; m_havePropertiesChanged = true; m_verticalMapping = value; SetVerticesDirty(); }
        }
        [SerializeField]
        protected TextureMappingOptions m_verticalMapping = TextureMappingOptions.Character;


        /// <summary>
        /// Controls the UV Offset for the various texture mapping mode on the text object.
        /// </summary>
        //public Vector2 mappingUvOffset
        //{
        //    get { return m_uvOffset; }
        //    set { if (m_uvOffset == value) return; m_havePropertiesChanged = true; m_uvOffset = value; SetVerticesDirty(); }
        //}
        //[SerializeField]
        //protected Vector2 m_uvOffset = Vector2.zero; // Used to offset UV on Texturing


        /// <summary>
        /// Controls the horizontal offset of the UV of the texture mapping mode for each line of the text object.
        /// </summary>
        public float mappingUvLineOffset
        {
            get { return m_uvLineOffset; }
            set { if (m_uvLineOffset == value) return; m_havePropertiesChanged = true; m_uvLineOffset = value; SetVerticesDirty(); }
        }
        [SerializeField]
        protected float m_uvLineOffset = 0.0f; // Used for UV line offset per line


        /// <summary>
        /// Determines if the Mesh will be rendered.
        /// </summary>
        public TextRenderFlags renderMode
        {
            get { return m_renderMode; }
            set { if (m_renderMode == value) return; m_renderMode = value; m_havePropertiesChanged = true; }
        }
        protected TextRenderFlags m_renderMode = TextRenderFlags.Render;


        /// <summary>
        /// Determines the sorting order of the geometry of the text object.
        /// </summary>
        public VertexSortingOrder geometrySortingOrder
        {
            get { return m_geometrySortingOrder; }

            set { m_geometrySortingOrder = value; m_havePropertiesChanged = true; SetVerticesDirty(); }

        }
        [SerializeField]
        protected VertexSortingOrder m_geometrySortingOrder;

        /// <summary>
        /// Determines if the data structures allocated to contain the geometry of the text object will be reduced in size if the number of characters required to display the text is reduced by more than 256 characters.
        /// This reduction has the benefit of reducing the amount of vertex data being submitted to the graphic device but results in GC when it occurs. 
        /// </summary>
        public bool vertexBufferAutoSizeReduction
        {
            get { return m_VertexBufferAutoSizeReduction; }
            set { m_VertexBufferAutoSizeReduction = value; m_havePropertiesChanged = true; SetVerticesDirty(); }
        }
        [SerializeField]
        protected bool m_VertexBufferAutoSizeReduction = true;

        /// <summary>
        /// The first character which should be made visible in conjunction with the Text Overflow Linked mode.
        /// </summary>
        public int firstVisibleCharacter
        {
            get { return m_firstVisibleCharacter; }
            set { if (m_firstVisibleCharacter == value) return; m_havePropertiesChanged = true; m_firstVisibleCharacter = value; SetVerticesDirty(); }
        }
        [SerializeField]
        protected int m_firstVisibleCharacter;

        /// <summary>
        /// Allows to control how many characters are visible from the input.
        /// </summary>
        public int maxVisibleCharacters
        {
            get { return m_maxVisibleCharacters; }
            set { if (m_maxVisibleCharacters == value) return; m_havePropertiesChanged = true; m_maxVisibleCharacters = value; SetVerticesDirty(); }
        }
        protected int m_maxVisibleCharacters = 99999;


        /// <summary>
        /// Allows to control how many words are visible from the input.
        /// </summary>
        public int maxVisibleWords
        {
            get { return m_maxVisibleWords; }
            set { if (m_maxVisibleWords == value) return; m_havePropertiesChanged = true; m_maxVisibleWords = value; SetVerticesDirty(); }
        }
        protected int m_maxVisibleWords = 99999;


        /// <summary>
        /// Allows control over how many lines of text are displayed.
        /// </summary>
        public int maxVisibleLines
        {
            get { return m_maxVisibleLines; }
            set { if (m_maxVisibleLines == value) return; m_havePropertiesChanged = true; m_isInputParsingRequired = true; m_maxVisibleLines = value; SetVerticesDirty(); }
        }
        protected int m_maxVisibleLines = 99999;


        /// <summary>
        /// Determines if the text's vertical alignment will be adjusted based on visible descender of the text.
        /// </summary>
        public bool useMaxVisibleDescender
        {
            get { return m_useMaxVisibleDescender; }
            set { if (m_useMaxVisibleDescender == value) return; m_havePropertiesChanged = true; m_isInputParsingRequired = true; SetVerticesDirty(); }
        }
        [SerializeField]
        protected bool m_useMaxVisibleDescender = true;


        /// <summary>
        /// Controls which page of text is shown
        /// </summary>
        public int pageToDisplay
        {
            get { return m_pageToDisplay; }
            set { if (m_pageToDisplay == value) return; m_havePropertiesChanged = true; m_pageToDisplay = value; SetVerticesDirty(); }
        }
        [SerializeField]
        protected int m_pageToDisplay = 1;
        protected bool m_isNewPage = false;

        /// <summary>
        /// The margins of the text object.
        /// </summary>
        public virtual Vector4 margin
        {
            get { return m_margin; }
            set { if (m_margin == value) return; m_margin = value; ComputeMarginSize(); m_havePropertiesChanged = true; SetVerticesDirty(); }
        }
        [SerializeField]
        protected Vector4 m_margin = new Vector4(0, 0, 0, 0);
        protected float m_marginLeft;
        protected float m_marginRight;
        protected float m_marginWidth;  // Width of the RectTransform minus left and right margins.
        protected float m_marginHeight; // Height of the RectTransform minus top and bottom margins.
        protected float m_width = -1;


        /// <summary>
        /// Returns data about the text object which includes information about each character, word, line, link, etc.
        /// </summary>
        public TMP_TextInfo textInfo
        {
            get { return m_textInfo; }
        }
        [SerializeField]
        protected TMP_TextInfo m_textInfo; // Class which holds information about the Text object such as characters, lines, mesh data as well as metrics. 

        /// <summary>
        /// Property tracking if any of the text properties have changed. Flag is set before the text is regenerated.
        /// </summary>
        public bool havePropertiesChanged
        {
            get { return m_havePropertiesChanged; }
            set { if (m_havePropertiesChanged == value) return; m_havePropertiesChanged = value; m_isInputParsingRequired = true; SetAllDirty(); }
        }
        //[SerializeField]
        protected bool m_havePropertiesChanged;  // Used to track when properties of the text object have changed.


        /// <summary>
        /// Property to handle legacy animation component.
        /// </summary>
        public bool isUsingLegacyAnimationComponent
        {
            get { return m_isUsingLegacyAnimationComponent; }
            set { m_isUsingLegacyAnimationComponent = value; }
        }
        [SerializeField]
        protected bool m_isUsingLegacyAnimationComponent;


        /// <summary>
        /// Returns are reference to the Transform
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
        protected Transform m_transform;


        /// <summary>
        /// Returns are reference to the RectTransform
        /// </summary>
        public new RectTransform rectTransform
        {
            get
            {
                if (m_rectTransform == null)
                    m_rectTransform = GetComponent<RectTransform>();
                return m_rectTransform;
            }
        }
        protected RectTransform m_rectTransform;


        /// <summary>
        /// Enables control over setting the size of the text container to match the text object.
        /// </summary>
        public virtual bool autoSizeTextContainer
        {
            get;
            set;
        }
        protected bool m_autoSizeTextContainer;


        /// <summary>
        /// The mesh used by the font asset and material assigned to the text object.
        /// </summary>
        public virtual Mesh mesh
        {
            get { return m_mesh; }
        }
        protected Mesh m_mesh;


        /// <summary>
        /// Determines if the geometry of the characters will be quads or volumetric (cubes).
        /// </summary>
        public bool isVolumetricText
        {
            get { return m_isVolumetricText; }
            set { if (m_isVolumetricText == value) return; m_havePropertiesChanged = value; m_textInfo.ResetVertexLayout(value); m_isInputParsingRequired = true; SetVerticesDirty(); SetLayoutDirty(); }
        }
        [SerializeField]
        protected bool m_isVolumetricText;

        /// <summary>
        /// Returns the bounds of the mesh of the text object in world space.
        /// </summary>
        public Bounds bounds
        {
            get
            {
                if (m_mesh == null) return new Bounds();

                return GetCompoundBounds();
            }
        }

        /// <summary>
        /// Returns the bounds of the text of the text object.
        /// </summary>
        public Bounds textBounds
        {
            get
            {
                if (m_textInfo == null) return new Bounds();

                return GetTextBounds();
            }
        }

        // *** Unity Event Handling ***

        //[Serializable]
        //public class TextChangedEvent : UnityEvent { }

        ///// <summary>
        ///// Event delegate triggered when text has changed and been rendered.
        ///// </summary>
        //public TextChangedEvent onTextChanged
        //{
        //    get { return m_OnTextChanged; }
        //    set { m_OnTextChanged = value; }
        //}
        //[SerializeField]
        //private TextChangedEvent m_OnTextChanged = new TextChangedEvent();

        //protected void SendOnTextChanged()
        //{
        //    if (onTextChanged != null)
        //        onTextChanged.Invoke();
        //}


        // *** SPECIAL COMPONENTS ***

        /// <summary>
        /// Component used to control wrapping of text following some arbitrary shape.
        /// </summary>
        //public MarginShaper marginShaper
        //{
        //    get
        //    {
        //        if (m_marginShaper == null) m_marginShaper = GetComponent<MarginShaper>();

        //        return m_marginShaper;
        //    }
        //}
        //[SerializeField]
        //protected MarginShaper m_marginShaper;


        /// <summary>
        /// Component used to control and animate sprites in the text object.
        /// </summary>
        protected TMP_SpriteAnimator spriteAnimator
        {
            get
            {
                if (m_spriteAnimator == null)
                {
                    m_spriteAnimator = GetComponent<TMP_SpriteAnimator>();
                    if (m_spriteAnimator == null) m_spriteAnimator = gameObject.AddComponent<TMP_SpriteAnimator>();
                }

                return m_spriteAnimator;
            }

        }
        [SerializeField]
        protected TMP_SpriteAnimator m_spriteAnimator;


        /// <summary>
        /// 
        /// </summary>
        //public TMP_TextShaper textShaper
        //{
        //    get
        //    {
        //        if (m_textShaper == null)
        //            m_textShaper = GetComponent<TMP_TextShaper>();

        //        return m_textShaper;
        //    }
        //}
        //[SerializeField]
        //protected TMP_TextShaper m_textShaper;

        // *** PROPERTIES RELATED TO UNITY LAYOUT SYSTEM ***
        /// <summary>
        /// 
        /// </summary>
        public float flexibleHeight { get { return m_flexibleHeight; } }
        protected float m_flexibleHeight = -1f;

        /// <summary>
        /// 
        /// </summary>
        public float flexibleWidth { get { return m_flexibleWidth; } }
        protected float m_flexibleWidth = -1f;

        /// <summary>
        /// 
        /// </summary>
        public float minWidth { get { return m_minWidth; } }
        protected float m_minWidth;

        /// <summary>
        /// 
        /// </summary>
        public float minHeight { get { return m_minHeight; } }
        protected float m_minHeight;

        /// <summary>
        /// 
        /// </summary>
        public float maxWidth { get { return m_maxWidth; } }
        protected float m_maxWidth;

        /// <summary>
        /// 
        /// </summary>
        public float maxHeight { get { return m_maxHeight; } }
        protected float m_maxHeight;

        /// <summary>
        /// 
        /// </summary>
        protected LayoutElement layoutElement
        {
            get
            {
                if (m_LayoutElement == null)
                {
                    m_LayoutElement = GetComponent<LayoutElement>();
                }

                return m_LayoutElement;
            }
        }
        protected LayoutElement m_LayoutElement;

        /// <summary>
        /// Computed preferred width of the text object.
        /// </summary>
        public virtual float preferredWidth { get { if (!m_isPreferredWidthDirty) return m_preferredWidth; m_preferredWidth = GetPreferredWidth(); return m_preferredWidth; } }
        protected float m_preferredWidth;
        protected float m_renderedWidth;
        protected bool m_isPreferredWidthDirty;

        /// <summary>
        /// Computed preferred height of the text object.
        /// </summary>
        public virtual float preferredHeight { get { if (!m_isPreferredHeightDirty) return m_preferredHeight; m_preferredHeight = GetPreferredHeight(); return m_preferredHeight; } }
        protected float m_preferredHeight;
        protected float m_renderedHeight;
        protected bool m_isPreferredHeightDirty;

        protected bool m_isCalculatingPreferredValues;
        private int m_recursiveCount;

        /// <summary>
        /// Compute the rendered width of the text object.
        /// </summary>
        public virtual float renderedWidth { get { return GetRenderedWidth(); } }


        /// <summary>
        /// Compute the rendered height of the text object.
        /// </summary>
        public virtual float renderedHeight { get { return GetRenderedHeight(); } }


        /// <summary>
        /// 
        /// </summary>
        public int layoutPriority { get { return m_layoutPriority; } }
        protected int m_layoutPriority = 0;

        protected bool m_isCalculateSizeRequired = false;
        protected bool m_isLayoutDirty;

        protected bool m_verticesAlreadyDirty;
        protected bool m_layoutAlreadyDirty;

        protected bool m_isAwake;
        internal bool m_isWaitingOnResourceLoad;

        internal bool m_isInputParsingRequired = false; // Used to determine if the input text needs to be re-parsed.

        // Protected Fields
        internal enum TextInputSources { Text = 0, SetText = 1, SetCharArray = 2, String = 3 };
        //[SerializeField]
        internal TextInputSources m_inputSource;
        protected string old_text; // Used by SetText to determine if the text has changed.
        //protected float old_arg0, old_arg1, old_arg2; // Used by SetText to determine if the args have changed.


        protected float m_fontScale; // Scaling of the font based on Atlas true Font Size and Rendered Font Size.  
        protected float m_fontScaleMultiplier; // Used for handling of superscript and subscript.

        protected char[] m_htmlTag = new char[128]; // Maximum length of rich text tag. This is preallocated to avoid GC.
        protected RichTextTagAttribute[] m_xmlAttribute = new RichTextTagAttribute[8];

        protected float[] m_attributeParameterValues = new float[16];

        protected float tag_LineIndent = 0;
        protected float tag_Indent = 0;
        protected TMP_RichTextTagStack<float> m_indentStack = new TMP_RichTextTagStack<float>(new float[16]);
        protected bool tag_NoParsing;
        //protected TMP_LinkInfo tag_LinkInfo = new TMP_LinkInfo();

        protected bool m_isParsingText;
        protected Matrix4x4 m_FXMatrix;
        protected bool m_isFXMatrixSet;


        protected UnicodeChar[] m_TextParsingBuffer; // This array holds the characters to be processed by GenerateMesh();

        protected struct UnicodeChar
        {
            public int unicode;
            public int stringIndex;
            public int length;
        }
        //protected UnicodeChar[] m_InternalParsingBuffer;

        private TMP_CharacterInfo[] m_internalCharacterInfo; // Used by functions to calculate preferred values.
        protected char[] m_input_CharArray = new char[256]; // This array hold the characters from the SetText();
        private int m_charArray_Length = 0;
        protected int m_totalCharacterCount;

        // Structures used to save the state of the text layout in conjunction with line breaking / word wrapping.
        protected WordWrapState m_SavedWordWrapState = new WordWrapState();
        protected WordWrapState m_SavedLineState = new WordWrapState();
		//protected WordWrapState m_SavedAlignment = new WordWrapState ();


        // Fields whose state is saved in conjunction with text parsing and word wrapping.
        protected int m_characterCount;
        //protected int m_visibleCharacterCount;
        //protected int m_visibleSpriteCount;
        protected int m_firstCharacterOfLine;
        protected int m_firstVisibleCharacterOfLine;
        protected int m_lastCharacterOfLine;
        protected int m_lastVisibleCharacterOfLine;
        protected int m_lineNumber;
        protected int m_lineVisibleCharacterCount;
        protected int m_pageNumber;
        protected float m_maxAscender;
        protected float m_maxCapHeight;
        protected float m_maxDescender;
        protected float m_maxLineAscender;
        protected float m_maxLineDescender;
        protected float m_startOfLineAscender;
        //protected float m_maxFontScale;
        protected float m_lineOffset;
        protected Extents m_meshExtents;


        // Fields used for vertex colors
        protected Color32 m_htmlColor = new Color(255, 255, 255, 128);
        protected TMP_RichTextTagStack<Color32> m_colorStack = new TMP_RichTextTagStack<Color32>(new Color32[16]);
        protected TMP_RichTextTagStack<Color32> m_underlineColorStack = new TMP_RichTextTagStack<Color32>(new Color32[16]);
        protected TMP_RichTextTagStack<Color32> m_strikethroughColorStack = new TMP_RichTextTagStack<Color32>(new Color32[16]);
        protected TMP_RichTextTagStack<Color32> m_highlightColorStack = new TMP_RichTextTagStack<Color32>(new Color32[16]);

        protected TMP_ColorGradient m_colorGradientPreset;
        protected TMP_RichTextTagStack<TMP_ColorGradient> m_colorGradientStack = new TMP_RichTextTagStack<TMP_ColorGradient>(new TMP_ColorGradient[16]);

        protected float m_tabSpacing = 0;
        protected float m_spacing = 0;


        //protected bool IsRectTransformDriven;


        // STYLE TAGS
        protected TMP_RichTextTagStack<int> m_styleStack = new TMP_RichTextTagStack<int>(new int[16]);
        protected TMP_RichTextTagStack<int> m_actionStack = new TMP_RichTextTagStack<int>(new int[16]);

        protected float m_padding = 0;
        protected float m_baselineOffset; // Used for superscript and subscript.
        protected TMP_RichTextTagStack<float> m_baselineOffsetStack = new TMP_RichTextTagStack<float>(new float[16]);
        protected float m_xAdvance; // Tracks x advancement from character to character.

        protected TMP_TextElementType m_textElementType;
        protected TMP_TextElement m_cached_TextElement; // Glyph / Character information is cached into this variable which is faster than having to fetch from the Dictionary multiple times.
        protected TMP_Character m_cached_Underline_Character; // Same as above but for the underline character which is used for Underline.
        protected TMP_Character m_cached_Ellipsis_Character;

        protected TMP_SpriteAsset m_defaultSpriteAsset;
        protected TMP_SpriteAsset m_currentSpriteAsset;
        protected int m_spriteCount = 0;
        protected int m_spriteIndex;
        protected int m_spriteAnimationID;
        //protected TMP_XmlTagStack<int> m_spriteAnimationStack = new TMP_XmlTagStack<int>(new int[16]);


        /// <summary>
        /// Method which derived classes need to override to load Font Assets.
        /// </summary>
        protected virtual void LoadFontAsset() { }

        /// <summary>
        /// Function called internally when a new shared material is assigned via the fontSharedMaterial property.
        /// </summary>
        /// <param name="mat"></param>
        protected virtual void SetSharedMaterial(Material mat) { }

        /// <summary>
        /// Function called internally when a new material is assigned via the fontMaterial property.
        /// </summary>
        protected virtual Material GetMaterial(Material mat) { return null; }

        /// <summary>
        /// Function called internally when assigning a new base material.
        /// </summary>
        /// <param name="mat"></param>
        protected virtual void SetFontBaseMaterial(Material mat) { }

        /// <summary>
        /// Method which returns an array containing the materials used by the text object.
        /// </summary>
        /// <returns></returns>
        protected virtual Material[] GetSharedMaterials() { return null; }

        /// <summary>
        /// 
        /// </summary>
        protected virtual void SetSharedMaterials(Material[] materials) { }

        /// <summary>
        /// Method returning instances of the materials used by the text object.
        /// </summary>
        /// <returns></returns>
        protected virtual Material[] GetMaterials(Material[] mats) { return null; }

        /// <summary>
        /// Method to set the materials of the text and sub text objects.
        /// </summary>
        /// <param name="mats"></param>
        //protected virtual void SetMaterials (Material[] mats) { }

        /// <summary>
        /// Function used to create an instance of the material
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        protected virtual Material CreateMaterialInstance(Material source)
        {
            Material mat = new Material(source);
            mat.shaderKeywords = source.shaderKeywords;
            mat.name += " (Instance)";

            return mat;
        }

        protected void SetVertexColorGradient(TMP_ColorGradient gradient)
        {
            if (gradient == null) return;

            m_fontColorGradient.bottomLeft = gradient.bottomLeft;
            m_fontColorGradient.bottomRight = gradient.bottomRight;
            m_fontColorGradient.topLeft = gradient.topLeft;
            m_fontColorGradient.topRight = gradient.topRight;

            SetVerticesDirty();
        }

        /// <summary>
        /// Function to control the sorting of the geometry of the text object.
        /// </summary>
        protected void SetTextSortingOrder(VertexSortingOrder order)
        {
            
        }

        /// <summary>
        /// Function to sort the geometry of the text object in accordance to the provided order.
        /// </summary>
        /// <param name="order"></param>
        protected void SetTextSortingOrder(int[] order)
        {

        }

        /// <summary>
        /// Function called internally to set the face color of the material. This will results in an instance of the material.
        /// </summary>
        /// <param name="color"></param>
        protected virtual void SetFaceColor(Color32 color) { }

        /// <summary>
        /// Function called internally to set the outline color of the material. This will results in an instance of the material.
        /// </summary>
        /// <param name="color"></param>
        protected virtual void SetOutlineColor(Color32 color) { }

        /// <summary>
        /// Function called internally to set the outline thickness property of the material. This will results in an instance of the material.
        /// </summary>
        /// <param name="thickness"></param>
        protected virtual void SetOutlineThickness(float thickness) { }

        /// <summary>
        /// Set the Render Queue and ZTest mode on the current material
        /// </summary>
        protected virtual void SetShaderDepth() { }

        /// <summary>
        /// Set the culling mode on the material.
        /// </summary>
        protected virtual void SetCulling() { }

        /// <summary>
        /// Get the padding value for the currently assigned material
        /// </summary>
        /// <returns></returns>
        protected virtual float GetPaddingForMaterial() { return 0; }


        /// <summary>
        /// Get the padding value for the given material
        /// </summary>
        /// <returns></returns>
        protected virtual float GetPaddingForMaterial(Material mat) { return 0; }


        /// <summary>
        /// Method to return the local corners of the Text Container or RectTransform.
        /// </summary>
        /// <returns></returns>
        protected virtual Vector3[] GetTextContainerLocalCorners() { return null; }


        // PUBLIC FUNCTIONS
        protected bool m_ignoreActiveState;
        /// <summary>
        /// Function to force the regeneration of the text object.
        /// </summary>
        public virtual void ForceMeshUpdate() { }


        /// <summary>
        /// Method used for resetting vertex layout when switching to and from Volumetric Text mode.
        /// </summary>
        /// <param name="updateMesh"></param>
        //protected virtual void ResetVertexLayout() { }


        /// <summary>
        /// Function to force the regeneration of the text object.
        /// </summary>
        /// <param name="ignoreActiveState">If set to true, the text object will be regenerated regardless of is active state.</param>
        public virtual void ForceMeshUpdate(bool ignoreActiveState) { }


        /// <summary>
        /// Internal function used by the Text Input Field to populate TMP_TextInfo data. 
        /// </summary>
        internal void SetTextInternal(string text)
        {
            m_text = text;
            m_renderMode = TextRenderFlags.DontRender;
            m_isInputParsingRequired = true;
            ForceMeshUpdate();
            m_renderMode = TextRenderFlags.Render;
        }

        /// <summary>
        /// Function to force the regeneration of the text object.
        /// </summary>
        /// <param name="flags"> Flags to control which portions of the geometry gets uploaded.</param>
        //public virtual void ForceMeshUpdate(TMP_VertexDataUpdateFlags flags) { }


        /// <summary>
        /// Function to update the geometry of the main and sub text objects.
        /// </summary>
        /// <param name="mesh"></param>
        /// <param name="index"></param>
        public virtual void UpdateGeometry(Mesh mesh, int index) { }


        /// <summary>
        /// Function to push the updated vertex data into the mesh and renderer.
        /// </summary>
        public virtual void UpdateVertexData(TMP_VertexDataUpdateFlags flags) { }


        /// <summary>
        /// Function to push the updated vertex data into the mesh and renderer.
        /// </summary>
        public virtual void UpdateVertexData() { }


        /// <summary>
        /// Function to push a new set of vertices to the mesh.
        /// </summary>
        /// <param name="vertices"></param>
        public virtual void SetVertices(Vector3[] vertices) { }


        /// <summary>
        /// Function to be used to force recomputing of character padding when Shader / Material properties have been changed via script.
        /// </summary>
        public virtual void UpdateMeshPadding() { }


        /// <summary>
        /// 
        /// </summary>
        //public virtual new void UpdateGeometry() { }


        /// <summary>
        /// Tweens the CanvasRenderer color associated with this Graphic.
        /// </summary>
        /// <param name="targetColor">Target color.</param>
        /// <param name="duration">Tween duration.</param>
        /// <param name="ignoreTimeScale">Should ignore Time.scale?</param>
        /// <param name="useAlpha">Should also Tween the alpha channel?</param>
        public override void CrossFadeColor(Color targetColor, float duration, bool ignoreTimeScale, bool useAlpha)
        {
            base.CrossFadeColor(targetColor, duration, ignoreTimeScale, useAlpha);
            InternalCrossFadeColor(targetColor, duration, ignoreTimeScale, useAlpha);
        }


        /// <summary>
        /// Tweens the alpha of the CanvasRenderer color associated with this Graphic.
        /// </summary>
        /// <param name="alpha">Target alpha.</param>
        /// <param name="duration">Duration of the tween in seconds.</param>
        /// <param name="ignoreTimeScale">Should ignore Time.scale?</param>
        public override void CrossFadeAlpha(float alpha, float duration, bool ignoreTimeScale)
        {
            base.CrossFadeAlpha(alpha, duration, ignoreTimeScale);
            InternalCrossFadeAlpha(alpha, duration, ignoreTimeScale);
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="targetColor"></param>
        /// <param name="duration"></param>
        /// <param name="ignoreTimeScale"></param>
        /// <param name="useAlpha"></param>
        /// <param name="useRGB"></param>
        protected virtual void InternalCrossFadeColor(Color targetColor, float duration, bool ignoreTimeScale, bool useAlpha) { }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="alpha"></param>
        /// <param name="duration"></param>
        /// <param name="ignoreTimeScale"></param>
        protected virtual void InternalCrossFadeAlpha(float alpha, float duration, bool ignoreTimeScale) { }


        /// <summary>
        /// Method to parse the input text based on its source
        /// </summary>
        protected void ParseInputText()
        {
            //Debug.Log("Re-parsing Text.");
            ////Profiler.BeginSample("ParseInputText()");

            m_isInputParsingRequired = false;

            switch (m_inputSource)
            {
                case TextInputSources.String:
                case TextInputSources.Text:
                    StringToCharArray(m_text, ref m_TextParsingBuffer);
                    break;
                case TextInputSources.SetText:
                    SetTextArrayToCharArray(m_input_CharArray, ref m_TextParsingBuffer);
                    break;
                case TextInputSources.SetCharArray:
                    break;
            }

            SetArraySizes(m_TextParsingBuffer);
            ////Profiler.EndSample();
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="text"></param>
        public void SetText(string text)
        {
            SetText(text, true);
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="text"></param>
        public void SetText(string text, bool syncTextInputBox)
        {
            //if (text == old_text) return;

            //old_text = text;

            m_inputSource = TextInputSources.SetCharArray;

            StringToCharArray(text, ref m_TextParsingBuffer);

            #if UNITY_EDITOR
            // Set the text in the Text Input Box in the Unity Editor only.
            // TODO: Could revise to convert to string literal
            if (syncTextInputBox)
                m_text = text;
            #endif

            m_isInputParsingRequired = true;
            m_havePropertiesChanged = true;
            m_isCalculateSizeRequired = true;

            SetVerticesDirty();
            SetLayoutDirty();
        }


        /// <summary>
        /// <para>Formatted string containing a pattern and a value representing the text to be rendered.</para>
        /// <para>ex. TextMeshPro.SetText ("Number is {0:1}.", 5.56f);</para>
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="text">String containing the pattern."</param>
        /// <param name="arg0">Value is a float.</param>
        public void SetText(string text, float arg0)
        {
            SetText(text, arg0, 255, 255);
        }

        /// <summary>
        /// <para>Formatted string containing a pattern and a value representing the text to be rendered.</para>
        /// <para>ex. TextMeshPro.SetText ("First number is {0} and second is {1:2}.", 10, 5.756f);</para>
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="text">String containing the pattern."</param>
        /// <param name="arg0">Value is a float.</param>
        /// <param name="arg1">Value is a float.</param>
        public void SetText(string text, float arg0, float arg1)
        {
            SetText(text, arg0, arg1, 255);
        }

        /// <summary>
        /// <para>Formatted string containing a pattern and a value representing the text to be rendered.</para>
        /// <para>ex. TextMeshPro.SetText ("A = {0}, B = {1} and C = {2}.", 2, 5, 7);</para>
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="text">String containing the pattern."</param>
        /// <param name="arg0">Value is a float.</param>
        /// <param name="arg1">Value is a float.</param>
        /// <param name="arg2">Value is a float.</param>
        public void SetText(string text, float arg0, float arg1, float arg2)
        {
            int decimalPrecision = 0;
            int index = 0;

            for (int i = 0; i < text.Length; i++)
            {
                char c = text[i];

                if (c == 123) // '{'
                {
                    // Check if user is requesting some decimal precision. Format is {0:2}
                    if (text[i + 2] == 58) // ':'
                    {
                        decimalPrecision = text[i + 3] - 48;
                    }

                    switch (text[i + 1] - 48)
                    {
                        case 0: // 1st Arg
                            AddFloatToCharArray(arg0, ref index, decimalPrecision);
                            break;
                        case 1: // 2nd Arg
                            AddFloatToCharArray(arg1, ref index, decimalPrecision);
                            break;
                        case 2: // 3rd Arg
                            AddFloatToCharArray(arg2, ref index, decimalPrecision);
                            break;
                    }

                    if (text[i + 2] == 58)
                        i += 4;
                    else
                        i += 2;

                    continue;
                }
                m_input_CharArray[index] = c;
                index += 1;
            }

            m_input_CharArray[index] = (char)0;
            m_charArray_Length = index; // Set the length to where this '0' termination is.

            #if UNITY_EDITOR
            // Create new string to be displayed in the Input Text Box of the Editor Panel.
            m_text = new string(m_input_CharArray, 0, index);
            #endif

            m_inputSource = TextInputSources.SetText;
            m_isInputParsingRequired = true;
            m_havePropertiesChanged = true;
            m_isCalculateSizeRequired = true;

            SetVerticesDirty();
            SetLayoutDirty();
        }


        /// <summary>
        /// Set the text using a StringBuilder.
        /// </summary>
        /// <description>
        /// Using a StringBuilder instead of concatenating strings prevents memory pollution with temporary objects.
        /// </description>
        /// <param name="text">StringBuilder with text to display.</param>
        public void SetText(StringBuilder text)
        {
            m_inputSource = TextInputSources.SetCharArray;

            #if UNITY_EDITOR
            // Set the text in the Text Input Box in the Unity Editor only.
            m_text = text.ToString();
            #endif

            StringBuilderToIntArray(text, ref m_TextParsingBuffer);

            m_isInputParsingRequired = true;
            m_havePropertiesChanged = true;
            m_isCalculateSizeRequired = true;

            SetVerticesDirty();
            SetLayoutDirty();
        }


        /// <summary>
        /// Character array containing the text to be displayed.
        /// </summary>
        /// <param name="sourceText"></param>
        public void SetCharArray(char[] sourceText)
        {
            // Initialize internal character buffer if necessary
            if (m_TextParsingBuffer == null) m_TextParsingBuffer = new UnicodeChar[8];

            #if UNITY_EDITOR
            // Create new string to be displayed in the Input Text Box of the Editor Panel.
            if (sourceText == null || sourceText.Length == 0)
                m_text = string.Empty;
            else
                m_text = new string(sourceText);
            #endif

            // Clear the Style stack.
            m_styleStack.Clear();

            int writeIndex = 0;

            for (int i = 0; sourceText != null && i < sourceText.Length; i++)
            {
                if (sourceText[i] == 92 && i < sourceText.Length - 1)
                {
                    switch ((int)sourceText[i + 1])
                    {
                        case 110: // \n LineFeed
                            if (writeIndex == m_TextParsingBuffer.Length) ResizeInternalArray(ref m_TextParsingBuffer);

                            m_TextParsingBuffer[writeIndex].unicode = 10;
                            i += 1;
                            writeIndex += 1;
                            continue;
                        case 114: // \r LineFeed
                            if (writeIndex == m_TextParsingBuffer.Length) ResizeInternalArray(ref m_TextParsingBuffer);

                            m_TextParsingBuffer[writeIndex].unicode = 13;
                            i += 1;
                            writeIndex += 1;
                            continue;
                        case 116: // \t Tab
                            if (writeIndex == m_TextParsingBuffer.Length) ResizeInternalArray(ref m_TextParsingBuffer);

                            m_TextParsingBuffer[writeIndex].unicode = 9;
                            i += 1;
                            writeIndex += 1;
                            continue;
                    }
                }

                // Handle inline replacement of <stlye> and <br> tags.
                if (sourceText[i] == 60)
                {
                    if (IsTagName(ref sourceText, "<BR>", i))
                    {
                        if (writeIndex == m_TextParsingBuffer.Length) ResizeInternalArray(ref m_TextParsingBuffer);

                        m_TextParsingBuffer[writeIndex].unicode = 10; ;
                        writeIndex += 1;
                        i += 3;

                        continue;
                    }
                    else if (IsTagName(ref sourceText, "<STYLE=", i))
                    {
                        int srcOffset;

                        if (ReplaceOpeningStyleTag(ref sourceText, i, out srcOffset, ref m_TextParsingBuffer, ref writeIndex))
                        {
                            i = srcOffset;
                            continue;
                        }
                    }
                    else if (IsTagName(ref sourceText, "</STYLE>", i))
                    {
                        ReplaceClosingStyleTag(ref sourceText, i, ref m_TextParsingBuffer, ref writeIndex);

                        // Strip </style> even if style is invalid.
                        i += 7;
                        continue;
                    }
                }

                if (writeIndex == m_TextParsingBuffer.Length) ResizeInternalArray(ref m_TextParsingBuffer);

                m_TextParsingBuffer[writeIndex].unicode = sourceText[i];
                writeIndex += 1;
            }

            if (writeIndex == m_TextParsingBuffer.Length) ResizeInternalArray(ref m_TextParsingBuffer);

            m_TextParsingBuffer[writeIndex].unicode = 0;

            m_inputSource = TextInputSources.SetCharArray;
            m_isInputParsingRequired = true;
            m_havePropertiesChanged = true;
            m_isCalculateSizeRequired = true;

            SetVerticesDirty();
            SetLayoutDirty();
        }


        /// <summary>
        /// Character array containing the text to be displayed.
        /// </summary>
        /// <param name="sourceText"></param>
        public void SetCharArray(char[] sourceText, int start, int length)
        {
            // Initialize internal character buffer if necessary
            if (m_TextParsingBuffer == null) m_TextParsingBuffer = new UnicodeChar[8];

            #if UNITY_EDITOR
            // Create new string to be displayed in the Input Text Box of the Editor Panel.
            if (sourceText == null || sourceText.Length == 0 || length == 0)
            {
                m_text = string.Empty;
                start = 0;
                length = 0;
            }
            else
            {
                // TODO: Add potential range check on start + length relative to array size.
                m_text = new string(sourceText, start, length);
            }
            #endif

            // Clear the Style stack.
            m_styleStack.Clear();

            int writeIndex = 0;

            int i = start;
            int end = start + length;
            for (; i < end; i++)
            {
                if (sourceText[i] == 92 && i < length - 1)
                {
                    switch ((int)sourceText[i + 1])
                    {
                        case 110: // \n LineFeed
                            if (writeIndex == m_TextParsingBuffer.Length) ResizeInternalArray(ref m_TextParsingBuffer);

                            m_TextParsingBuffer[writeIndex].unicode = 10;
                            i += 1;
                            writeIndex += 1;
                            continue;
                        case 114: // \r LineFeed
                            if (writeIndex == m_TextParsingBuffer.Length) ResizeInternalArray(ref m_TextParsingBuffer);

                            m_TextParsingBuffer[writeIndex].unicode = 13;
                            i += 1;
                            writeIndex += 1;
                            continue;
                        case 116: // \t Tab
                            if (writeIndex == m_TextParsingBuffer.Length) ResizeInternalArray(ref m_TextParsingBuffer);

                            m_TextParsingBuffer[writeIndex].unicode = 9;
                            i += 1;
                            writeIndex += 1;
                            continue;
                    }
                }

                // Handle inline replacement of <stlye> and <br> tags.
                if (sourceText[i] == 60)
                {
                    if (IsTagName(ref sourceText, "<BR>", i))
                    {
                        if (writeIndex == m_TextParsingBuffer.Length) ResizeInternalArray(ref m_TextParsingBuffer);

                        m_TextParsingBuffer[writeIndex].unicode = 10;
                        writeIndex += 1;
                        i += 3;

                        continue;
                    }
                    else if (IsTagName(ref sourceText, "<STYLE=", i))
                    {
                        int srcOffset;

                        if (ReplaceOpeningStyleTag(ref sourceText, i, out srcOffset, ref m_TextParsingBuffer, ref writeIndex))
                        {
                            i = srcOffset;
                            continue;
                        }
                    }
                    else if (IsTagName(ref sourceText, "</STYLE>", i))
                    {
                        ReplaceClosingStyleTag(ref sourceText, i, ref m_TextParsingBuffer, ref writeIndex);

                        // Strip </style> even if style is invalid.
                        i += 7;
                        continue;
                    }
                }

                if (writeIndex == m_TextParsingBuffer.Length) ResizeInternalArray(ref m_TextParsingBuffer);

                m_TextParsingBuffer[writeIndex].unicode = sourceText[i];
                writeIndex += 1;
            }

            if (writeIndex == m_TextParsingBuffer.Length) ResizeInternalArray(ref m_TextParsingBuffer);

            m_TextParsingBuffer[writeIndex].unicode = 0;

            m_inputSource = TextInputSources.SetCharArray;
            m_havePropertiesChanged = true;
            m_isInputParsingRequired = true;
            m_isCalculateSizeRequired = true;

            SetVerticesDirty();
            SetLayoutDirty();
        }


        /// <summary>
        /// Character array containing the text to be displayed.
        /// </summary>
        /// <param name="sourceText"></param>
        public void SetCharArray(int[] sourceText, int start, int length)
        {
            // Initialize internal character buffer if necessary
            if (m_TextParsingBuffer == null) m_TextParsingBuffer = new UnicodeChar[8];

            #if UNITY_EDITOR
            // Create new string to be displayed in the Input Text Box of the Editor Panel.
            if (sourceText == null || sourceText.Length == 0 || length == 0)
            {
                m_text = string.Empty;
                start = 0;
                length = 0;
            }
            else
            {
                m_text = sourceText.IntToString(start, length);
            }
            #endif

            // Clear the Style stack.
            m_styleStack.Clear();

            int writeIndex = 0;

            int end = start + length;
            for (int i = start; i < end && i < sourceText.Length; i++)
            {
                if (sourceText[i] == 92 && i < length - 1)
                {
                    switch ((int)sourceText[i + 1])
                    {
                        case 110: // \n LineFeed
                            if (writeIndex == m_TextParsingBuffer.Length) ResizeInternalArray(ref m_TextParsingBuffer);

                            m_TextParsingBuffer[writeIndex].unicode = 10;
                            i += 1;
                            writeIndex += 1;
                            continue;
                        case 114: // \r LineFeed
                            if (writeIndex == m_TextParsingBuffer.Length) ResizeInternalArray(ref m_TextParsingBuffer);

                            m_TextParsingBuffer[writeIndex].unicode = 13;
                            i += 1;
                            writeIndex += 1;
                            continue;
                        case 116: // \t Tab
                            if (writeIndex == m_TextParsingBuffer.Length) ResizeInternalArray(ref m_TextParsingBuffer);

                            m_TextParsingBuffer[writeIndex].unicode = 9;
                            i += 1;
                            writeIndex += 1;
                            continue;
                    }
                }

                // Handle inline replacement of <stlye> and <br> tags.
                if (sourceText[i] == 60)
                {
                    if (IsTagName(ref sourceText, "<BR>", i))
                    {
                        if (writeIndex == m_TextParsingBuffer.Length) ResizeInternalArray(ref m_TextParsingBuffer);

                        m_TextParsingBuffer[writeIndex].unicode = 10;
                        writeIndex += 1;
                        i += 3;

                        continue;
                    }
                    else if (IsTagName(ref sourceText, "<STYLE=", i))
                    {
                        int srcOffset;

                        if (ReplaceOpeningStyleTag(ref sourceText, i, out srcOffset, ref m_TextParsingBuffer, ref writeIndex))
                        {
                            i = srcOffset;
                            continue;
                        }
                    }
                    else if (IsTagName(ref sourceText, "</STYLE>", i))
                    {
                        ReplaceClosingStyleTag(ref sourceText, i, ref m_TextParsingBuffer, ref writeIndex);

                        // Strip </style> even if style is invalid.
                        i += 7;
                        continue;
                    }
                }

                if (writeIndex == m_TextParsingBuffer.Length) ResizeInternalArray(ref m_TextParsingBuffer);

                m_TextParsingBuffer[writeIndex].unicode = sourceText[i];
                writeIndex += 1;
            }

            if (writeIndex == m_TextParsingBuffer.Length) ResizeInternalArray(ref m_TextParsingBuffer);

            m_TextParsingBuffer[writeIndex].unicode = 0;

            m_inputSource = TextInputSources.SetCharArray;
            m_havePropertiesChanged = true;
            m_isInputParsingRequired = true;
            m_isCalculateSizeRequired = true;

            SetVerticesDirty();
            SetLayoutDirty();
        }


        /// <summary>
        /// Copies Content of formatted SetText() to charBuffer.
        /// </summary>
        /// <param name="sourceText"></param>
        /// <param name="charBuffer"></param>
        protected void SetTextArrayToCharArray(char[] sourceText, ref UnicodeChar[] charBuffer)
        {
            //Debug.Log("SetText Array to Char called.");
            if (sourceText == null || m_charArray_Length == 0)
                return;

            if (charBuffer == null) charBuffer = new UnicodeChar[8];

            // Clear the Style stack.
            m_styleStack.Clear();

            int writeIndex = 0;

            for (int i = 0; i < m_charArray_Length; i++)
            {
                // Handle UTF-32 in the input text (string).
                if (char.IsHighSurrogate(sourceText[i]) && char.IsLowSurrogate(sourceText[i + 1]))
                {
                    if (writeIndex == charBuffer.Length) ResizeInternalArray(ref charBuffer);

                    charBuffer[writeIndex].unicode = char.ConvertToUtf32(sourceText[i], sourceText[i + 1]);
                    i += 1;
                    writeIndex += 1;
                    continue;
                }

                // Handle inline replacement of <stlye> and <br> tags.
                if (sourceText[i] == 60)
                {
                    if (IsTagName(ref sourceText, "<BR>", i))
                    {
                        if (writeIndex == charBuffer.Length) ResizeInternalArray(ref charBuffer);

                        charBuffer[writeIndex].unicode = 10;
                        writeIndex += 1;
                        i += 3;

                        continue;
                    }
                    else if (IsTagName(ref sourceText, "<STYLE=", i))
                    {
                        int srcOffset;

                        if (ReplaceOpeningStyleTag(ref sourceText, i, out srcOffset, ref charBuffer, ref writeIndex))
                        {
                            i = srcOffset;
                            continue;
                        }
                    }
                    else if (IsTagName(ref sourceText, "</STYLE>", i))
                    {
                        ReplaceClosingStyleTag(ref sourceText, i, ref charBuffer, ref writeIndex);

                        // Strip </style> even if style is invalid.
                        i += 7;
                        continue;
                    }
                }

                if (writeIndex == charBuffer.Length) ResizeInternalArray(ref charBuffer);

                charBuffer[writeIndex].unicode = sourceText[i];
                writeIndex += 1;
            }

            if (writeIndex == charBuffer.Length) ResizeInternalArray(ref charBuffer);

            charBuffer[writeIndex].unicode = 0;
        }


        /// <summary>
        /// Method to store the content of a string into an integer array.
        /// </summary>
        /// <param name="sourceText"></param>
        /// <param name="charBuffer"></param>
        protected void StringToCharArray(string sourceText, ref UnicodeChar[] charBuffer)
        {
            if (sourceText == null)
            {
                charBuffer[0].unicode = 0;
                return;
            }

            if (charBuffer == null) charBuffer = new UnicodeChar[8];

            // Clear the Style stack.
            m_styleStack.SetDefault(0);

            int writeIndex = 0;

            for (int i = 0; i < sourceText.Length; i++)
            {
                if (m_inputSource == TextInputSources.Text && sourceText[i] == 92 && sourceText.Length > i + 1)
                {
                    switch ((int)sourceText[i + 1])
                    {
                        case 85: // \U00000000 for UTF-32 Unicode
                            if (sourceText.Length > i + 9)
                            {
                                if (writeIndex == charBuffer.Length) ResizeInternalArray(ref charBuffer);

                                charBuffer[writeIndex].unicode = GetUTF32(sourceText, i + 2);
                                charBuffer[writeIndex].stringIndex = i;
                                charBuffer[writeIndex].length = 10;

                                i += 9;
                                writeIndex += 1;
                                continue;
                            }
                            break;
                        case 92: // \ escape
                            if (!m_parseCtrlCharacters) break;

                            if (sourceText.Length <= i + 2) break;

                            if (writeIndex + 2 > charBuffer.Length) ResizeInternalArray(ref charBuffer);

                            charBuffer[writeIndex].unicode = sourceText[i + 1];
                            charBuffer[writeIndex + 1].unicode = sourceText[i + 2];
                            i += 2;
                            writeIndex += 2;
                            continue;
                        case 110: // \n LineFeed
                            if (!m_parseCtrlCharacters) break;

                            if (writeIndex == charBuffer.Length) ResizeInternalArray(ref charBuffer);

                            charBuffer[writeIndex].unicode = 10;
                            charBuffer[writeIndex].stringIndex = i;
                            charBuffer[writeIndex].length = 1;

                            i += 1;
                            writeIndex += 1;
                            continue;
                        case 114: // \r
                            if (!m_parseCtrlCharacters) break;

                            if (writeIndex == charBuffer.Length) ResizeInternalArray(ref charBuffer);

                            charBuffer[writeIndex].unicode = 13;
                            charBuffer[writeIndex].stringIndex = i;
                            charBuffer[writeIndex].length = 1;

                            i += 1;
                            writeIndex += 1;
                            continue;
                        case 116: // \t Tab
                            if (!m_parseCtrlCharacters) break;

                            if (writeIndex == charBuffer.Length) ResizeInternalArray(ref charBuffer);

                            charBuffer[writeIndex].unicode = 9;
                            charBuffer[writeIndex].stringIndex = i;
                            charBuffer[writeIndex].length = 1;

                            i += 1;
                            writeIndex += 1;
                            continue;
                        case 117: // \u0000 for UTF-16 Unicode
                            if (sourceText.Length > i + 5)
                            {
                                if (writeIndex == charBuffer.Length) ResizeInternalArray(ref charBuffer);

                                charBuffer[writeIndex].unicode = GetUTF16(sourceText, i + 2);
                                charBuffer[writeIndex].stringIndex = i;
                                charBuffer[writeIndex].length = 6;

                                i += 5;
                                writeIndex += 1;
                                continue;
                            }
                            break;
                    }
                }

                // Handle UTF-32 in the input text (string). // Not sure this is needed //
                if (char.IsHighSurrogate(sourceText[i]) && char.IsLowSurrogate(sourceText[i + 1]))
                {
                    if (writeIndex == charBuffer.Length) ResizeInternalArray(ref charBuffer);

                    charBuffer[writeIndex].unicode = char.ConvertToUtf32(sourceText[i], sourceText[i + 1]);
                    charBuffer[writeIndex].stringIndex = i;
                    charBuffer[writeIndex].length = 2;

                    i += 1;
                    writeIndex += 1;
                    continue;
                }

                //// Handle inline replacement of <stlye> and <br> tags.
                if (sourceText[i] == 60 && m_isRichText)
                {
                    if (IsTagName(ref sourceText, "<BR>", i))
                    {
                        if (writeIndex == charBuffer.Length) ResizeInternalArray(ref charBuffer);

                        charBuffer[writeIndex].unicode = 10;
                        charBuffer[writeIndex].stringIndex = i;
                        charBuffer[writeIndex].length = 1;

                        writeIndex += 1;
                        i += 3;

                        continue;
                    }
                    else if (IsTagName(ref sourceText, "<STYLE=", i))
                    {
                        int srcOffset;

                        if (ReplaceOpeningStyleTag(ref sourceText, i, out srcOffset, ref charBuffer, ref writeIndex))
                        {
                            i = srcOffset;
                            continue;
                        }
                    }
                    else if (IsTagName(ref sourceText, "</STYLE>", i))
                    {
                        ReplaceClosingStyleTag(ref sourceText, i, ref charBuffer, ref writeIndex);

                        // Strip </style> even if style is invalid.
                        i += 7;
                        continue;
                    }
                }

                if (writeIndex == charBuffer.Length) ResizeInternalArray(ref charBuffer);

                charBuffer[writeIndex].unicode = sourceText[i];
                charBuffer[writeIndex].stringIndex = i;
                charBuffer[writeIndex].length = 1;

                writeIndex += 1;
            }

            if (writeIndex == charBuffer.Length) ResizeInternalArray(ref charBuffer);

            charBuffer[writeIndex].unicode = 0;
        }


        /// <summary>
        /// Copy contents of StringBuilder into int array.
        /// </summary>
        /// <param name="sourceText">Text to copy.</param>
        /// <param name="charBuffer">Array to store contents.</param>
        protected void StringBuilderToIntArray(StringBuilder sourceText, ref UnicodeChar[] charBuffer)
        {
            if (sourceText == null)
            {
                charBuffer[0].unicode = 0;
                return;
            }

            if (charBuffer == null) charBuffer = new UnicodeChar[8];

            // Clear the Style stack.
            m_styleStack.Clear();

            #if UNITY_EDITOR
            // Create new string to be displayed in the Input Text Box of the Editor Panel.
            m_text = sourceText.ToString();
            #endif

            int writeIndex = 0;

            for (int i = 0; i < sourceText.Length; i++)
            {
                if (m_parseCtrlCharacters && sourceText[i] == 92 && sourceText.Length > i + 1)
                {
                    switch ((int)sourceText[i + 1])
                    {
                        case 85: // \U00000000 for UTF-32 Unicode
                            if (sourceText.Length > i + 9)
                            {
                                if (writeIndex == charBuffer.Length) ResizeInternalArray(ref charBuffer);

                                charBuffer[writeIndex].unicode = GetUTF32(sourceText, i + 2);
                                i += 9;
                                writeIndex += 1;
                                continue;
                            }
                            break;
                        case 92: // \ escape
                            if (sourceText.Length <= i + 2) break;

                            if (writeIndex + 2 > charBuffer.Length) ResizeInternalArray(ref charBuffer);

                            charBuffer[writeIndex].unicode = sourceText[i + 1];
                            charBuffer[writeIndex + 1].unicode = sourceText[i + 2];
                            i += 2;
                            writeIndex += 2;
                            continue;
                        case 110: // \n LineFeed
                            if (writeIndex == charBuffer.Length) ResizeInternalArray(ref charBuffer);

                            charBuffer[writeIndex].unicode = 10;
                            i += 1;
                            writeIndex += 1;
                            continue;
                        case 114: // \r
                            if (writeIndex == charBuffer.Length) ResizeInternalArray(ref charBuffer);

                            charBuffer[writeIndex].unicode = 13;
                            i += 1;
                            writeIndex += 1;
                            continue;
                        case 116: // \t Tab
                            if (writeIndex == charBuffer.Length) ResizeInternalArray(ref charBuffer);

                            charBuffer[writeIndex].unicode = 9;
                            i += 1;
                            writeIndex += 1;
                            continue;
                        case 117: // \u0000 for UTF-16 Unicode
                            if (sourceText.Length > i + 5)
                            {
                                if (writeIndex == charBuffer.Length) ResizeInternalArray(ref charBuffer);

                                charBuffer[writeIndex].unicode = GetUTF16(sourceText, i + 2);
                                i += 5;
                                writeIndex += 1;
                                continue;
                            }
                            break;
                    }
                }

                // Handle UTF-32 in the input text (string).
                if (char.IsHighSurrogate(sourceText[i]) && char.IsLowSurrogate(sourceText[i + 1]))
                {
                    if (writeIndex == charBuffer.Length) ResizeInternalArray(ref charBuffer);

                    charBuffer[writeIndex].unicode = char.ConvertToUtf32(sourceText[i], sourceText[i + 1]);
                    i += 1;
                    writeIndex += 1;
                    continue;
                }

                // Handle inline replacement of <stlye> and <br> tags.
                if (sourceText[i] == 60)
                {
                    if (IsTagName(ref sourceText, "<BR>", i))
                    {
                        if (writeIndex == charBuffer.Length) ResizeInternalArray(ref charBuffer);

                        charBuffer[writeIndex].unicode = 10;
                        writeIndex += 1;
                        i += 3;

                        continue;
                    }
                    else if (IsTagName(ref sourceText, "<STYLE=", i))
                    {
                        int srcOffset;

                        if (ReplaceOpeningStyleTag(ref sourceText, i, out srcOffset, ref charBuffer, ref writeIndex))
                        {
                            i = srcOffset;
                            continue;
                        }
                    }
                    else if (IsTagName(ref sourceText, "</STYLE>", i))
                    {
                        ReplaceClosingStyleTag(ref sourceText, i, ref charBuffer, ref writeIndex);

                        // Strip </style> even if style is invalid.
                        i += 7;
                        continue;
                    }
                }

                if (writeIndex == charBuffer.Length) ResizeInternalArray(ref charBuffer);

                charBuffer[writeIndex].unicode = sourceText[i];
                writeIndex += 1;
            }

            if (writeIndex == charBuffer.Length) ResizeInternalArray(ref charBuffer);

            charBuffer[writeIndex].unicode = 0;
        }


        /// <summary>
        /// Method to handle inline replacement of style tag by opening style definition.
        /// </summary>
        /// <param name="sourceText"></param>
        /// <param name="srcIndex"></param>
        /// <param name="srcOffset"></param>
        /// <param name="charBuffer"></param>
        /// <param name="writeIndex"></param>
        /// <returns></returns>
        bool ReplaceOpeningStyleTag(ref string sourceText, int srcIndex, out int srcOffset, ref UnicodeChar[] charBuffer, ref int writeIndex)
        {
            // Validate <style> tag.
            int hashCode = GetTagHashCode(ref sourceText, srcIndex + 7, out srcOffset);

            TMP_Style style = TMP_StyleSheet.GetStyle(hashCode);

            // Return if we don't have a valid style.
            if (style == null || srcOffset == 0) return false;

            m_styleStack.Add(style.hashCode);

            int styleLength = style.styleOpeningTagArray.Length;

            // Replace <style> tag with opening definition
            int[] openingTagArray = style.styleOpeningTagArray;

            for (int i = 0; i < styleLength; i++)
            {
                int c = openingTagArray[i];

                if (c == 60)
                {
                    if (IsTagName(ref openingTagArray, "<BR>", i))
                    {
                        if (writeIndex == charBuffer.Length) ResizeInternalArray(ref charBuffer);

                        charBuffer[writeIndex].unicode = 10;
                        writeIndex += 1;
                        i += 3;

                        continue;
                    }
                    else if (IsTagName(ref openingTagArray, "<STYLE=", i))
                    {
                        int offset;

                        if (ReplaceOpeningStyleTag(ref openingTagArray, i, out offset, ref charBuffer, ref writeIndex))
                        {
                            i = offset;
                            continue;
                        }
                    }
                    else if (IsTagName(ref openingTagArray, "</STYLE>", i))
                    {
                        ReplaceClosingStyleTag(ref openingTagArray, i, ref charBuffer, ref writeIndex);
                        
                        // Strip </style> even if style is invalid.
                        i += 7;
                        continue;
                    }
                }

                if (writeIndex == charBuffer.Length) ResizeInternalArray(ref charBuffer);

                charBuffer[writeIndex].unicode = c;
                writeIndex += 1;
            }

            return true;
        }


        /// <summary>
        /// Method to handle inline replacement of style tag by opening style definition.
        /// </summary>
        /// <param name="sourceText"></param>
        /// <param name="srcIndex"></param>
        /// <param name="srcOffset"></param>
        /// <param name="charBuffer"></param>
        /// <param name="writeIndex"></param>
        /// <returns></returns>
        bool ReplaceOpeningStyleTag(ref int[] sourceText, int srcIndex, out int srcOffset, ref UnicodeChar[] charBuffer, ref int writeIndex)
        {
            // Validate <style> tag.
            int hashCode = GetTagHashCode(ref sourceText, srcIndex + 7, out srcOffset);

            TMP_Style style = TMP_StyleSheet.GetStyle(hashCode);

            // Return if we don't have a valid style.
            if (style == null || srcOffset == 0) return false;

            m_styleStack.Add(style.hashCode);

            int styleLength = style.styleOpeningTagArray.Length;

            // Replace <style> tag with opening definition
            int[] openingTagArray = style.styleOpeningTagArray;

            for (int i = 0; i < styleLength; i++)
            {
                int c = openingTagArray[i];

                if (c == 60)
                {
                    if (IsTagName(ref openingTagArray, "<BR>", i))
                    {
                        if (writeIndex == charBuffer.Length) ResizeInternalArray(ref charBuffer);

                        charBuffer[writeIndex].unicode = 10;
                        writeIndex += 1;
                        i += 3;

                        continue;
                    }
                    else if (IsTagName(ref openingTagArray, "<STYLE=", i))
                    {
                        int offset;

                        if (ReplaceOpeningStyleTag(ref openingTagArray, i, out offset, ref charBuffer, ref writeIndex))
                        {
                            i = offset;
                            continue;
                        }
                    }
                    else if (IsTagName(ref openingTagArray, "</STYLE>", i))
                    {
                        ReplaceClosingStyleTag(ref openingTagArray, i, ref charBuffer, ref writeIndex);
                        
                        // Strip </style> even if style is invalid.
                        i += 7;
                        continue;
                    }
                }

                if (writeIndex == charBuffer.Length) ResizeInternalArray(ref charBuffer);

                charBuffer[writeIndex].unicode = c;
                writeIndex += 1;
            }

            return true;
        }


        /// <summary>
        /// Method to handle inline replacement of style tag by opening style definition.
        /// </summary>
        /// <param name="sourceText"></param>
        /// <param name="srcIndex"></param>
        /// <param name="srcOffset"></param>
        /// <param name="charBuffer"></param>
        /// <param name="writeIndex"></param>
        /// <returns></returns>
        bool ReplaceOpeningStyleTag(ref char[] sourceText, int srcIndex, out int srcOffset, ref UnicodeChar[] charBuffer, ref int writeIndex)
        {
            // Validate <style> tag.
            int hashCode = GetTagHashCode(ref sourceText, srcIndex + 7, out srcOffset);

            TMP_Style style = TMP_StyleSheet.GetStyle(hashCode);

            // Return if we don't have a valid style.
            if (style == null || srcOffset == 0) return false;

            m_styleStack.Add(style.hashCode);

            int styleLength = style.styleOpeningTagArray.Length;

            // Replace <style> tag with opening definition
            int[] openingTagArray = style.styleOpeningTagArray;

            for (int i = 0; i < styleLength; i++)
            {
                int c = openingTagArray[i];

                if (c == 60)
                {
                    if (IsTagName(ref openingTagArray, "<BR>", i))
                    {
                        if (writeIndex == charBuffer.Length) ResizeInternalArray(ref charBuffer);

                        charBuffer[writeIndex].unicode = 10;
                        writeIndex += 1;
                        i += 3;

                        continue;
                    }
                    else if (IsTagName(ref openingTagArray, "<STYLE=", i))
                    {
                        int offset;

                        if (ReplaceOpeningStyleTag(ref openingTagArray, i, out offset, ref charBuffer, ref writeIndex))
                        {
                            i = offset;
                            continue;
                        }
                    }
                    else if (IsTagName(ref openingTagArray, "</STYLE>", i))
                    {
                        ReplaceClosingStyleTag(ref openingTagArray, i, ref charBuffer, ref writeIndex);

                        // Strip </style> even if style is invalid.
                        i += 7;
                        continue;
                    }
                }

                if (writeIndex == charBuffer.Length) ResizeInternalArray(ref charBuffer);

                charBuffer[writeIndex].unicode = c;
                writeIndex += 1;
            }

            return true;
        }


        /// <summary>
        /// Method to handle inline replacement of style tag by opening style definition.
        /// </summary>
        /// <param name="sourceText"></param>
        /// <param name="srcIndex"></param>
        /// <param name="srcOffset"></param>
        /// <param name="charBuffer"></param>
        /// <param name="writeIndex"></param>
        /// <returns></returns>
        bool ReplaceOpeningStyleTag(ref StringBuilder sourceText, int srcIndex, out int srcOffset, ref UnicodeChar[] charBuffer, ref int writeIndex)
        {
            // Validate <style> tag.
            int hashCode = GetTagHashCode(ref sourceText, srcIndex + 7, out srcOffset);

            TMP_Style style = TMP_StyleSheet.GetStyle(hashCode);

            // Return if we don't have a valid style.
            if (style == null || srcOffset == 0) return false;

            m_styleStack.Add(style.hashCode);

            int styleLength = style.styleOpeningTagArray.Length;

            // Replace <style> tag with opening definition
            int[] openingTagArray = style.styleOpeningTagArray;

            for (int i = 0; i < styleLength; i++)
            {
                int c = openingTagArray[i];

                if (c == 60)
                {
                    if (IsTagName(ref openingTagArray, "<BR>", i))
                    {
                        if (writeIndex == charBuffer.Length) ResizeInternalArray(ref charBuffer);

                        charBuffer[writeIndex].unicode = 10;
                        writeIndex += 1;
                        i += 3;

                        continue;
                    }
                    else if (IsTagName(ref openingTagArray, "<STYLE=", i))
                    {
                        int offset;

                        if (ReplaceOpeningStyleTag(ref openingTagArray, i, out offset, ref charBuffer, ref writeIndex))
                        {
                            i = offset;
                            continue;
                        }
                    }
                    else if (IsTagName(ref openingTagArray, "</STYLE>", i))
                    {
                        ReplaceClosingStyleTag(ref openingTagArray, i, ref charBuffer, ref writeIndex);

                        // Strip </style> even if style is invalid.
                        i += 7;
                        continue;
                    }
                }

                if (writeIndex == charBuffer.Length) ResizeInternalArray(ref charBuffer);

                charBuffer[writeIndex].unicode = c;
                writeIndex += 1;
            }

            return true;
        }


        /// <summary>
        /// Method to handle inline replacement of style tag by closing style definition.
        /// </summary>
        /// <param name="sourceText"></param>
        /// <param name="srcIndex"></param>
        /// <param name="charBuffer"></param>
        /// <param name="writeIndex"></param>
        /// <returns></returns>
        bool ReplaceClosingStyleTag(ref string sourceText, int srcIndex, ref UnicodeChar[] charBuffer, ref int writeIndex)
        {
            // Get style from the Style Stack
            int hashCode = m_styleStack.CurrentItem();
            TMP_Style style = TMP_StyleSheet.GetStyle(hashCode);

            m_styleStack.Remove();

            // Return if we don't have a valid style.
            if (style == null) return false;

            int styleLength = style.styleClosingTagArray.Length;

            // Replace <style> tag with opening definition
            int[] closingTagArray = style.styleClosingTagArray;

            for (int i = 0; i < styleLength; i++)
            {
                int c = closingTagArray[i];

                if (c == 60)
                {
                    if (IsTagName(ref closingTagArray, "<BR>", i))
                    {
                        if (writeIndex == charBuffer.Length) ResizeInternalArray(ref charBuffer);

                        charBuffer[writeIndex].unicode = 10;
                        writeIndex += 1;
                        i += 3;

                        continue;
                    }
                    else if (IsTagName(ref closingTagArray, "<STYLE=", i))
                    {
                        int offset;

                        if (ReplaceOpeningStyleTag(ref closingTagArray, i, out offset, ref charBuffer, ref writeIndex))
                        {
                            i = offset;
                            continue;
                        }
                    }
                    else if (IsTagName(ref closingTagArray, "</STYLE>", i))
                    {
                        ReplaceClosingStyleTag(ref closingTagArray, i, ref charBuffer, ref writeIndex);

                        // Strip </style> even if style is invalid.
                        i += 7;
                        continue;
                    }
                }

                if (writeIndex == charBuffer.Length) ResizeInternalArray(ref charBuffer);

                charBuffer[writeIndex].unicode = c;
                writeIndex += 1;
            }

            return true;
        }


        /// <summary>
        /// Method to handle inline replacement of style tag by closing style definition.
        /// </summary>
        /// <param name="sourceText"></param>
        /// <param name="srcIndex"></param>
        /// <param name="charBuffer"></param>
        /// <param name="writeIndex"></param>
        /// <returns></returns>
        bool ReplaceClosingStyleTag(ref int[] sourceText, int srcIndex, ref UnicodeChar[] charBuffer, ref int writeIndex)
        {
            // Get style from the Style Stack
            int hashCode = m_styleStack.CurrentItem();
            TMP_Style style = TMP_StyleSheet.GetStyle(hashCode);

            m_styleStack.Remove();

            // Return if we don't have a valid style.
            if (style == null) return false;

            int styleLength = style.styleClosingTagArray.Length;

            // Replace <style> tag with opening definition
            int[] closingTagArray = style.styleClosingTagArray;

            for (int i = 0; i < styleLength; i++)
            {
                int c = closingTagArray[i];

                if (c == 60)
                {
                    if (IsTagName(ref closingTagArray, "<BR>", i))
                    {
                        if (writeIndex == charBuffer.Length) ResizeInternalArray(ref charBuffer);

                        charBuffer[writeIndex].unicode = 10;
                        writeIndex += 1;
                        i += 3;

                        continue;
                    }
                    else if (IsTagName(ref closingTagArray, "<STYLE=", i))
                    {
                        int offset;

                        if (ReplaceOpeningStyleTag(ref closingTagArray, i, out offset, ref charBuffer, ref writeIndex))
                        {
                            i = offset;
                            continue;
                        }
                    }
                    else if (IsTagName(ref closingTagArray, "</STYLE>", i))
                    {
                        ReplaceClosingStyleTag(ref closingTagArray, i, ref charBuffer, ref writeIndex);

                        // Strip </style> even if style is invalid.
                        i += 7;
                        continue;
                    }
                }

                if (writeIndex == charBuffer.Length) ResizeInternalArray(ref charBuffer);

                charBuffer[writeIndex].unicode = c;
                writeIndex += 1;
            }

            return true;
        }


        /// <summary>
        /// Method to handle inline replacement of style tag by closing style definition.
        /// </summary>
        /// <param name="sourceText"></param>
        /// <param name="srcIndex"></param>
        /// <param name="charBuffer"></param>
        /// <param name="writeIndex"></param>
        /// <returns></returns>
        bool ReplaceClosingStyleTag(ref char[] sourceText, int srcIndex, ref UnicodeChar[] charBuffer, ref int writeIndex)
        {
            // Get style from the Style Stack
            int hashCode = m_styleStack.CurrentItem();
            TMP_Style style = TMP_StyleSheet.GetStyle(hashCode);

            m_styleStack.Remove();

            // Return if we don't have a valid style.
            if (style == null) return false;

            int styleLength = style.styleClosingTagArray.Length;

            // Replace <style> tag with opening definition
            int[] closingTagArray = style.styleClosingTagArray;

            for (int i = 0; i < styleLength; i++)
            {
                int c = closingTagArray[i];

                if (c == 60)
                {
                    if (IsTagName(ref closingTagArray, "<BR>", i))
                    {
                        if (writeIndex == charBuffer.Length) ResizeInternalArray(ref charBuffer);

                        charBuffer[writeIndex].unicode = 10;
                        writeIndex += 1;
                        i += 3;

                        continue;
                    }
                    else if (IsTagName(ref closingTagArray, "<STYLE=", i))
                    {
                        int offset;

                        if (ReplaceOpeningStyleTag(ref closingTagArray, i, out offset, ref charBuffer, ref writeIndex))
                        {
                            i = offset;
                            continue;
                        }
                    }
                    else if (IsTagName(ref closingTagArray, "</STYLE>", i))
                    {
                        ReplaceClosingStyleTag(ref closingTagArray, i, ref charBuffer, ref writeIndex);

                        // Strip </style> even if style is invalid.
                        i += 7;
                        continue;
                    }
                }

                if (writeIndex == charBuffer.Length) ResizeInternalArray(ref charBuffer);

                charBuffer[writeIndex].unicode = c;
                writeIndex += 1;
            }

            return true;
        }

        /// <summary>
        /// Method to handle inline replacement of style tag by closing style definition.
        /// </summary>
        /// <param name="sourceText"></param>
        /// <param name="srcIndex"></param>
        /// <param name="charBuffer"></param>
        /// <param name="writeIndex"></param>
        /// <returns></returns>
        bool ReplaceClosingStyleTag(ref StringBuilder sourceText, int srcIndex, ref UnicodeChar[] charBuffer, ref int writeIndex)
        {
            // Get style from the Style Stack
            int hashCode = m_styleStack.CurrentItem();
            TMP_Style style = TMP_StyleSheet.GetStyle(hashCode);

            m_styleStack.Remove();

            // Return if we don't have a valid style.
            if (style == null) return false;

            int styleLength = style.styleClosingTagArray.Length;

            // Replace <style> tag with opening definition
            int[] closingTagArray = style.styleClosingTagArray;

            for (int i = 0; i < styleLength; i++)
            {
                int c = closingTagArray[i];

                if (c == 60)
                {
                    if (IsTagName(ref closingTagArray, "<BR>", i))
                    {
                        if (writeIndex == charBuffer.Length) ResizeInternalArray(ref charBuffer);

                        charBuffer[writeIndex].unicode = 10;
                        writeIndex += 1;
                        i += 3;

                        continue;
                    }
                    else if (IsTagName(ref closingTagArray, "<STYLE=", i))
                    {
                        int offset;

                        if (ReplaceOpeningStyleTag(ref closingTagArray, i, out offset, ref charBuffer, ref writeIndex))
                        {
                            i = offset;
                            continue;
                        }
                    }
                    else if (IsTagName(ref closingTagArray, "</STYLE>", i))
                    {
                        ReplaceClosingStyleTag(ref closingTagArray, i, ref charBuffer, ref writeIndex);

                        // Strip </style> even if style is invalid.
                        i += 7;
                        continue;
                    }
                }

                if (writeIndex == charBuffer.Length) ResizeInternalArray(ref charBuffer);

                charBuffer[writeIndex].unicode = c;
                writeIndex += 1;
            }

            return true;
        }


        /// <summary>
        /// Method to check for a matching rich text tag.
        /// </summary>
        /// <param name="text"></param>
        /// <param name="tag"></param>
        /// <param name="index"></param>
        /// <returns></returns>
        bool IsTagName (ref string text, string tag, int index)
        {
            if (text.Length < index + tag.Length) return false;
            
            for (int i = 0; i < tag.Length; i++)
            {
                if (TMP_TextUtilities.ToUpperFast(text[index + i]) != tag[i]) return false;
            }

            return true;
        }

        /// <summary>
        /// Method to check for a matching rich text tag.
        /// </summary>
        /// <param name="text"></param>
        /// <param name="tag"></param>
        /// <param name="index"></param>
        /// <returns></returns>
        bool IsTagName(ref char[] text, string tag, int index)
        {
            if (text.Length < index + tag.Length) return false;

            for (int i = 0; i < tag.Length; i++)
            {
                if (TMP_TextUtilities.ToUpperFast(text[index + i]) != tag[i]) return false;
            }

            return true;
        }

        /// <summary>
        /// Method to check for a matching rich text tag.
        /// </summary>
        /// <param name="text"></param>
        /// <param name="tag"></param>
        /// <param name="index"></param>
        /// <returns></returns>
        bool IsTagName(ref int[] text, string tag, int index)
        {
            if (text.Length < index + tag.Length) return false;

            for (int i = 0; i < tag.Length; i++)
            {
                if (TMP_TextUtilities.ToUpperFast((char)text[index + i]) != tag[i]) return false;
            }

            return true;
        }

        /// <summary>
        /// Method to check for a matching rich text tag.
        /// </summary>
        /// <param name="text"></param>
        /// <param name="tag"></param>
        /// <param name="index"></param>
        /// <returns></returns>
        bool IsTagName(ref StringBuilder text, string tag, int index)
        {
            if (text.Length < index + tag.Length) return false;

            for (int i = 0; i < tag.Length; i++)
            {
                if (TMP_TextUtilities.ToUpperFast(text[index + i]) != tag[i]) return false;
            }

            return true;
        }

        /// <summary>
        /// Get Hashcode for a given tag.
        /// </summary>
        /// <param name="text"></param>
        /// <param name="index"></param>
        /// <param name="closeIndex"></param>
        /// <returns></returns>
        int GetTagHashCode(ref string text, int index, out int closeIndex)
        {
            int hashCode = 0;
            closeIndex = 0;

            for (int i = index; i < text.Length; i++)
            {
                // Skip quote '"' character
                if (text[i] == 34) continue;

                // Break at '>'
                if (text[i] == 62) { closeIndex = i; break; }

                hashCode = (hashCode << 5) + hashCode ^ text[i];
            }

            return hashCode;
        }

        /// <summary>
        /// Get Hashcode for a given tag.
        /// </summary>
        /// <param name="text"></param>
        /// <param name="index"></param>
        /// <param name="closeIndex"></param>
        /// <returns></returns>
        int GetTagHashCode(ref char[] text, int index, out int closeIndex)
        {
            int hashCode = 0;
            closeIndex = 0;

            for (int i = index; i < text.Length; i++)
            {
                // Skip quote '"' character
                if (text[i] == 34) continue;

                // Break at '>'
                if (text[i] == 62) { closeIndex = i; break; }

                hashCode = (hashCode << 5) + hashCode ^ text[i];
            }

            return hashCode;
        }

        /// <summary>
        /// Get Hashcode for a given tag.
        /// </summary>
        /// <param name="text"></param>
        /// <param name="index"></param>
        /// <param name="closeIndex"></param>
        /// <returns></returns>
        int GetTagHashCode(ref int[] text, int index, out int closeIndex)
        {
            int hashCode = 0;
            closeIndex = 0;

            for (int i = index; i < text.Length; i++)
            {
                // Skip quote '"' character
                if (text[i] == 34) continue;

                // Break at '>'
                if (text[i] == 62) { closeIndex = i; break; }

                hashCode = (hashCode << 5) + hashCode ^ text[i];
            }

            return hashCode;
        }

        /// <summary>
        ///  Get Hashcode for a given tag.
        /// </summary>
        /// <param name="text"></param>
        /// <param name="index"></param>
        /// <param name="closeIndex"></param>
        /// <returns></returns>
        int GetTagHashCode(ref StringBuilder text, int index, out int closeIndex)
        {
            int hashCode = 0;
            closeIndex = 0;

            for (int i = index; i < text.Length; i++)
            {
                // Skip quote '"' character
                if (text[i] == 34) continue;

                // Break at '>'
                if (text[i] == 62) { closeIndex = i; break; }

                hashCode = (hashCode << 5) + hashCode ^ text[i];
            }

            return hashCode;
        }

        /// <summary>
        /// 
        /// </summary>
        void ResizeInternalArray <T>(ref T[] array)
        {
            int size = Mathf.NextPowerOfTwo(array.Length + 1);

            System.Array.Resize(ref array, size);
        }


        private readonly float[] k_Power = { 5e-1f, 5e-2f, 5e-3f, 5e-4f, 5e-5f, 5e-6f, 5e-7f, 5e-8f, 5e-9f, 5e-10f }; // Used by FormatText to enable rounding and avoid using Mathf.Pow.

        /// <summary>
        /// Function used in conjunction with SetText()
        /// </summary>
        /// <param name="number"></param>
        /// <param name="index"></param>
        /// <param name="precision"></param>
        protected void AddFloatToCharArray(double number, ref int index, int precision)
        {
            if (number < 0)
            {
                m_input_CharArray[index++] = '-';
                number = -number;
            }

            number += k_Power[Mathf.Min(9, precision)];

            double integer = Math.Truncate(number);

            AddIntToCharArray(integer, ref index, precision);

            if (precision > 0)
            {
                // Add the decimal point
                m_input_CharArray[index++] = '.';

                number -= integer;
                for (int p = 0; p < precision; p++)
                {
                    number *= 10;
                    long d = (long)(number);

                    m_input_CharArray[index++] = (char)(d + 48);
                    number -= d;
                }
            }
        }


        /// <summary>
        /// // Function used in conjunction with SetText()
        /// </summary>
        /// <param name="number"></param>
        /// <param name="index"></param>
        /// <param name="precision"></param>
        protected void AddIntToCharArray(double number, ref int index, int precision)
        {
            if (number < 0)
            {
                m_input_CharArray[index++] = '-';
                number = -number;
            }

            int i = index;
            do
            {
                m_input_CharArray[i++] = (char)(number % 10 + 48);
                number /= 10;
            } while (number > 0.999d);

            int lastIndex = i;

            // Reverse string
            while (index + 1 < i)
            {
                i -= 1;
                char t = m_input_CharArray[index];
                m_input_CharArray[index] = m_input_CharArray[i];
                m_input_CharArray[i] = t;
                index += 1;
            }
            index = lastIndex;
        }


        /// <summary>
        /// Method used to determine the number of visible characters and required buffer allocations.
        /// </summary>
        /// <param name="chars"></param>
        /// <returns></returns>
        protected virtual int SetArraySizes(UnicodeChar[] chars) { return 0; }


        /// <summary>
        /// Method which parses the text input, does the layout of the text as well as generating the geometry.
        /// </summary>
        protected virtual void GenerateTextMesh() { }


        /// <summary>
        /// Function to Calculate the Preferred Width and Height of the text object.
        /// </summary>
        /// <returns></returns>
        public Vector2 GetPreferredValues()
        {
            if (m_isInputParsingRequired || m_isTextTruncated)
            {
                m_isCalculatingPreferredValues = true;
                ParseInputText();
            }

            // CALCULATE PREFERRED WIDTH
            float preferredWidth = GetPreferredWidth();

            // CALCULATE PREFERRED HEIGHT
            float preferredHeight = GetPreferredHeight();

            return new Vector2(preferredWidth, preferredHeight);
        }


        /// <summary>
        /// Function to Calculate the Preferred Width and Height of the text object given the provided width and height.
        /// </summary>
        /// <returns></returns>
        public Vector2 GetPreferredValues(float width, float height)
        {
            if (m_isInputParsingRequired || m_isTextTruncated)
            {
                m_isCalculatingPreferredValues = true;
                ParseInputText();
            }

            Vector2 margin = new Vector2(width, height);

            // CALCULATE PREFERRED WIDTH
            float preferredWidth = GetPreferredWidth(margin);

            // CALCULATE PREFERRED HEIGHT
            float preferredHeight = GetPreferredHeight(margin);

            return new Vector2(preferredWidth, preferredHeight);
        }


        /// <summary>
        /// Function to Calculate the Preferred Width and Height of the text object given a certain string.
        /// </summary>
        /// <param name="text"></param>
        /// <returns></returns>
        public Vector2 GetPreferredValues(string text)
        {
            m_isCalculatingPreferredValues = true;

            StringToCharArray(text, ref m_TextParsingBuffer);
            SetArraySizes(m_TextParsingBuffer);

            Vector2 margin = k_LargePositiveVector2;

            // CALCULATE PREFERRED WIDTH
            float preferredWidth = GetPreferredWidth(margin);

            // CALCULATE PREFERRED HEIGHT
            float preferredHeight = GetPreferredHeight(margin);

            return new Vector2(preferredWidth, preferredHeight);
        }


        /// <summary>
        ///  Function to Calculate the Preferred Width and Height of the text object given a certain string and size of text container.
        /// </summary>
        /// <param name="text"></param>
        /// <returns></returns>
        public Vector2 GetPreferredValues(string text, float width, float height)
        {
            m_isCalculatingPreferredValues = true;

            StringToCharArray(text, ref m_TextParsingBuffer);
            SetArraySizes(m_TextParsingBuffer);

            Vector2 margin = new Vector2(width, height);

            // CALCULATE PREFERRED WIDTH
            float preferredWidth = GetPreferredWidth(margin);

            // CALCULATE PREFERRED HEIGHT
            float preferredHeight = GetPreferredHeight(margin);

            return new Vector2(preferredWidth, preferredHeight);
        }


        /// <summary>
        /// Method to calculate the preferred width of a text object.
        /// </summary>
        /// <returns></returns>
        protected float GetPreferredWidth()
        {
            if (TMP_Settings.instance == null) return 0;

            float fontSize = m_enableAutoSizing ? m_fontSizeMax : m_fontSize;

            // Reset auto sizing point size bounds
            m_minFontSize = m_fontSizeMin;
            m_maxFontSize = m_fontSizeMax;
            m_charWidthAdjDelta = 0;

            // Set Margins to Infinity
            Vector2 margin = k_LargePositiveVector2;

            if (m_isInputParsingRequired || m_isTextTruncated)
            {
                m_isCalculatingPreferredValues = true;
                ParseInputText();
            }

            m_recursiveCount = 0;
            float preferredWidth = CalculatePreferredValues(fontSize, margin, true).x;

            m_isPreferredWidthDirty = false;

            //Debug.Log("GetPreferredWidth() Called at frame " + Time.frameCount + ". Returning width of " + preferredWidth);

            return preferredWidth;
        }


        /// <summary>
        /// Method to calculate the preferred width of a text object.
        /// </summary>
        /// <param name="margin"></param>
        /// <returns></returns>
        protected float GetPreferredWidth(Vector2 margin)
        {
            float fontSize = m_enableAutoSizing ? m_fontSizeMax : m_fontSize;

            // Reset auto sizing point size bounds
            m_minFontSize = m_fontSizeMin;
            m_maxFontSize = m_fontSizeMax;
            m_charWidthAdjDelta = 0;

            m_recursiveCount = 0;
            float preferredWidth = CalculatePreferredValues(fontSize, margin, true).x;

            //Debug.Log("GetPreferredWidth() Called. Returning width of " + preferredWidth);

            return preferredWidth;
        }


        /// <summary>
        /// Method to calculate the preferred height of a text object.
        /// </summary>
        /// <returns></returns>
        protected float GetPreferredHeight()
        {
            if (TMP_Settings.instance == null) return 0;

            float fontSize = m_enableAutoSizing ? m_fontSizeMax : m_fontSize;

            // Reset auto sizing point size bounds
            m_minFontSize = m_fontSizeMin;
            m_maxFontSize = m_fontSizeMax;
            m_charWidthAdjDelta = 0;

            Vector2 margin = new Vector2(m_marginWidth != 0 ? m_marginWidth : k_LargePositiveFloat, k_LargePositiveFloat);

            if (m_isInputParsingRequired || m_isTextTruncated)
            {
                m_isCalculatingPreferredValues = true;
                ParseInputText();
            }

            m_recursiveCount = 0;
            float preferredHeight = CalculatePreferredValues(fontSize, margin, !m_enableAutoSizing).y;

            m_isPreferredHeightDirty = false;

            //Debug.Log("GetPreferredHeight() Called. Returning height of " + preferredHeight);

            return preferredHeight;
        }


        /// <summary>
        /// Method to calculate the preferred height of a text object.
        /// </summary>
        /// <param name="margin"></param>
        /// <returns></returns>
        protected float GetPreferredHeight(Vector2 margin)
        {
            float fontSize = m_enableAutoSizing ? m_fontSizeMax : m_fontSize;

            // Reset auto sizing point size bounds
            m_minFontSize = m_fontSizeMin;
            m_maxFontSize = m_fontSizeMax;
            m_charWidthAdjDelta = 0;

            m_recursiveCount = 0;
            float preferredHeight = CalculatePreferredValues(fontSize, margin, true).y;

            //Debug.Log("GetPreferredHeight() Called. Returning height of " + preferredHeight);

            return preferredHeight;
        }


        /// <summary>
        /// Method returning the rendered width and height of the text object.
        /// </summary>
        /// <returns></returns>
        public Vector2 GetRenderedValues()
        {
            return GetTextBounds().size;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="onlyVisibleCharacters">Should returned value only factor in visible characters and exclude those greater than maxVisibleCharacters for instance.</param>
        /// <returns></returns>
        public Vector2 GetRenderedValues(bool onlyVisibleCharacters)
        {
            return GetTextBounds(onlyVisibleCharacters).size;
        }


        /// <summary>
        /// Method returning the rendered width of the text object.
        /// </summary>
        /// <returns></returns>
        protected float GetRenderedWidth()
        {
            return GetRenderedValues().x;
        }

        /// <summary>
        /// Method returning the rendered width of the text object.
        /// </summary>
        /// <returns></returns>
        protected float GetRenderedWidth(bool onlyVisibleCharacters)
        {
            return GetRenderedValues(onlyVisibleCharacters).x;
        }

        /// <summary>
        /// Method returning the rendered height of the text object.
        /// </summary>
        /// <returns></returns>
        protected float GetRenderedHeight()
        {
            return GetRenderedValues().y;
        }

        /// <summary>
        /// Method returning the rendered height of the text object.
        /// </summary>
        /// <returns></returns>
        protected float GetRenderedHeight(bool onlyVisibleCharacters)
        {
            return GetRenderedValues(onlyVisibleCharacters).y;
        }


        /// <summary>
        /// Method to calculate the preferred width and height of the text object.
        /// </summary>
        /// <returns></returns>
        protected virtual Vector2 CalculatePreferredValues(float defaultFontSize, Vector2 marginSize, bool ignoreTextAutoSizing)
        {
            //Debug.Log("*** CalculatePreferredValues() ***"); // ***** Frame: " + Time.frameCount);

            ////Profiler.BeginSample("TMP Generate Text - Phase I");

            // Early exit if no font asset was assigned. This should not be needed since LiberationSans SDF will be assigned by default.
            if (m_fontAsset == null || m_fontAsset.characterLookupTable == null)
            {
                Debug.LogWarning("Can't Generate Mesh! No Font Asset has been assigned to Object ID: " + this.GetInstanceID());

                return Vector2.zero;
            }

            // Early exit if we don't have any Text to generate.
            if (m_TextParsingBuffer == null || m_TextParsingBuffer.Length == 0 || m_TextParsingBuffer[0].unicode == (char)0)
            {
                return Vector2.zero;
            }

            m_currentFontAsset = m_fontAsset;
            m_currentMaterial = m_sharedMaterial;
            m_currentMaterialIndex = 0;
            m_materialReferenceStack.SetDefault(new MaterialReference(0, m_currentFontAsset, null, m_currentMaterial, m_padding));

            // Total character count is computed when the text is parsed.
            int totalCharacterCount = m_totalCharacterCount; // m_VisibleCharacters.Count;

            if (m_internalCharacterInfo == null || totalCharacterCount > m_internalCharacterInfo.Length)
            {
                m_internalCharacterInfo = new TMP_CharacterInfo[totalCharacterCount > 1024 ? totalCharacterCount + 256 : Mathf.NextPowerOfTwo(totalCharacterCount)];
            }

            // Calculate the scale of the font based on selected font size and sampling point size.
            // baseScale is calculated using the font asset assigned to the text object.
            float baseScale = m_fontScale = (defaultFontSize / m_fontAsset.faceInfo.pointSize * m_fontAsset.faceInfo.scale * (m_isOrthographic ? 1 : 0.1f));
            float currentElementScale = baseScale;
            m_fontScaleMultiplier = 1;

            m_currentFontSize = defaultFontSize;
            m_sizeStack.SetDefault(m_currentFontSize);
            float fontSizeDelta = 0;

            int charCode = 0; // Holds the character code of the currently being processed character.

            m_FontStyleInternal = m_fontStyle; // Set the default style.

            m_lineJustification = m_textAlignment; // Sets the line justification mode to match editor alignment.
            m_lineJustificationStack.SetDefault(m_lineJustification);

            float bold_xAdvance_multiplier = 1; // Used to increase spacing between character when style is bold.

            m_baselineOffset = 0; // Used by subscript characters.
            m_baselineOffsetStack.Clear();

            m_lineOffset = 0; // Amount of space between lines (font line spacing + m_linespacing).
            m_lineHeight = TMP_Math.FLOAT_UNSET;
            float lineGap = m_currentFontAsset.faceInfo.lineHeight - (m_currentFontAsset.faceInfo.ascentLine - m_currentFontAsset.faceInfo.descentLine);

            m_cSpacing = 0; // Amount of space added between characters as a result of the use of the <cspace> tag.
            m_monoSpacing = 0;
            float lineOffsetDelta = 0;
            m_xAdvance = 0; // Used to track the position of each character.
            float maxXAdvance = 0; // Used to determine Preferred Width.

            tag_LineIndent = 0; // Used for indentation of text.
            tag_Indent = 0;
            m_indentStack.SetDefault(0);
            tag_NoParsing = false;
            //m_isIgnoringAlignment = false;

            m_characterCount = 0; // Total characters in the char[]


            // Tracking of line information
            m_firstCharacterOfLine = 0;
            m_maxLineAscender = k_LargeNegativeFloat;
            m_maxLineDescender = k_LargePositiveFloat;
            m_lineNumber = 0;

            float marginWidth = marginSize.x;
            //float marginHeight = marginSize.y;
            m_marginLeft = 0;
            m_marginRight = 0;
            m_width = -1;

            // Used by Unity's Auto Layout system.
            float renderedWidth = 0;
            float renderedHeight = 0;
            float linebreakingWidth = 0;
            m_isCalculatingPreferredValues = true;

            // Tracking of the highest Ascender
            m_maxAscender = 0;
            m_maxDescender = 0;


            // Initialize struct to track states of word wrapping
            bool isFirstWord = true;
            bool isLastBreakingChar = false;
            WordWrapState savedLineState = new WordWrapState();
            SaveWordWrappingState(ref savedLineState, 0, 0);
            WordWrapState savedWordWrapState = new WordWrapState();
            int wrappingIndex = 0;

            // Counter to prevent recursive lockup when computing preferred values.
            m_recursiveCount += 1;

            // Parse through Character buffer to read HTML tags and begin creating mesh.
            for (int i = 0; m_TextParsingBuffer[i].unicode != 0; i++)
            {
                charCode = (int)m_TextParsingBuffer[i].unicode;

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

                // Handle Font Styles like LowerCase, UpperCase and SmallCaps.
                #region Handling of LowerCase, UpperCase and SmallCaps Font Styles

                float smallCapsMultiplier = 1.0f;

                if (m_textElementType == TMP_TextElementType.Character)
                {
                    if (/*(m_fontStyle & FontStyles.UpperCase) == FontStyles.UpperCase ||*/ (m_FontStyleInternal & FontStyles.UpperCase) == FontStyles.UpperCase)
                    {
                        // If this character is lowercase, switch to uppercase.
                        if (char.IsLower((char)charCode))
                            charCode = char.ToUpper((char)charCode);

                    }
                    else if (/*(m_fontStyle & FontStyles.LowerCase) == FontStyles.LowerCase ||*/ (m_FontStyleInternal & FontStyles.LowerCase) == FontStyles.LowerCase)
                    {
                        // If this character is uppercase, switch to lowercase.
                        if (char.IsUpper((char)charCode))
                            charCode = char.ToLower((char)charCode);
                    }
                    else if (/*(m_fontStyle & FontStyles.SmallCaps) == FontStyles.SmallCaps ||*/ (m_FontStyleInternal & FontStyles.SmallCaps) == FontStyles.SmallCaps)
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

                    m_currentFontAsset = m_fontAsset;

                    // The sprite scale calculations are based on the font asset assigned to the text object.
                    // Sprite scale is used to determine line height
                    // Current element scale represents a modified scale to normalize the sprite based on the font baseline to ascender.
                    float spriteScale = (m_currentFontSize / m_fontAsset.faceInfo.pointSize * m_fontAsset.faceInfo.scale * (m_isOrthographic ? 1 : 0.1f));
                    currentElementScale = m_fontAsset.faceInfo.ascentLine / sprite.glyph.metrics.height * sprite.scale * spriteScale;

                    m_cached_TextElement = sprite;

                    m_internalCharacterInfo[m_characterCount].elementType = TMP_TextElementType.Sprite;
                    m_internalCharacterInfo[m_characterCount].scale = spriteScale;
                    //m_internalCharacterInfo[m_characterCount].spriteAsset = m_currentSpriteAsset;
                    //m_internalCharacterInfo[m_characterCount].fontAsset = m_currentFontAsset;
                    //m_internalCharacterInfo[m_characterCount].materialReferenceIndex = m_currentMaterialIndex;

                    m_currentMaterialIndex = prev_MaterialIndex;
                }
                else if (m_textElementType == TMP_TextElementType.Character)
                {
                    m_cached_TextElement = m_textInfo.characterInfo[m_characterCount].textElement;
                    if (m_cached_TextElement == null) continue;

                    //m_currentFontAsset = m_textInfo.characterInfo[m_characterCount].fontAsset;
                    //m_currentMaterial = m_textInfo.characterInfo[m_characterCount].material;
                    m_currentMaterialIndex = m_textInfo.characterInfo[m_characterCount].materialReferenceIndex;

                    // Re-calculate font scale as the font asset may have changed.
                    m_fontScale = m_currentFontSize * smallCapsMultiplier / m_currentFontAsset.faceInfo.pointSize * m_currentFontAsset.faceInfo.scale * (m_isOrthographic ? 1 : 0.1f);

                    currentElementScale = m_fontScale * m_fontScaleMultiplier * m_cached_TextElement.scale;

                    m_internalCharacterInfo[m_characterCount].elementType = TMP_TextElementType.Character;

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
                m_internalCharacterInfo[m_characterCount].character = (char)charCode;


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
                //if (m_isRightToLeft)
                //{
                //    m_xAdvance -= ((m_cached_TextElement.xAdvance * bold_xAdvance_multiplier + m_characterSpacing + m_wordSpacing + m_currentFontAsset.normalSpacingOffset) * currentElementScale + m_cSpacing) * (1 - m_charWidthAdjDelta);

                //    if (char.IsWhiteSpace((char)charCode) || charCode == 0x200B)
                //        m_xAdvance -= m_wordSpacing * currentElementScale;
                //}
                #endregion


                // Handle Mono Spacing
                #region Handle Mono Spacing
                float monoAdvance = 0;
                if (m_monoSpacing != 0)
                {
                    monoAdvance = (m_monoSpacing / 2 - (m_cached_TextElement.glyph.metrics.width / 2 + m_cached_TextElement.glyph.metrics.horizontalBearingX) * currentElementScale);
                    m_xAdvance += monoAdvance;
                }
                #endregion


                // Set Padding based on selected font style
                #region Handle Style Padding
                if (m_textElementType == TMP_TextElementType.Character && !isUsingAltTypeface && ((m_FontStyleInternal & FontStyles.Bold) == FontStyles.Bold)) // Checks for any combination of Bold Style.
                {
                    bold_xAdvance_multiplier = 1 + m_currentFontAsset.boldSpacing * 0.01f;
                }
                else
                {
                    //style_padding = m_currentFontAsset.normalStyle * 2;
                    bold_xAdvance_multiplier = 1.0f;
                }
                #endregion Handle Style Padding

                m_internalCharacterInfo[m_characterCount].baseLine = 0 - m_lineOffset + m_baselineOffset;


                // Compute and save text element Ascender and maximum line Ascender.
                float elementAscender = m_currentFontAsset.faceInfo.ascentLine * (m_textElementType == TMP_TextElementType.Character ? currentElementScale / smallCapsMultiplier : m_internalCharacterInfo[m_characterCount].scale) + m_baselineOffset;
                m_internalCharacterInfo[m_characterCount].ascender = elementAscender - m_lineOffset;
                m_maxLineAscender = elementAscender > m_maxLineAscender ? elementAscender : m_maxLineAscender;

                // Compute and save text element Descender and maximum line Descender.
                float elementDescender = m_currentFontAsset.faceInfo.descentLine * (m_textElementType == TMP_TextElementType.Character ? currentElementScale / smallCapsMultiplier: m_internalCharacterInfo[m_characterCount].scale) + m_baselineOffset;
                float elementDescenderII = m_internalCharacterInfo[m_characterCount].descender = elementDescender - m_lineOffset;
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

                if (m_lineNumber == 0) m_maxAscender = m_maxAscender > elementAscender ? m_maxAscender : elementAscender;
                //if (m_lineOffset == 0) pageAscender = pageAscender > elementAscender ? pageAscender : elementAscender;

                // Setup Mesh for visible text elements. ie. not a SPACE / LINEFEED / CARRIAGE RETURN.
                #region Handle Visible Characters
                if (charCode == 9 || charCode == 0xA0 || charCode == 0x2007 || (!char.IsWhiteSpace((char)charCode) && charCode != 0x200B) || m_textElementType == TMP_TextElementType.Sprite)
                {
                    // Check if Character exceeds the width of the Text Container
                    #region Handle Line Breaking, Text Auto-Sizing and Horizontal Overflow
                    float width = m_width != -1 ? Mathf.Min(marginWidth + 0.0001f - m_marginLeft - m_marginRight, m_width) : marginWidth + 0.0001f - m_marginLeft - m_marginRight;

                    bool isJustifiedOrFlush = ((_HorizontalAlignmentOptions)m_lineJustification & _HorizontalAlignmentOptions.Flush) == _HorizontalAlignmentOptions.Flush || ((_HorizontalAlignmentOptions)m_lineJustification & _HorizontalAlignmentOptions.Justified) == _HorizontalAlignmentOptions.Justified;

                    // Calculate the line breaking width of the text.
                    linebreakingWidth = m_xAdvance + m_cached_TextElement.glyph.metrics.horizontalAdvance * (1 - m_charWidthAdjDelta) * (charCode != 0xAD ? currentElementScale : old_scale);

                    // Check if Character exceeds the width of the Text Container
                    if (linebreakingWidth > width * (isJustifiedOrFlush ? 1.05f : 1.0f))
                    {
                        // Word Wrapping
                        #region Handle Word Wrapping
                        if (enableWordWrapping && m_characterCount != m_firstCharacterOfLine)
                        {
                            // Check if word wrapping is still possible
                            #region Line Breaking Check
                            if (wrappingIndex == savedWordWrapState.previous_WordBreak || isFirstWord)
                            {
                                // Word wrapping is no longer possible. Shrink size of text if auto-sizing is enabled.
                                #region Text Auto-Sizing
                                if (ignoreTextAutoSizing == false && m_currentFontSize > m_fontSizeMin)
                                {
                                    // Handle Character Width Adjustments
                                    #region Character Width Adjustments
                                    if (m_charWidthAdjDelta < m_charWidthMaxAdj / 100)
                                    {
                                        m_recursiveCount = 0;
                                        m_charWidthAdjDelta += 0.01f;
                                        return CalculatePreferredValues(defaultFontSize, marginSize, false);
                                    }
                                    #endregion

                                    // Adjust Point Size
                                    m_maxFontSize = defaultFontSize;

                                    defaultFontSize -= Mathf.Max((defaultFontSize - m_minFontSize) / 2, 0.05f);
                                    defaultFontSize = (int)(Mathf.Max(defaultFontSize, m_fontSizeMin) * 20 + 0.5f) / 20f;

                                    if (m_recursiveCount > 20) return new Vector2(renderedWidth, renderedHeight);
                                    return CalculatePreferredValues(defaultFontSize, marginSize, false);
                                }
                                #endregion

                                // Word wrapping is no longer possible, now breaking up individual words.
                                if (m_isCharacterWrappingEnabled == false)
                                {
                                    m_isCharacterWrappingEnabled = true;
                                }
                                else
                                    isLastBreakingChar = true;
                            }
                            #endregion

                            // Restore to previously stored state of last valid (space character or linefeed)
                            i = RestoreWordWrappingState(ref savedWordWrapState);
                            wrappingIndex = i;  // Used to detect when line length can no longer be reduced.

                            // Handling for Soft Hyphen
                            if (m_TextParsingBuffer[i].unicode == 0xAD) // && !m_isCharacterWrappingEnabled) // && ellipsisIndex != i && !m_isCharacterWrappingEnabled)
                            {
                                m_isTextTruncated = true;
                                m_TextParsingBuffer[i].unicode = 0x2D;
                                return CalculatePreferredValues(defaultFontSize, marginSize, true);
                            }

                            // Check if Line Spacing of previous line needs to be adjusted.
                            if (m_lineNumber > 0 && !TMP_Math.Approximately(m_maxLineAscender, m_startOfLineAscender) && m_lineHeight == TMP_Math.FLOAT_UNSET)
                            {
                                //Debug.Log("(1) Adjusting Line Spacing on line #" + m_lineNumber);
                                float offsetDelta = m_maxLineAscender - m_startOfLineAscender;
                                //AdjustLineOffset(m_firstCharacterOfLine, m_characterCount, offsetDelta);
                                m_lineOffset += offsetDelta;
                                savedWordWrapState.lineOffset = m_lineOffset;
                                savedWordWrapState.previousLineAscender = m_maxLineAscender;

                                // TODO - Add check for character exceeding vertical bounds
                            }
                            //m_isNewPage = false;

                            // Calculate lineAscender & make sure if last character is superscript or subscript that we check that as well.
                            float lineAscender = m_maxLineAscender - m_lineOffset;
                            float lineDescender = m_maxLineDescender - m_lineOffset;


                            // Update maxDescender and maxVisibleDescender
                            m_maxDescender = m_maxDescender < lineDescender ? m_maxDescender : lineDescender;


                            m_firstCharacterOfLine = m_characterCount; // Store first character of the next line.

                            // Compute Preferred Width & Height
                            renderedWidth += m_xAdvance;

                            if (m_enableWordWrapping)
                                renderedHeight = m_maxAscender - m_maxDescender;
                            else
                                renderedHeight = Mathf.Max(renderedHeight, lineAscender - lineDescender);


                            // Store the state of the line before starting on the new line.
                            SaveWordWrappingState(ref savedLineState, i, m_characterCount - 1);

                            m_lineNumber += 1;
                            //isStartOfNewLine = true;

                            // Check to make sure Array is large enough to hold a new line.
                            //if (m_lineNumber >= m_internalTextInfo.lineInfo.Length)
                            //    ResizeLineExtents(m_lineNumber);

                            // Apply Line Spacing based on scale of the last character of the line.
                            if (m_lineHeight == TMP_Math.FLOAT_UNSET)
                            {
                                float ascender = m_internalCharacterInfo[m_characterCount].ascender - m_internalCharacterInfo[m_characterCount].baseLine;
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
                        if (ignoreTextAutoSizing == false && defaultFontSize > m_fontSizeMin)
                        {
                            // Handle Character Width Adjustments
                            #region Character Width Adjustments
                            if (m_charWidthAdjDelta < m_charWidthMaxAdj / 100)
                            {
                                m_recursiveCount = 0;
                                m_charWidthAdjDelta += 0.01f;
                                return CalculatePreferredValues(defaultFontSize, marginSize, false);
                            }
                            #endregion

                            // Adjust Point Size
                            m_maxFontSize = defaultFontSize;

                            defaultFontSize -= Mathf.Max((defaultFontSize - m_minFontSize) / 2, 0.05f);
                            defaultFontSize = (int)(Mathf.Max(defaultFontSize, m_fontSizeMin) * 20 + 0.5f) / 20f;

                            if (m_recursiveCount > 20) return new Vector2(renderedWidth, renderedHeight);
                            return CalculatePreferredValues(defaultFontSize, marginSize, false);
                        }
                        #endregion End Text Auto-Sizing
                    }
                    #endregion End Check for Characters Exceeding Width of Text Container

                }
                #endregion Handle Visible Characters


                // Check if Line Spacing of previous line needs to be adjusted.
                #region Adjust Line Spacing
                if (m_lineNumber > 0 && !TMP_Math.Approximately(m_maxLineAscender, m_startOfLineAscender) && m_lineHeight == TMP_Math.FLOAT_UNSET && !m_isNewPage)
                {
                    //Debug.Log("Inline - Adjusting Line Spacing on line #" + m_lineNumber);
                    //float gap = 0; // Compute gap.

                    float offsetDelta = m_maxLineAscender - m_startOfLineAscender;
                    //AdjustLineOffset(m_firstCharacterOfLine, m_characterCount, offsetDelta);
                    elementDescenderII -= offsetDelta;
                    m_lineOffset += offsetDelta;

                    m_startOfLineAscender += offsetDelta;
                    savedWordWrapState.lineOffset = m_lineOffset;
                    savedWordWrapState.previousLineAscender = m_startOfLineAscender;
                }
                #endregion


                // Check if text Exceeds the vertical bounds of the margin area.
                #region Check Vertical Bounds & Auto-Sizing
                /*
                if (m_maxAscender - elementDescenderII > marginHeight + 0.0001f)
                {
                    // Handle Line spacing adjustments
                    #region Line Spacing Adjustments
                    if (m_enableAutoSizing && m_lineSpacingDelta > m_lineSpacingMax && m_lineNumber > 0)
                    {
                        //loopCountA = 0;

                        //m_lineSpacingDelta -= 1;
                        //GenerateTextMesh();
                        //return;
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
                        //if (loopCountA > 20) return; // Added to debug 
                        CalculatePreferredValues(m_fontSize, marginSize, false);
                        return Vector2.zero;
                    }
                    #endregion Text Auto-Sizing
                }
                */
                #endregion Check Vertical Bounds


                // Handle xAdvance & Tabulation Stops. Tab stops at every 25% of Font Size.
                #region XAdvance, Tabulation & Stops
                if (charCode == 9)
                {
                    float tabSize = m_currentFontAsset.faceInfo.tabWidth * currentElementScale;
                    float tabs = Mathf.Ceil(m_xAdvance / tabSize) * tabSize;
                    m_xAdvance = tabs > m_xAdvance ? tabs : m_xAdvance + tabSize;
                }
                else if (m_monoSpacing != 0)
                {
                    m_xAdvance += (m_monoSpacing - monoAdvance + ((characterSpacingAdjustment + m_currentFontAsset.normalSpacingOffset) * currentElementScale) + m_cSpacing) * (1 - m_charWidthAdjDelta);

                    if (char.IsWhiteSpace((char)charCode) || charCode == 0x200B)
                        m_xAdvance += m_wordSpacing * currentElementScale;
                }
                else
                {
                    m_xAdvance += ((m_cached_TextElement.glyph.metrics.horizontalAdvance * bold_xAdvance_multiplier + characterSpacingAdjustment + m_currentFontAsset.normalSpacingOffset + glyphAdjustments.xAdvance) * currentElementScale + m_cSpacing) * (1 - m_charWidthAdjDelta);

                    if (char.IsWhiteSpace((char)charCode) || charCode == 0x200B)
                        m_xAdvance += m_wordSpacing * currentElementScale;
                }


                #endregion Tabulation & Stops


                // Handle Carriage Return
                #region Carriage Return
                if (charCode == 13)
                {
                    maxXAdvance = Mathf.Max(maxXAdvance, renderedWidth + m_xAdvance);
                    renderedWidth = 0;
                    m_xAdvance = 0 + tag_Indent;
                }
                #endregion Carriage Return


                // Handle Line Spacing Adjustments + Word Wrapping & special case for last line.
                #region Check for Line Feed and Last Character
                if (charCode == 10 || m_characterCount == totalCharacterCount - 1)
                {
                    // Check if Line Spacing of previous line needs to be adjusted.
                    if (m_lineNumber > 0 && !TMP_Math.Approximately(m_maxLineAscender, m_startOfLineAscender) && m_lineHeight == TMP_Math.FLOAT_UNSET)
                    {
                        //Debug.Log("(2) Adjusting Line Spacing on line #" + m_lineNumber);
                        float offsetDelta = m_maxLineAscender - m_startOfLineAscender;
                        //AdjustLineOffset(m_firstCharacterOfLine, m_characterCount, offsetDelta);
                        elementDescenderII -= offsetDelta;
                        m_lineOffset += offsetDelta;
                    }

                    // Calculate lineAscender & make sure if last character is superscript or subscript that we check that as well.
                    //float lineAscender = m_maxLineAscender - m_lineOffset;
                    float lineDescender = m_maxLineDescender - m_lineOffset;

                    // Update maxDescender and maxVisibleDescender
                    m_maxDescender = m_maxDescender < lineDescender ? m_maxDescender : lineDescender;

                    m_firstCharacterOfLine = m_characterCount + 1;

                    // Store PreferredWidth paying attention to linefeed and last character of text.
                    if (charCode == 10 && m_characterCount != totalCharacterCount - 1)
                    {
                        maxXAdvance = Mathf.Max(maxXAdvance, renderedWidth + linebreakingWidth);
                        renderedWidth = 0;
                    }
                    else
                        renderedWidth = Mathf.Max(maxXAdvance, renderedWidth + linebreakingWidth);


                    renderedHeight = m_maxAscender - m_maxDescender;


                    // Add new line if not last lines or character.
                    if (charCode == 10)
                    {
                        // Store the state of the line before starting on the new line.
                        SaveWordWrappingState(ref savedLineState, i, m_characterCount);
                        // Store the state of the last Character before the new line.
                        SaveWordWrappingState(ref savedWordWrapState, i, m_characterCount);

                        m_lineNumber += 1;

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

                        m_characterCount += 1;
                        continue;
                    }
                }
                #endregion Check for Linefeed or Last Character


                // Save State of Mesh Creation for handling of Word Wrapping
                #region Save Word Wrapping State
                if (m_enableWordWrapping || m_overflowMode == TextOverflowModes.Truncate || m_overflowMode == TextOverflowModes.Ellipsis)
                {
                    if ((char.IsWhiteSpace((char)charCode) || charCode == 0x200B || charCode == 0x2D || charCode == 0xAD) && !m_isNonBreakingSpace && charCode != 0xA0 && charCode != 0x2011 && charCode != 0x202F && charCode != 0x2060)
                    {
                        // We store the state of numerous variables for the most recent Space, LineFeed or Carriage Return to enable them to be restored 
                        // for Word Wrapping.
                        SaveWordWrappingState(ref savedWordWrapState, i, m_characterCount);
                        m_isCharacterWrappingEnabled = false;
                        isFirstWord = false;
                    }
                    // Handling for East Asian languages
                    else if ((charCode > 0x1100 && charCode < 0x11ff || /* Hangul Jamo */
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
                            TMP_Settings.linebreakingRules.followingCharacters.ContainsKey(m_internalCharacterInfo[m_characterCount + 1].character) == false))
                        {
                            SaveWordWrappingState(ref savedWordWrapState, i, m_characterCount);
                            m_isCharacterWrappingEnabled = false;
                            isFirstWord = false;
                        }
                    }
                    else if ((isFirstWord || m_isCharacterWrappingEnabled == true || isLastBreakingChar))
                        SaveWordWrappingState(ref savedWordWrapState, i, m_characterCount);
                }
                #endregion Save Word Wrapping State

                m_characterCount += 1;
            }

            // Check Auto Sizing and increase font size to fill text container.
            #region Check Auto-Sizing (Upper Font Size Bounds)
            fontSizeDelta = m_maxFontSize - m_minFontSize;
            if (!m_isCharacterWrappingEnabled && ignoreTextAutoSizing == false && fontSizeDelta > 0.051f && defaultFontSize < m_fontSizeMax)
            {
                m_minFontSize = defaultFontSize;
                defaultFontSize += Mathf.Max((m_maxFontSize - defaultFontSize) / 2, 0.05f);
                defaultFontSize = (int)(Mathf.Min(defaultFontSize, m_fontSizeMax) * 20 + 0.5f) / 20f;

                if (m_recursiveCount > 20) return new Vector2(renderedWidth, renderedHeight);
                return CalculatePreferredValues(defaultFontSize, marginSize, false);
            }
            #endregion End Auto-sizing Check


            m_isCharacterWrappingEnabled = false;
            m_isCalculatingPreferredValues = false;

            // Adjust Preferred Width and Height to account for Margins.
            renderedWidth += m_margin.x > 0 ? m_margin.x : 0;
            renderedWidth += m_margin.z > 0 ? m_margin.z : 0;

            renderedHeight += m_margin.y > 0 ? m_margin.y : 0;
            renderedHeight += m_margin.w > 0 ? m_margin.w : 0;

            // Round Preferred Values to nearest 5/100.
            renderedWidth = (int)(renderedWidth * 100 + 1f) / 100f;
            renderedHeight = (int)(renderedHeight * 100 + 1f) / 100f;

            //Debug.Log("Preferred Values: (" + renderedWidth + ", " + renderedHeight + ") with Recursive count of " + m_recursiveCount);

            ////Profiler.EndSample();
            return new Vector2(renderedWidth, renderedHeight);
        }


        /// <summary>
        /// Method returning the compound bounds of the text object and child sub objects.
        /// </summary>
        /// <returns></returns>
        protected virtual Bounds GetCompoundBounds() { return new Bounds(); }


        /// <summary>
        /// Method which returns the bounds of the text object;
        /// </summary>
        /// <returns></returns>
        protected Bounds GetTextBounds()
        {
            if (m_textInfo == null || m_textInfo.characterCount > m_textInfo.characterInfo.Length) return new Bounds();

            Extents extent = new Extents(k_LargePositiveVector2, k_LargeNegativeVector2);

            for (int i = 0; i < m_textInfo.characterCount && i < m_textInfo.characterInfo.Length; i++)
            {
                if (!m_textInfo.characterInfo[i].isVisible) continue;

                extent.min.x = Mathf.Min(extent.min.x, m_textInfo.characterInfo[i].bottomLeft.x);
                extent.min.y = Mathf.Min(extent.min.y, m_textInfo.characterInfo[i].descender);

                extent.max.x = Mathf.Max(extent.max.x, m_textInfo.characterInfo[i].xAdvance);
                extent.max.y = Mathf.Max(extent.max.y, m_textInfo.characterInfo[i].ascender);
            }

            Vector2 size;
            size.x = extent.max.x - extent.min.x;
            size.y = extent.max.y - extent.min.y;

            Vector3 center = (extent.min + extent.max) / 2;

            return new Bounds(center, size);
        }


        /// <summary>
        /// Method which returns the bounds of the text object;
        /// </summary>
        /// <param name="onlyVisibleCharacters"></param>
        /// <returns></returns>
        protected Bounds GetTextBounds(bool onlyVisibleCharacters)
        {
            if (m_textInfo == null) return new Bounds();

            Extents extent = new Extents(k_LargePositiveVector2, k_LargeNegativeVector2);

            for (int i = 0; i < m_textInfo.characterCount; i++)
            {
                if ((i > maxVisibleCharacters || m_textInfo.characterInfo[i].lineNumber > m_maxVisibleLines) && onlyVisibleCharacters) break;

                if (onlyVisibleCharacters && !m_textInfo.characterInfo[i].isVisible) continue;

                extent.min.x = Mathf.Min(extent.min.x, m_textInfo.characterInfo[i].origin);
                extent.min.y = Mathf.Min(extent.min.y, m_textInfo.characterInfo[i].descender);

                extent.max.x = Mathf.Max(extent.max.x, m_textInfo.characterInfo[i].xAdvance);
                extent.max.y = Mathf.Max(extent.max.y, m_textInfo.characterInfo[i].ascender);
            }

            Vector2 size;
            size.x = extent.max.x - extent.min.x;
            size.y = extent.max.y - extent.min.y;

            Vector2 center = (extent.min + extent.max) / 2;

            return new Bounds(center, size);
        }


        /// <summary>
        /// Method to adjust line spacing as a result of using different fonts or font point size.
        /// </summary>
        /// <param name="startIndex"></param>
        /// <param name="endIndex"></param>
        /// <param name="offset"></param>
        protected virtual void AdjustLineOffset(int startIndex, int endIndex, float offset) { }


        /// <summary>
        /// Function to increase the size of the Line Extents Array.
        /// </summary>
        /// <param name="size"></param>
        protected void ResizeLineExtents(int size)
        {
            size = size > 1024 ? size + 256 : Mathf.NextPowerOfTwo(size + 1);

            TMP_LineInfo[] temp_lineInfo = new TMP_LineInfo[size];
            for (int i = 0; i < size; i++)
            {
                if (i < m_textInfo.lineInfo.Length)
                    temp_lineInfo[i] = m_textInfo.lineInfo[i];
                else
                {
                    temp_lineInfo[i].lineExtents.min = k_LargePositiveVector2;
                    temp_lineInfo[i].lineExtents.max = k_LargeNegativeVector2;

                    temp_lineInfo[i].ascender = k_LargeNegativeFloat;
                    temp_lineInfo[i].descender = k_LargePositiveFloat;
                }
            }

            m_textInfo.lineInfo = temp_lineInfo;
        }
        protected static Vector2 k_LargePositiveVector2 = new Vector2(TMP_Math.INT_MAX, TMP_Math.INT_MAX);
        protected static Vector2 k_LargeNegativeVector2 = new Vector2(TMP_Math.INT_MIN, TMP_Math.INT_MIN);
        protected static float k_LargePositiveFloat = TMP_Math.FLOAT_MAX;
        protected static float k_LargeNegativeFloat = TMP_Math.FLOAT_MIN;
        protected static int k_LargePositiveInt = TMP_Math.INT_MAX;
        protected static int k_LargeNegativeInt = TMP_Math.INT_MIN;

        /// <summary>
        /// Function used to evaluate the length of a text string.
        /// </summary>
        /// <param name="text"></param>
        /// <returns></returns>
        public virtual TMP_TextInfo GetTextInfo(string text) { return null; }


        /// <summary>
        /// Function to force an update of the margin size.
        /// </summary>
        public virtual void ComputeMarginSize() { }


        /// <summary>
        /// Function used in conjunction with GetTextInfo to figure out Array allocations.
        /// </summary>
        /// <param name="chars"></param>
        /// <returns></returns>
        //protected int GetArraySizes(int[] chars)
        //{
        //    //Debug.Log("Set Array Size called.");

        //    //int visibleCount = 0;
        //    //int totalCount = 0;
        //    int tagEnd = 0;

        //    m_totalCharacterCount = 0;
        //    m_isUsingBold = false;
        //    m_isParsingText = false;


        //    //m_VisibleCharacters.Clear();

        //    for (int i = 0; chars[i] != 0; i++)
        //    {
        //        int c = chars[i];

        //        if (m_isRichText && c == 60) // if Char '<'
        //        {
        //            // Check if Tag is Valid
        //            if (ValidateHtmlTag(chars, i + 1, out tagEnd))
        //            {
        //                i = tagEnd;
        //                //if ((m_style & FontStyles.Underline) == FontStyles.Underline) visibleCount += 3;

        //                if ((m_style & FontStyles.Bold) == FontStyles.Bold) m_isUsingBold = true;

        //                continue;
        //            }
        //        }

        //        //if (!char.IsWhiteSpace((char)c) && c != 0x200B)
        //        //{
        //            //visibleCount += 1;
        //        //}

        //        //m_VisibleCharacters.Add((char)c);
        //        m_totalCharacterCount += 1;
        //    }

        //    return m_totalCharacterCount;
        //}


        /// <summary>
        /// Save the State of various variables used in the mesh creation loop in conjunction with Word Wrapping
        /// </summary>
        /// <param name="state"></param>
        /// <param name="index"></param>
        /// <param name="count"></param>
        protected void SaveWordWrappingState(ref WordWrapState state, int index, int count)
        {
            // Multi Font & Material support related
            state.currentFontAsset = m_currentFontAsset;
            state.currentSpriteAsset = m_currentSpriteAsset;
            state.currentMaterial = m_currentMaterial;
            state.currentMaterialIndex = m_currentMaterialIndex;

            state.previous_WordBreak = index;
            state.total_CharacterCount = count;
            state.visible_CharacterCount = m_lineVisibleCharacterCount;
            //state.visible_CharacterCount = m_visibleCharacterCount;
            //state.visible_SpriteCount = m_visibleSpriteCount;
            state.visible_LinkCount = m_textInfo.linkCount;

            state.firstCharacterIndex = m_firstCharacterOfLine;
            state.firstVisibleCharacterIndex = m_firstVisibleCharacterOfLine;
            state.lastVisibleCharIndex = m_lastVisibleCharacterOfLine;

            state.fontStyle = m_FontStyleInternal;
            state.fontScale = m_fontScale;
            //state.maxFontScale = m_maxFontScale;
            state.fontScaleMultiplier = m_fontScaleMultiplier;
            state.currentFontSize = m_currentFontSize;

            state.xAdvance = m_xAdvance;
            state.maxCapHeight = m_maxCapHeight;
            state.maxAscender = m_maxAscender;
            state.maxDescender = m_maxDescender;
            state.maxLineAscender = m_maxLineAscender;
            state.maxLineDescender = m_maxLineDescender;
            state.previousLineAscender = m_startOfLineAscender;
            state.preferredWidth = m_preferredWidth;
            state.preferredHeight = m_preferredHeight;
            state.meshExtents = m_meshExtents;

            state.lineNumber = m_lineNumber;
            state.lineOffset = m_lineOffset;
            state.baselineOffset = m_baselineOffset;

            //state.alignment = m_lineJustification;
            state.vertexColor = m_htmlColor;
            state.underlineColor = m_underlineColor;
            state.strikethroughColor = m_strikethroughColor;
            state.highlightColor = m_highlightColor;

            state.isNonBreakingSpace = m_isNonBreakingSpace;
            state.tagNoParsing = tag_NoParsing;

            // XML Tag Stack
            state.basicStyleStack = m_fontStyleStack;
            state.colorStack = m_colorStack;
            state.underlineColorStack = m_underlineColorStack;
            state.strikethroughColorStack = m_strikethroughColorStack;
            state.highlightColorStack = m_highlightColorStack;
            state.colorGradientStack = m_colorGradientStack;
            state.sizeStack = m_sizeStack;
            state.indentStack = m_indentStack;
            state.fontWeightStack = m_FontWeightStack;
            state.styleStack = m_styleStack;
            state.baselineStack = m_baselineOffsetStack;
            state.actionStack = m_actionStack;
            state.materialReferenceStack = m_materialReferenceStack;
            state.lineJustificationStack = m_lineJustificationStack;
            //state.spriteAnimationStack = m_spriteAnimationStack;

            state.spriteAnimationID = m_spriteAnimationID;

            if (m_lineNumber < m_textInfo.lineInfo.Length)
                state.lineInfo = m_textInfo.lineInfo[m_lineNumber];
        }


        /// <summary>
        /// Restore the State of various variables used in the mesh creation loop.
        /// </summary>
        /// <param name="state"></param>
        /// <returns></returns>
        protected int RestoreWordWrappingState(ref WordWrapState state)
        {
            int index = state.previous_WordBreak;

            // Multi Font & Material support related
            m_currentFontAsset = state.currentFontAsset;
            m_currentSpriteAsset = state.currentSpriteAsset;
            m_currentMaterial = state.currentMaterial;
            m_currentMaterialIndex = state.currentMaterialIndex;

            m_characterCount = state.total_CharacterCount + 1;
            m_lineVisibleCharacterCount = state.visible_CharacterCount;
            //m_visibleCharacterCount = state.visible_CharacterCount;
            //m_visibleSpriteCount = state.visible_SpriteCount;
            m_textInfo.linkCount = state.visible_LinkCount;

            m_firstCharacterOfLine = state.firstCharacterIndex;
            m_firstVisibleCharacterOfLine = state.firstVisibleCharacterIndex;
            m_lastVisibleCharacterOfLine = state.lastVisibleCharIndex;

            m_FontStyleInternal = state.fontStyle;
            m_fontScale = state.fontScale;
            m_fontScaleMultiplier = state.fontScaleMultiplier;
            //m_maxFontScale = state.maxFontScale;
            m_currentFontSize = state.currentFontSize;

            m_xAdvance = state.xAdvance;
            m_maxCapHeight = state.maxCapHeight;
            m_maxAscender = state.maxAscender;
            m_maxDescender = state.maxDescender;
            m_maxLineAscender = state.maxLineAscender;
            m_maxLineDescender = state.maxLineDescender;
            m_startOfLineAscender = state.previousLineAscender;
            m_preferredWidth = state.preferredWidth;
            m_preferredHeight = state.preferredHeight;
            m_meshExtents = state.meshExtents;

            m_lineNumber = state.lineNumber;
            m_lineOffset = state.lineOffset;
            m_baselineOffset = state.baselineOffset;

            //m_lineJustification = state.alignment;
            m_htmlColor = state.vertexColor;
            m_underlineColor = state.underlineColor;
            m_strikethroughColor = state.strikethroughColor;
            m_highlightColor = state.highlightColor;

            m_isNonBreakingSpace = state.isNonBreakingSpace;
            tag_NoParsing = state.tagNoParsing;

            // XML Tag Stack
            m_fontStyleStack = state.basicStyleStack;
            m_colorStack = state.colorStack;
            m_underlineColorStack = state.underlineColorStack;
            m_strikethroughColorStack = state.strikethroughColorStack;
            m_highlightColorStack = state.highlightColorStack;
            m_colorGradientStack = state.colorGradientStack;
            m_sizeStack = state.sizeStack;
            m_indentStack = state.indentStack;
            m_FontWeightStack = state.fontWeightStack;
            m_styleStack = state.styleStack;
            m_baselineOffsetStack = state.baselineStack;
            m_actionStack = state.actionStack;
            m_materialReferenceStack = state.materialReferenceStack;
            m_lineJustificationStack = state.lineJustificationStack;
            //m_spriteAnimationStack = state.spriteAnimationStack;

            m_spriteAnimationID = state.spriteAnimationID;

            if (m_lineNumber < m_textInfo.lineInfo.Length)
                m_textInfo.lineInfo[m_lineNumber] = state.lineInfo;

            return index;
        }


        /// <summary>
        /// Store vertex information for each character.
        /// </summary>
        /// <param name="style_padding">Style_padding.</param>
        /// <param name="vertexColor">Vertex color.</param>
        protected virtual void SaveGlyphVertexInfo(float padding, float style_padding, Color32 vertexColor)
        {
            // Save the Vertex Position for the Character
            #region Setup Mesh Vertices
            m_textInfo.characterInfo[m_characterCount].vertex_BL.position = m_textInfo.characterInfo[m_characterCount].bottomLeft;
            m_textInfo.characterInfo[m_characterCount].vertex_TL.position = m_textInfo.characterInfo[m_characterCount].topLeft;
            m_textInfo.characterInfo[m_characterCount].vertex_TR.position = m_textInfo.characterInfo[m_characterCount].topRight;
            m_textInfo.characterInfo[m_characterCount].vertex_BR.position = m_textInfo.characterInfo[m_characterCount].bottomRight;
            #endregion


            #region Setup Vertex Colors
            // Alpha is the lower of the vertex color or tag color alpha used.
            vertexColor.a = m_fontColor32.a < vertexColor.a ? (byte)(m_fontColor32.a) : (byte)(vertexColor.a);

            // Handle Vertex Colors & Vertex Color Gradient
            if (!m_enableVertexGradient)
            {
                m_textInfo.characterInfo[m_characterCount].vertex_BL.color = vertexColor;
                m_textInfo.characterInfo[m_characterCount].vertex_TL.color = vertexColor;
                m_textInfo.characterInfo[m_characterCount].vertex_TR.color = vertexColor;
                m_textInfo.characterInfo[m_characterCount].vertex_BR.color = vertexColor;
            }
            else
            {
                if (!m_overrideHtmlColors && m_colorStack.m_Index > 1)
                {
                    m_textInfo.characterInfo[m_characterCount].vertex_BL.color = vertexColor;
                    m_textInfo.characterInfo[m_characterCount].vertex_TL.color = vertexColor;
                    m_textInfo.characterInfo[m_characterCount].vertex_TR.color = vertexColor;
                    m_textInfo.characterInfo[m_characterCount].vertex_BR.color = vertexColor;
                }
                else // Handle Vertex Color Gradient
                {
                    // Use Vertex Color Gradient Preset (if one is assigned)
                    if (m_fontColorGradientPreset != null)
                    {
                        m_textInfo.characterInfo[m_characterCount].vertex_BL.color = m_fontColorGradientPreset.bottomLeft * vertexColor;
                        m_textInfo.characterInfo[m_characterCount].vertex_TL.color = m_fontColorGradientPreset.topLeft * vertexColor;
                        m_textInfo.characterInfo[m_characterCount].vertex_TR.color = m_fontColorGradientPreset.topRight * vertexColor;
                        m_textInfo.characterInfo[m_characterCount].vertex_BR.color = m_fontColorGradientPreset.bottomRight * vertexColor;
                    }
                    else
                    {
                        m_textInfo.characterInfo[m_characterCount].vertex_BL.color = m_fontColorGradient.bottomLeft * vertexColor;
                        m_textInfo.characterInfo[m_characterCount].vertex_TL.color = m_fontColorGradient.topLeft * vertexColor;
                        m_textInfo.characterInfo[m_characterCount].vertex_TR.color = m_fontColorGradient.topRight * vertexColor;
                        m_textInfo.characterInfo[m_characterCount].vertex_BR.color = m_fontColorGradient.bottomRight * vertexColor;
                    }
                }
            }

            if (m_colorGradientPreset != null)
            {
                m_textInfo.characterInfo[m_characterCount].vertex_BL.color *= m_colorGradientPreset.bottomLeft;
                m_textInfo.characterInfo[m_characterCount].vertex_TL.color *= m_colorGradientPreset.topLeft;
                m_textInfo.characterInfo[m_characterCount].vertex_TR.color *= m_colorGradientPreset.topRight;
                m_textInfo.characterInfo[m_characterCount].vertex_BR.color *= m_colorGradientPreset.bottomRight;
            }
            #endregion

            // Apply style_padding only if this is a SDF Shader.
            if (!m_isSDFShader)
                style_padding = 0f;


            // Setup UVs for the Character
            #region Setup UVs

            Vector2 uv0;
            uv0.x = (m_cached_TextElement.glyph.glyphRect.x - padding - style_padding) / m_currentFontAsset.atlasWidth;
            uv0.y = (m_cached_TextElement.glyph.glyphRect.y - padding - style_padding) / m_currentFontAsset.atlasHeight;

            Vector2 uv1;
            uv1.x = uv0.x;
            uv1.y = (m_cached_TextElement.glyph.glyphRect.y + padding + style_padding + m_cached_TextElement.glyph.glyphRect.height) / m_currentFontAsset.atlasHeight;

            Vector2 uv2;
            uv2.x = (m_cached_TextElement.glyph.glyphRect.x + padding + style_padding + m_cached_TextElement.glyph.glyphRect.width) / m_currentFontAsset.atlasWidth;
            uv2.y = uv1.y;

            Vector2 uv3;
            uv3.x = uv2.x;
            uv3.y = uv0.y;

            // Store UV Information
            m_textInfo.characterInfo[m_characterCount].vertex_BL.uv = uv0;
            m_textInfo.characterInfo[m_characterCount].vertex_TL.uv = uv1;
            m_textInfo.characterInfo[m_characterCount].vertex_TR.uv = uv2;
            m_textInfo.characterInfo[m_characterCount].vertex_BR.uv = uv3;
            #endregion Setup UVs


            // Normal
            #region Setup Normals & Tangents
            //Vector3 normal = new Vector3(0, 0, -1);
            //m_textInfo.characterInfo[m_characterCount].vertex_BL.normal = normal;
            //m_textInfo.characterInfo[m_characterCount].vertex_TL.normal = normal;
            //m_textInfo.characterInfo[m_characterCount].vertex_TR.normal = normal;
            //m_textInfo.characterInfo[m_characterCount].vertex_BR.normal = normal;

            // Tangents
            //Vector4 tangent = new Vector4(-1, 0, 0, 1);
            //m_textInfo.characterInfo[m_characterCount].vertex_BL.tangent = tangent;
            //m_textInfo.characterInfo[m_characterCount].vertex_TL.tangent = tangent;
            //m_textInfo.characterInfo[m_characterCount].vertex_TR.tangent = tangent;
            //m_textInfo.characterInfo[m_characterCount].vertex_BR.tangent = tangent;
            #endregion end Normals & Tangents
        }


        /// <summary>
        /// Store vertex information for each sprite.
        /// </summary>
        /// <param name="padding"></param>
        /// <param name="style_padding"></param>
        /// <param name="vertexColor"></param>
        protected virtual void SaveSpriteVertexInfo(Color32 vertexColor)
        {
            // Save the Vertex Position for the Character
            #region Setup Mesh Vertices
            m_textInfo.characterInfo[m_characterCount].vertex_BL.position = m_textInfo.characterInfo[m_characterCount].bottomLeft;
            m_textInfo.characterInfo[m_characterCount].vertex_TL.position = m_textInfo.characterInfo[m_characterCount].topLeft;
            m_textInfo.characterInfo[m_characterCount].vertex_TR.position = m_textInfo.characterInfo[m_characterCount].topRight;
            m_textInfo.characterInfo[m_characterCount].vertex_BR.position = m_textInfo.characterInfo[m_characterCount].bottomRight;
            #endregion

            // Vertex Color Alpha
            if (m_tintAllSprites) m_tintSprite = true;
            Color32 spriteColor = m_tintSprite ? m_spriteColor.Multiply(vertexColor) : m_spriteColor;
            spriteColor.a = spriteColor.a < m_fontColor32.a ? spriteColor.a = spriteColor.a < vertexColor.a ? spriteColor.a : vertexColor.a : m_fontColor32.a;

            Color32 c0 = spriteColor;
            Color32 c1 = spriteColor;
            Color32 c2 = spriteColor;
            Color32 c3 = spriteColor;

            if (m_enableVertexGradient)
            {
                if (m_fontColorGradientPreset != null)
                {
                    c0 = m_tintSprite ? c0.Multiply(m_fontColorGradientPreset.bottomLeft) : c0;
                    c1 = m_tintSprite ? c1.Multiply(m_fontColorGradientPreset.topLeft) : c1;
                    c2 = m_tintSprite ? c2.Multiply(m_fontColorGradientPreset.topRight) : c2;
                    c3 = m_tintSprite ? c3.Multiply(m_fontColorGradientPreset.bottomRight) : c3;
                }
                else
                {
                    c0 = m_tintSprite ? c0.Multiply(m_fontColorGradient.bottomLeft) : c0;
                    c1 = m_tintSprite ? c1.Multiply(m_fontColorGradient.topLeft) : c1;
                    c2 = m_tintSprite ? c2.Multiply(m_fontColorGradient.topRight) : c2;
                    c3 = m_tintSprite ? c3.Multiply(m_fontColorGradient.bottomRight) : c3;
                }
            }

            if (m_colorGradientPreset != null)
            {
                c0 = m_tintSprite ? c0.Multiply(m_colorGradientPreset.bottomLeft) : c0;
                c1 = m_tintSprite ? c1.Multiply(m_colorGradientPreset.topLeft) : c1;
                c2 = m_tintSprite ? c2.Multiply(m_colorGradientPreset.topRight) : c2;
                c3 = m_tintSprite ? c3.Multiply(m_colorGradientPreset.bottomRight) : c3;
            }

            m_textInfo.characterInfo[m_characterCount].vertex_BL.color = c0;
            m_textInfo.characterInfo[m_characterCount].vertex_TL.color = c1;
            m_textInfo.characterInfo[m_characterCount].vertex_TR.color = c2;
            m_textInfo.characterInfo[m_characterCount].vertex_BR.color = c3;


            // Setup UVs for the Character
            #region Setup UVs
            Vector2 uv0 = new Vector2((float)m_cached_TextElement.glyph.glyphRect.x / m_currentSpriteAsset.spriteSheet.width, (float)m_cached_TextElement.glyph.glyphRect.y / m_currentSpriteAsset.spriteSheet.height);  // bottom left
            Vector2 uv1 = new Vector2(uv0.x, (float)(m_cached_TextElement.glyph.glyphRect.y + m_cached_TextElement.glyph.glyphRect.height) / m_currentSpriteAsset.spriteSheet.height);  // top left
            Vector2 uv2 = new Vector2((float)(m_cached_TextElement.glyph.glyphRect.x + m_cached_TextElement.glyph.glyphRect.width) / m_currentSpriteAsset.spriteSheet.width, uv1.y); // top right
            Vector2 uv3 = new Vector2(uv2.x, uv0.y); // bottom right

            // Store UV Information
            m_textInfo.characterInfo[m_characterCount].vertex_BL.uv = uv0;
            m_textInfo.characterInfo[m_characterCount].vertex_TL.uv = uv1;
            m_textInfo.characterInfo[m_characterCount].vertex_TR.uv = uv2;
            m_textInfo.characterInfo[m_characterCount].vertex_BR.uv = uv3;
            #endregion Setup UVs


            // Normal
            #region Setup Normals & Tangents
            //Vector3 normal = new Vector3(0, 0, -1);
            //m_textInfo.characterInfo[m_characterCount].vertex_BL.normal = normal;
            //m_textInfo.characterInfo[m_characterCount].vertex_TL.normal = normal;
            //m_textInfo.characterInfo[m_characterCount].vertex_TR.normal = normal;
            //m_textInfo.characterInfo[m_characterCount].vertex_BR.normal = normal;

            // Tangents
            //Vector4 tangent = new Vector4(-1, 0, 0, 1);
            //m_textInfo.characterInfo[m_characterCount].vertex_BL.tangent = tangent;
            //m_textInfo.characterInfo[m_characterCount].vertex_TL.tangent = tangent;
            //m_textInfo.characterInfo[m_characterCount].vertex_TR.tangent = tangent;
            //m_textInfo.characterInfo[m_characterCount].vertex_BR.tangent = tangent;
            #endregion end Normals & Tangents

        }


        /// <summary>
        /// Store vertex attributes into the appropriate TMP_MeshInfo.
        /// </summary>
        /// <param name="i"></param>
        /// <param name="index_X4"></param>
        protected virtual void FillCharacterVertexBuffers(int i, int index_X4)
        {
            int materialIndex = m_textInfo.characterInfo[i].materialReferenceIndex;
            index_X4 = m_textInfo.meshInfo[materialIndex].vertexCount;

            // Make sure buffers allocation are sufficient to hold the vertex data
            //if (m_textInfo.meshInfo[materialIndex].vertices.Length < index_X4 + 4)
            //    m_textInfo.meshInfo[materialIndex].ResizeMeshInfo(Mathf.NextPowerOfTwo(index_X4 + 4));


            TMP_CharacterInfo[] characterInfoArray = m_textInfo.characterInfo;
            m_textInfo.characterInfo[i].vertexIndex = index_X4;

            // Setup Vertices for Characters
            m_textInfo.meshInfo[materialIndex].vertices[0 + index_X4] = characterInfoArray[i].vertex_BL.position;
            m_textInfo.meshInfo[materialIndex].vertices[1 + index_X4] = characterInfoArray[i].vertex_TL.position;
            m_textInfo.meshInfo[materialIndex].vertices[2 + index_X4] = characterInfoArray[i].vertex_TR.position;
            m_textInfo.meshInfo[materialIndex].vertices[3 + index_X4] = characterInfoArray[i].vertex_BR.position;


            // Setup UVS0
            m_textInfo.meshInfo[materialIndex].uvs0[0 + index_X4] = characterInfoArray[i].vertex_BL.uv;
            m_textInfo.meshInfo[materialIndex].uvs0[1 + index_X4] = characterInfoArray[i].vertex_TL.uv;
            m_textInfo.meshInfo[materialIndex].uvs0[2 + index_X4] = characterInfoArray[i].vertex_TR.uv;
            m_textInfo.meshInfo[materialIndex].uvs0[3 + index_X4] = characterInfoArray[i].vertex_BR.uv;


            // Setup UVS2
            m_textInfo.meshInfo[materialIndex].uvs2[0 + index_X4] = characterInfoArray[i].vertex_BL.uv2;
            m_textInfo.meshInfo[materialIndex].uvs2[1 + index_X4] = characterInfoArray[i].vertex_TL.uv2;
            m_textInfo.meshInfo[materialIndex].uvs2[2 + index_X4] = characterInfoArray[i].vertex_TR.uv2;
            m_textInfo.meshInfo[materialIndex].uvs2[3 + index_X4] = characterInfoArray[i].vertex_BR.uv2;


            // Setup UVS4
            //m_textInfo.meshInfo[0].uvs4[0 + index_X4] = characterInfoArray[i].vertex_BL.uv4;
            //m_textInfo.meshInfo[0].uvs4[1 + index_X4] = characterInfoArray[i].vertex_TL.uv4;
            //m_textInfo.meshInfo[0].uvs4[2 + index_X4] = characterInfoArray[i].vertex_TR.uv4;
            //m_textInfo.meshInfo[0].uvs4[3 + index_X4] = characterInfoArray[i].vertex_BR.uv4;


            // setup Vertex Colors
            m_textInfo.meshInfo[materialIndex].colors32[0 + index_X4] = characterInfoArray[i].vertex_BL.color;
            m_textInfo.meshInfo[materialIndex].colors32[1 + index_X4] = characterInfoArray[i].vertex_TL.color;
            m_textInfo.meshInfo[materialIndex].colors32[2 + index_X4] = characterInfoArray[i].vertex_TR.color;
            m_textInfo.meshInfo[materialIndex].colors32[3 + index_X4] = characterInfoArray[i].vertex_BR.color;

            m_textInfo.meshInfo[materialIndex].vertexCount = index_X4 + 4;
        }


        protected virtual void FillCharacterVertexBuffers(int i, int index_X4, bool isVolumetric)
        {
            int materialIndex = m_textInfo.characterInfo[i].materialReferenceIndex;
            index_X4 = m_textInfo.meshInfo[materialIndex].vertexCount;

            TMP_CharacterInfo[] characterInfoArray = m_textInfo.characterInfo;
            m_textInfo.characterInfo[i].vertexIndex = index_X4;

            // Setup Vertices for Characters
            m_textInfo.meshInfo[materialIndex].vertices[0 + index_X4] = characterInfoArray[i].vertex_BL.position;
            m_textInfo.meshInfo[materialIndex].vertices[1 + index_X4] = characterInfoArray[i].vertex_TL.position;
            m_textInfo.meshInfo[materialIndex].vertices[2 + index_X4] = characterInfoArray[i].vertex_TR.position;
            m_textInfo.meshInfo[materialIndex].vertices[3 + index_X4] = characterInfoArray[i].vertex_BR.position;

            if (isVolumetric)
            {
                Vector3 depth = new Vector3(0, 0, m_fontSize * m_fontScale);
                m_textInfo.meshInfo[materialIndex].vertices[4 + index_X4] = characterInfoArray[i].vertex_BL.position + depth;
                m_textInfo.meshInfo[materialIndex].vertices[5 + index_X4] = characterInfoArray[i].vertex_TL.position + depth;
                m_textInfo.meshInfo[materialIndex].vertices[6 + index_X4] = characterInfoArray[i].vertex_TR.position + depth;
                m_textInfo.meshInfo[materialIndex].vertices[7 + index_X4] = characterInfoArray[i].vertex_BR.position + depth;
            }

            // Setup UVS0
            m_textInfo.meshInfo[materialIndex].uvs0[0 + index_X4] = characterInfoArray[i].vertex_BL.uv;
            m_textInfo.meshInfo[materialIndex].uvs0[1 + index_X4] = characterInfoArray[i].vertex_TL.uv;
            m_textInfo.meshInfo[materialIndex].uvs0[2 + index_X4] = characterInfoArray[i].vertex_TR.uv;
            m_textInfo.meshInfo[materialIndex].uvs0[3 + index_X4] = characterInfoArray[i].vertex_BR.uv;

            if (isVolumetric)
            {
                m_textInfo.meshInfo[materialIndex].uvs0[4 + index_X4] = characterInfoArray[i].vertex_BL.uv;
                m_textInfo.meshInfo[materialIndex].uvs0[5 + index_X4] = characterInfoArray[i].vertex_TL.uv;
                m_textInfo.meshInfo[materialIndex].uvs0[6 + index_X4] = characterInfoArray[i].vertex_TR.uv;
                m_textInfo.meshInfo[materialIndex].uvs0[7 + index_X4] = characterInfoArray[i].vertex_BR.uv;
            }


            // Setup UVS2
            m_textInfo.meshInfo[materialIndex].uvs2[0 + index_X4] = characterInfoArray[i].vertex_BL.uv2;
            m_textInfo.meshInfo[materialIndex].uvs2[1 + index_X4] = characterInfoArray[i].vertex_TL.uv2;
            m_textInfo.meshInfo[materialIndex].uvs2[2 + index_X4] = characterInfoArray[i].vertex_TR.uv2;
            m_textInfo.meshInfo[materialIndex].uvs2[3 + index_X4] = characterInfoArray[i].vertex_BR.uv2;

            if (isVolumetric)
            {
                m_textInfo.meshInfo[materialIndex].uvs2[4 + index_X4] = characterInfoArray[i].vertex_BL.uv2;
                m_textInfo.meshInfo[materialIndex].uvs2[5 + index_X4] = characterInfoArray[i].vertex_TL.uv2;
                m_textInfo.meshInfo[materialIndex].uvs2[6 + index_X4] = characterInfoArray[i].vertex_TR.uv2;
                m_textInfo.meshInfo[materialIndex].uvs2[7 + index_X4] = characterInfoArray[i].vertex_BR.uv2;
            }


            // Setup UVS4
            //m_textInfo.meshInfo[0].uvs4[0 + index_X4] = characterInfoArray[i].vertex_BL.uv4;
            //m_textInfo.meshInfo[0].uvs4[1 + index_X4] = characterInfoArray[i].vertex_TL.uv4;
            //m_textInfo.meshInfo[0].uvs4[2 + index_X4] = characterInfoArray[i].vertex_TR.uv4;
            //m_textInfo.meshInfo[0].uvs4[3 + index_X4] = characterInfoArray[i].vertex_BR.uv4;


            // setup Vertex Colors
            m_textInfo.meshInfo[materialIndex].colors32[0 + index_X4] = characterInfoArray[i].vertex_BL.color;
            m_textInfo.meshInfo[materialIndex].colors32[1 + index_X4] = characterInfoArray[i].vertex_TL.color;
            m_textInfo.meshInfo[materialIndex].colors32[2 + index_X4] = characterInfoArray[i].vertex_TR.color;
            m_textInfo.meshInfo[materialIndex].colors32[3 + index_X4] = characterInfoArray[i].vertex_BR.color;

            if (isVolumetric)
            {
                Color32 backColor = new Color32(255, 255, 128, 255);
                m_textInfo.meshInfo[materialIndex].colors32[4 + index_X4] = backColor; //characterInfoArray[i].vertex_BL.color;
                m_textInfo.meshInfo[materialIndex].colors32[5 + index_X4] = backColor; //characterInfoArray[i].vertex_TL.color;
                m_textInfo.meshInfo[materialIndex].colors32[6 + index_X4] = backColor; //characterInfoArray[i].vertex_TR.color;
                m_textInfo.meshInfo[materialIndex].colors32[7 + index_X4] = backColor; //characterInfoArray[i].vertex_BR.color;
            }

            m_textInfo.meshInfo[materialIndex].vertexCount = index_X4 + (!isVolumetric ? 4 : 8);
        }


        /// <summary>
        /// Fill Vertex Buffers for Sprites
        /// </summary>
        /// <param name="i"></param>
        /// <param name="spriteIndex_X4"></param>
        protected virtual void FillSpriteVertexBuffers(int i, int index_X4)
        {
            int materialIndex = m_textInfo.characterInfo[i].materialReferenceIndex;
            index_X4 = m_textInfo.meshInfo[materialIndex].vertexCount;

            TMP_CharacterInfo[] characterInfoArray = m_textInfo.characterInfo;
            m_textInfo.characterInfo[i].vertexIndex = index_X4;

            // Setup Vertices for Characters
            m_textInfo.meshInfo[materialIndex].vertices[0 + index_X4] = characterInfoArray[i].vertex_BL.position;
            m_textInfo.meshInfo[materialIndex].vertices[1 + index_X4] = characterInfoArray[i].vertex_TL.position;
            m_textInfo.meshInfo[materialIndex].vertices[2 + index_X4] = characterInfoArray[i].vertex_TR.position;
            m_textInfo.meshInfo[materialIndex].vertices[3 + index_X4] = characterInfoArray[i].vertex_BR.position;


            // Setup UVS0
            m_textInfo.meshInfo[materialIndex].uvs0[0 + index_X4] = characterInfoArray[i].vertex_BL.uv;
            m_textInfo.meshInfo[materialIndex].uvs0[1 + index_X4] = characterInfoArray[i].vertex_TL.uv;
            m_textInfo.meshInfo[materialIndex].uvs0[2 + index_X4] = characterInfoArray[i].vertex_TR.uv;
            m_textInfo.meshInfo[materialIndex].uvs0[3 + index_X4] = characterInfoArray[i].vertex_BR.uv;


            // Setup UVS2
            m_textInfo.meshInfo[materialIndex].uvs2[0 + index_X4] = characterInfoArray[i].vertex_BL.uv2;
            m_textInfo.meshInfo[materialIndex].uvs2[1 + index_X4] = characterInfoArray[i].vertex_TL.uv2;
            m_textInfo.meshInfo[materialIndex].uvs2[2 + index_X4] = characterInfoArray[i].vertex_TR.uv2;
            m_textInfo.meshInfo[materialIndex].uvs2[3 + index_X4] = characterInfoArray[i].vertex_BR.uv2;


            // Setup UVS4
            //m_textInfo.meshInfo[0].uvs4[0 + index_X4] = characterInfoArray[i].vertex_BL.uv4;
            //m_textInfo.meshInfo[0].uvs4[1 + index_X4] = characterInfoArray[i].vertex_TL.uv4;
            //m_textInfo.meshInfo[0].uvs4[2 + index_X4] = characterInfoArray[i].vertex_TR.uv4;
            //m_textInfo.meshInfo[0].uvs4[3 + index_X4] = characterInfoArray[i].vertex_BR.uv4;


            // setup Vertex Colors
            m_textInfo.meshInfo[materialIndex].colors32[0 + index_X4] = characterInfoArray[i].vertex_BL.color;
            m_textInfo.meshInfo[materialIndex].colors32[1 + index_X4] = characterInfoArray[i].vertex_TL.color;
            m_textInfo.meshInfo[materialIndex].colors32[2 + index_X4] = characterInfoArray[i].vertex_TR.color;
            m_textInfo.meshInfo[materialIndex].colors32[3 + index_X4] = characterInfoArray[i].vertex_BR.color;

            m_textInfo.meshInfo[materialIndex].vertexCount = index_X4 + 4;
        }


        /// <summary>
        /// Method to add the underline geometry.
        /// </summary>
        /// <param name="start"></param>
        /// <param name="end"></param>
        /// <param name="startScale"></param>
        /// <param name="endScale"></param>
        /// <param name="maxScale"></param>
        /// <param name="underlineColor"></param>
        protected virtual void DrawUnderlineMesh(Vector3 start, Vector3 end, ref int index, float startScale, float endScale, float maxScale, float sdfScale, Color32 underlineColor)
        {
            if (m_cached_Underline_Character == null)
            {
                if (!TMP_Settings.warningsDisabled)
                    Debug.LogWarning("Unable to add underline since the Font Asset doesn't contain the underline character.", this);

                return;
            }

            int verticesCount = index + 12;
            // Check to make sure our current mesh buffer allocations can hold these new Quads.
            if (verticesCount > m_textInfo.meshInfo[0].vertices.Length)
            {
                // Resize Mesh Buffers
                m_textInfo.meshInfo[0].ResizeMeshInfo(verticesCount / 4);
            }

            // Adjust the position of the underline based on the lowest character. This matters for subscript character.
            start.y = Mathf.Min(start.y, end.y);
            end.y = Mathf.Min(start.y, end.y);

            float segmentWidth = m_cached_Underline_Character.glyph.metrics.width / 2 * maxScale;

            if (end.x - start.x < m_cached_Underline_Character.glyph.metrics.width * maxScale)
            {
                segmentWidth = (end.x - start.x) / 2f;
            }

            float startPadding = m_padding * startScale / maxScale;
            float endPadding = m_padding * endScale / maxScale;

            float underlineThickness = m_fontAsset.faceInfo.underlineThickness;

            // UNDERLINE VERTICES FOR (3) LINE SEGMENTS
            #region UNDERLINE VERTICES
            Vector3[] vertices = m_textInfo.meshInfo[0].vertices;

            // Front Part of the Underline
            vertices[index + 0] = start + new Vector3(0, 0 - (underlineThickness + m_padding) * maxScale, 0); // BL
            vertices[index + 1] = start + new Vector3(0, m_padding * maxScale, 0); // TL
            vertices[index + 2] = vertices[index + 1] + new Vector3(segmentWidth, 0, 0); // TR
            vertices[index + 3] = vertices[index + 0] + new Vector3(segmentWidth, 0, 0); // BR

            // Middle Part of the Underline
            vertices[index + 4] = vertices[index + 3]; // BL
            vertices[index + 5] = vertices[index + 2]; // TL
            vertices[index + 6] = end + new Vector3(-segmentWidth, m_padding * maxScale, 0);  // TR
            vertices[index + 7] = end + new Vector3(-segmentWidth, -(underlineThickness + m_padding) * maxScale, 0); // BR

            // End Part of the Underline
            vertices[index + 8] = vertices[index + 7]; // BL
            vertices[index + 9] = vertices[index + 6]; // TL
            vertices[index + 10] = end + new Vector3(0, m_padding * maxScale, 0); // TR
            vertices[index + 11] = end + new Vector3(0, -(underlineThickness + m_padding) * maxScale, 0); // BR
            #endregion

            // UNDERLINE UV0
            #region HANDLE UV0
            Vector2[] uvs0 = m_textInfo.meshInfo[0].uvs0;

            // Calculate UV required to setup the 3 Quads for the Underline.
            Vector2 uv0 = new Vector2((m_cached_Underline_Character.glyph.glyphRect.x - startPadding) / m_fontAsset.atlasWidth, (m_cached_Underline_Character.glyph.glyphRect.y - m_padding) / m_fontAsset.atlasHeight);  // bottom left
            Vector2 uv1 = new Vector2(uv0.x, (m_cached_Underline_Character.glyph.glyphRect.y + m_cached_Underline_Character.glyph.glyphRect.height + m_padding) / m_fontAsset.atlasHeight);  // top left
            Vector2 uv2 = new Vector2((m_cached_Underline_Character.glyph.glyphRect.x - startPadding + (float)m_cached_Underline_Character.glyph.glyphRect.width / 2) / m_fontAsset.atlasWidth, uv1.y); // Mid Top Left
            Vector2 uv3 = new Vector2(uv2.x, uv0.y); // Mid Bottom Left
            Vector2 uv4 = new Vector2((m_cached_Underline_Character.glyph.glyphRect.x + endPadding + (float)m_cached_Underline_Character.glyph.glyphRect.width / 2) / m_fontAsset.atlasWidth, uv1.y); // Mid Top Right
            Vector2 uv5 = new Vector2(uv4.x, uv0.y); // Mid Bottom right
            Vector2 uv6 = new Vector2((m_cached_Underline_Character.glyph.glyphRect.x + endPadding + m_cached_Underline_Character.glyph.glyphRect.width) / m_fontAsset.atlasWidth, uv1.y); // End Part - Bottom Right
            Vector2 uv7 = new Vector2(uv6.x, uv0.y); // End Part - Top Right

            // Left Part of the Underline
            uvs0[0 + index] = uv0; // BL
            uvs0[1 + index] = uv1; // TL
            uvs0[2 + index] = uv2; // TR
            uvs0[3 + index] = uv3; // BR

            // Middle Part of the Underline
            uvs0[4 + index] = new Vector2(uv2.x - uv2.x * 0.001f, uv0.y);
            uvs0[5 + index] = new Vector2(uv2.x - uv2.x * 0.001f, uv1.y);
            uvs0[6 + index] = new Vector2(uv2.x + uv2.x * 0.001f, uv1.y);
            uvs0[7 + index] = new Vector2(uv2.x + uv2.x * 0.001f, uv0.y);

            // Right Part of the Underline
            uvs0[8 + index] = uv5;
            uvs0[9 + index] = uv4;
            uvs0[10 + index] = uv6;
            uvs0[11 + index] = uv7;
            #endregion

            // UNDERLINE UV2
            #region HANDLE UV2 - SDF SCALE
            // UV1 contains Face / Border UV layout.
            float min_UvX = 0;
            float max_UvX = (vertices[index + 2].x - start.x) / (end.x - start.x);

            //Calculate the xScale or how much the UV's are getting stretched on the X axis for the middle section of the underline.
            float xScale = Mathf.Abs(sdfScale);

            Vector2[] uvs2 = m_textInfo.meshInfo[0].uvs2;

            uvs2[0 + index] = PackUV(0, 0, xScale);
            uvs2[1 + index] = PackUV(0, 1, xScale);
            uvs2[2 + index] = PackUV(max_UvX, 1, xScale);
            uvs2[3 + index] = PackUV(max_UvX, 0, xScale);

            min_UvX = (vertices[index + 4].x - start.x) / (end.x - start.x);
            max_UvX = (vertices[index + 6].x - start.x) / (end.x - start.x);

            uvs2[4 + index] = PackUV(min_UvX, 0, xScale);
            uvs2[5 + index] = PackUV(min_UvX, 1, xScale);
            uvs2[6 + index] = PackUV(max_UvX, 1, xScale);
            uvs2[7 + index] = PackUV(max_UvX, 0, xScale);

            min_UvX = (vertices[index + 8].x - start.x) / (end.x - start.x);
            max_UvX = (vertices[index + 6].x - start.x) / (end.x - start.x);

            uvs2[8 + index] = PackUV(min_UvX, 0, xScale);
            uvs2[9 + index] = PackUV(min_UvX, 1, xScale);
            uvs2[10 + index] = PackUV(1, 1, xScale);
            uvs2[11 + index] = PackUV(1, 0, xScale);
            #endregion

            // UNDERLINE VERTEX COLORS
            #region
            // Alpha is the lower of the vertex color or tag color alpha used.
            underlineColor.a = m_fontColor32.a < underlineColor.a ? (byte)(m_fontColor32.a) : (byte)(underlineColor.a);

            Color32[] colors32 = m_textInfo.meshInfo[0].colors32;
            colors32[0 + index] = underlineColor;
            colors32[1 + index] = underlineColor;
            colors32[2 + index] = underlineColor;
            colors32[3 + index] = underlineColor;

            colors32[4 + index] = underlineColor;
            colors32[5 + index] = underlineColor;
            colors32[6 + index] = underlineColor;
            colors32[7 + index] = underlineColor;

            colors32[8 + index] = underlineColor;
            colors32[9 + index] = underlineColor;
            colors32[10 + index] = underlineColor;
            colors32[11 + index] = underlineColor;
            #endregion

            index += 12;
        }


        protected virtual void DrawTextHighlight(Vector3 start, Vector3 end, ref int index, Color32 highlightColor)
        {
            if (m_cached_Underline_Character == null)
            {
                if (!TMP_Settings.warningsDisabled) Debug.LogWarning("Unable to add underline since the Font Asset doesn't contain the underline character.", this);
                return;
            }

            int verticesCount = index + 4;
            // Check to make sure our current mesh buffer allocations can hold these new Quads.
            if (verticesCount > m_textInfo.meshInfo[0].vertices.Length)
            {
                // Resize Mesh Buffers
                m_textInfo.meshInfo[0].ResizeMeshInfo(verticesCount / 4);
            }

            // UNDERLINE VERTICES FOR (3) LINE SEGMENTS
            #region HIGHLIGHT VERTICES
            Vector3[] vertices = m_textInfo.meshInfo[0].vertices;

            // Front Part of the Underline
            vertices[index + 0] = start; // BL
            vertices[index + 1] = new Vector3(start.x, end.y, 0); // TL
            vertices[index + 2] = end; // TR
            vertices[index + 3] = new Vector3(end.x, start.y, 0); // BR
            #endregion

            // UNDERLINE UV0
            #region HANDLE UV0
            Vector2[] uvs0 = m_textInfo.meshInfo[0].uvs0;

            // Calculate UV required to setup the 3 Quads for the Underline.
            Vector2 uv0 = new Vector2(((float)m_cached_Underline_Character.glyph.glyphRect.x + m_cached_Underline_Character.glyph.glyphRect.width / 2) / m_fontAsset.atlasWidth, (m_cached_Underline_Character.glyph.glyphRect.y + (float)m_cached_Underline_Character.glyph.glyphRect.height / 2) / m_fontAsset.atlasHeight);  // bottom left
            //Vector2 uv1 = new Vector2(uv0.x, uv0.y);  // top left
            //Vector2 uv2 = new Vector2(uv0.x, uv0.y); // Top Right
            //Vector2 uv3 = new Vector2(uv2.x, uv0.y); // Bottom Right

            // Left Part of the Underline
            uvs0[0 + index] = uv0; // BL
            uvs0[1 + index] = uv0; // TL
            uvs0[2 + index] = uv0; // TR
            uvs0[3 + index] = uv0; // BR
            #endregion

            // UNDERLINE UV2
            #region HANDLE UV2 - SDF SCALE
            // UV1 contains Face / Border UV layout.
            //float min_UvX = 0;
            //float max_UvX = (vertices[index + 2].x - start.x) / (end.x - start.x);

            ////Calculate the xScale or how much the UV's are getting stretched on the X axis for the middle section of the underline.
            //float xScale = 0; // Mathf.Abs(sdfScale);

            Vector2[] uvs2 = m_textInfo.meshInfo[0].uvs2;
            Vector2 customUV = new Vector2(0, 1);
            uvs2[0 + index] = customUV; // PackUV(-0.2f, -0.2f, xScale);
            uvs2[1 + index] = customUV; // PackUV(-0.2f, -0.1f, xScale);
            uvs2[2 + index] = customUV; // PackUV(-0.1f, -0.1f, xScale);
            uvs2[3 + index] = customUV; // PackUV(-0.1f, -0.2f, xScale);
            #endregion

            // HIGHLIGHT VERTEX COLORS
            #region
            // Alpha is the lower of the vertex color or tag color alpha used.
            highlightColor.a = m_fontColor32.a < highlightColor.a ? m_fontColor32.a : highlightColor.a;

            Color32[] colors32 = m_textInfo.meshInfo[0].colors32;
            colors32[0 + index] = highlightColor;
            colors32[1 + index] = highlightColor;
            colors32[2 + index] = highlightColor;
            colors32[3 + index] = highlightColor;
            #endregion

            index += 4;
        }


        /// <summary>
        /// Internal function used to load the default settings of text objects.
        /// </summary>
        protected void LoadDefaultSettings()
        {
            if (m_text == null || m_isWaitingOnResourceLoad)
            {
                if (TMP_Settings.autoSizeTextContainer)
                    autoSizeTextContainer = true;
                else
                {
                    m_rectTransform = this.rectTransform;

                    if (GetType() == typeof(TextMeshPro))
                        m_rectTransform.sizeDelta = TMP_Settings.defaultTextMeshProTextContainerSize;
                    else
                        m_rectTransform.sizeDelta = TMP_Settings.defaultTextMeshProUITextContainerSize;
                }

                m_enableWordWrapping = TMP_Settings.enableWordWrapping;
                m_enableKerning = TMP_Settings.enableKerning;
                m_enableExtraPadding = TMP_Settings.enableExtraPadding;
                m_tintAllSprites = TMP_Settings.enableTintAllSprites;
                m_parseCtrlCharacters = TMP_Settings.enableParseEscapeCharacters;
                m_fontSize = m_fontSizeBase = TMP_Settings.defaultFontSize;
                m_fontSizeMin = m_fontSize * TMP_Settings.defaultTextAutoSizingMinRatio;
                m_fontSizeMax = m_fontSize * TMP_Settings.defaultTextAutoSizingMaxRatio;
                m_isWaitingOnResourceLoad = false;
                raycastTarget = TMP_Settings.enableRaycastTarget;
            }
        }


        /// <summary>
        /// Method used to find and cache references to the Underline and Ellipsis characters.
        /// </summary>
        /// <param name=""></param>
        protected void GetSpecialCharacters(TMP_FontAsset fontAsset)
        {
            bool isUsingAlternativeTypeface;
            TMP_FontAsset tempFontAsset;

            // Check & Assign Underline Character for use with the Underline tag.
            if (!fontAsset.characterLookupTable.TryGetValue(95, out m_cached_Underline_Character))
            {
                m_cached_Underline_Character = TMP_FontAssetUtilities.GetCharacterFromFontAsset(95,fontAsset, false, m_FontStyleInternal, (FontWeight)m_FontWeightInternal, out isUsingAlternativeTypeface, out tempFontAsset);

                if (m_cached_Underline_Character == null)
            {
                    if (!TMP_Settings.warningsDisabled)
                        Debug.LogWarning("The character used for Underline and Strikethrough is not available in font asset [" + fontAsset.name + "].", this);
                }
            }

            // Check & Assign Underline Character for use with the Underline tag.
            if (!fontAsset.characterLookupTable.TryGetValue(8230, out m_cached_Ellipsis_Character)) //95
            {
                m_cached_Ellipsis_Character = TMP_FontAssetUtilities.GetCharacterFromFontAsset(8230, fontAsset, false, m_FontStyleInternal, (FontWeight)m_FontWeightInternal, out isUsingAlternativeTypeface, out tempFontAsset);

                if (m_cached_Ellipsis_Character == null)
            {
                    if (!TMP_Settings.warningsDisabled)
                        Debug.LogWarning("The character used for Ellipsis is not available in font asset [" + fontAsset.name + "].", this);
                }
            }
        }


        /// <summary>
        /// Replace a given number of characters (tag) in the array with a new character and shift subsequent characters in the array.
        /// </summary>
        /// <param name="chars">Array which contains the text.</param>
        /// <param name="insertionIndex">The index of where the new character will be inserted</param>
        /// <param name="tagLength">Length of the tag being replaced.</param>
        /// <param name="c">The replacement character.</param>
        protected void ReplaceTagWithCharacter(int[] chars, int insertionIndex, int tagLength, char c)
        {
            chars[insertionIndex] = c;

            for (int i = insertionIndex + tagLength; i < chars.Length; i++)
            {
                chars[i - 3] = chars[i];
            }
        }


        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        //protected int GetMaterialReferenceForFontWeight()
        //{
        //    //bool isItalic = (m_style & FontStyles.Italic) == FontStyles.Italic || (m_fontStyle & FontStyles.Italic) == FontStyles.Italic;

        //    m_currentMaterialIndex = MaterialReference.AddMaterialReference(m_currentFontAsset.fontWeights[0].italicTypeface.material, m_currentFontAsset.fontWeights[0].italicTypeface, m_materialReferences, m_materialReferenceIndexLookup);

        //    return 0;
        //}


        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        protected TMP_FontAsset GetFontAssetForWeight(int fontWeight)
        {
            bool isItalic = (m_FontStyleInternal & FontStyles.Italic) == FontStyles.Italic || (m_fontStyle & FontStyles.Italic) == FontStyles.Italic;

            TMP_FontAsset fontAsset = null;

            int weightIndex = fontWeight / 100;

            if (isItalic)
                fontAsset = m_currentFontAsset.fontWeightTable[weightIndex].italicTypeface;
            else
                fontAsset = m_currentFontAsset.fontWeightTable[weightIndex].regularTypeface;

            return fontAsset;
        }


        /// <summary>
        /// Method to Enable or Disable child SubMesh objects.
        /// </summary>
        /// <param name="state"></param>
        protected virtual void SetActiveSubMeshes(bool state) { }


        /// <summary>
        /// Destroy Sub Mesh Objects.
        /// </summary>
        protected virtual void ClearSubMeshObjects() { }


        /// <summary>
        /// Function to clear the geometry of the Primary and Sub Text objects.
        /// </summary>
        public virtual void ClearMesh() { }


        /// <summary>
        /// Function to clear the geometry of the Primary and Sub Text objects.
        /// </summary>
        public virtual void ClearMesh(bool uploadGeometry) { }


        /// <summary>
        /// Function which returns the text after it has been parsed and rich text tags removed.
        /// </summary>
        /// <returns></returns>
        public virtual string GetParsedText()
        {
            if (m_textInfo == null)
                return string.Empty;

            int characterCount = m_textInfo.characterCount;

            // TODO - Could implement some static buffer pool shared by all instances of TMP objects.
            char[] buffer = new char[characterCount];

            for (int i = 0; i < characterCount && i < m_textInfo.characterInfo.Length; i++)
            {
                buffer[i] = m_textInfo.characterInfo[i].character;
            }

            return new string(buffer);
        }


        /// <summary>
        /// Function to pack scale information in the UV2 Channel.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="scale"></param>
        /// <returns></returns>
        //protected Vector2 PackUV(float x, float y, float scale)
        //{
        //    Vector2 output;

        //    output.x = Mathf.Floor(x * 4095);
        //    output.y = Mathf.Floor(y * 4095);

        //    output.x = (output.x * 4096) + output.y;
        //    output.y = scale;

        //    return output;
        //}

        /// <summary>
        /// Function to pack scale information in the UV2 Channel.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="scale"></param>
        /// <returns></returns>
        protected Vector2 PackUV(float x, float y, float scale)
        {
            Vector2 output;

            output.x = (int)(x * 511);
            output.y = (int)(y * 511);

            output.x = (output.x * 4096) + output.y;
            output.y = scale;

            return output;
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        protected float PackUV(float x, float y)
        {
            double x0 = (int)(x * 511);
            double y0 = (int)(y * 511);

            return (float)((x0 * 4096) + y0);
        }


        /// <summary>
        /// Function used as a replacement for LateUpdate()
        /// </summary>
        internal virtual void InternalUpdate() { }


        /// <summary>
        /// Function to pack scale information in the UV2 Channel.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="scale"></param>
        /// <returns></returns>
        //protected Vector2 PackUV(float x, float y, float scale)
        //{
        //    Vector2 output;

        //    output.x = Mathf.Floor(x * 4095);
        //    output.y = Mathf.Floor(y * 4095);

        //    return new Vector2((output.x * 4096) + output.y, scale);
        //}


        /// <summary>
        /// 
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        //protected float PackUV(float x, float y)
        //{
        //    x = (x % 5) / 5;
        //    y = (y % 5) / 5;

        //    return Mathf.Round(x * 4096) + y;
        //}


        /// <summary>
        /// Method to convert Hex to Int
        /// </summary>
        /// <param name="hex"></param>
        /// <returns></returns>
        protected int HexToInt(char hex)
        {
            switch (hex)
            {
                case '0': return 0;
                case '1': return 1;
                case '2': return 2;
                case '3': return 3;
                case '4': return 4;
                case '5': return 5;
                case '6': return 6;
                case '7': return 7;
                case '8': return 8;
                case '9': return 9;
                case 'A': return 10;
                case 'B': return 11;
                case 'C': return 12;
                case 'D': return 13;
                case 'E': return 14;
                case 'F': return 15;
                case 'a': return 10;
                case 'b': return 11;
                case 'c': return 12;
                case 'd': return 13;
                case 'e': return 14;
                case 'f': return 15;
            }
            return 15;
        }


        /// <summary>
        /// Convert UTF-16 Hex to Char
        /// </summary>
        /// <returns>The Unicode hex.</returns>
        /// <param name="i">The index.</param>
        protected int GetUTF16(string text, int i)
        {
            int unicode = 0;
            unicode += HexToInt(text[i]) << 12;
            unicode += HexToInt(text[i + 1]) << 8;
            unicode += HexToInt(text[i + 2]) << 4;
            unicode += HexToInt(text[i + 3]);
            return unicode;
        }

        /// <summary>
        /// Convert UTF-16 Hex to Char
        /// </summary>
        /// <returns>The Unicode hex.</returns>
        /// <param name="i">The index.</param>
        protected int GetUTF16(StringBuilder text, int i)
        {
            int unicode = 0;
            unicode += HexToInt(text[i]) << 12;
            unicode += HexToInt(text[i + 1]) << 8;
            unicode += HexToInt(text[i + 2]) << 4;
            unicode += HexToInt(text[i + 3]);
            return unicode;
        }


        /// <summary>
        /// Convert UTF-32 Hex to Char
        /// </summary>
        /// <returns>The Unicode hex.</returns>
        /// <param name="i">The index.</param>
        protected int GetUTF32(string text, int i)
        {
            int unicode = 0;
            unicode += HexToInt(text[i]) << 30;
            unicode += HexToInt(text[i + 1]) << 24;
            unicode += HexToInt(text[i + 2]) << 20;
            unicode += HexToInt(text[i + 3]) << 16;
            unicode += HexToInt(text[i + 4]) << 12;
            unicode += HexToInt(text[i + 5]) << 8;
            unicode += HexToInt(text[i + 6]) << 4;
            unicode += HexToInt(text[i + 7]);
            return unicode;
        }

        /// <summary>
        /// Convert UTF-32 Hex to Char
        /// </summary>
        /// <returns>The Unicode hex.</returns>
        /// <param name="i">The index.</param>
        protected int GetUTF32(StringBuilder text, int i)
        {
            int unicode = 0;
            unicode += HexToInt(text[i]) << 30;
            unicode += HexToInt(text[i + 1]) << 24;
            unicode += HexToInt(text[i + 2]) << 20;
            unicode += HexToInt(text[i + 3]) << 16;
            unicode += HexToInt(text[i + 4]) << 12;
            unicode += HexToInt(text[i + 5]) << 8;
            unicode += HexToInt(text[i + 6]) << 4;
            unicode += HexToInt(text[i + 7]);
            return unicode;
        }


        /// <summary>
        /// Method to convert Hex color values to Color32
        /// </summary>
        /// <param name="hexChars"></param>
        /// <param name="tagCount"></param>
        /// <returns></returns>
        protected Color32 HexCharsToColor(char[] hexChars, int tagCount)
        {
            if (tagCount == 4)
            {
                byte r = (byte)(HexToInt(hexChars[1]) * 16 + HexToInt(hexChars[1]));
                byte g = (byte)(HexToInt(hexChars[2]) * 16 + HexToInt(hexChars[2]));
                byte b = (byte)(HexToInt(hexChars[3]) * 16 + HexToInt(hexChars[3]));

                return new Color32(r, g, b, 255);
            }
            else if (tagCount == 5)
            {
                byte r = (byte)(HexToInt(hexChars[1]) * 16 + HexToInt(hexChars[1]));
                byte g = (byte)(HexToInt(hexChars[2]) * 16 + HexToInt(hexChars[2]));
                byte b = (byte)(HexToInt(hexChars[3]) * 16 + HexToInt(hexChars[3]));
                byte a = (byte)(HexToInt(hexChars[4]) * 16 + HexToInt(hexChars[4]));

                return new Color32(r, g, b, a);
            }
            else if (tagCount == 7)
            {
                byte r = (byte)(HexToInt(hexChars[1]) * 16 + HexToInt(hexChars[2]));
                byte g = (byte)(HexToInt(hexChars[3]) * 16 + HexToInt(hexChars[4]));
                byte b = (byte)(HexToInt(hexChars[5]) * 16 + HexToInt(hexChars[6]));

                return new Color32(r, g, b, 255);
            }
            else if (tagCount == 9)
            {
                byte r = (byte)(HexToInt(hexChars[1]) * 16 + HexToInt(hexChars[2]));
                byte g = (byte)(HexToInt(hexChars[3]) * 16 + HexToInt(hexChars[4]));
                byte b = (byte)(HexToInt(hexChars[5]) * 16 + HexToInt(hexChars[6]));
                byte a = (byte)(HexToInt(hexChars[7]) * 16 + HexToInt(hexChars[8]));

                return new Color32(r, g, b, a);
            }
            else if (tagCount == 10)
            {
                byte r = (byte)(HexToInt(hexChars[7]) * 16 + HexToInt(hexChars[7]));
                byte g = (byte)(HexToInt(hexChars[8]) * 16 + HexToInt(hexChars[8]));
                byte b = (byte)(HexToInt(hexChars[9]) * 16 + HexToInt(hexChars[9]));

                return new Color32(r, g, b, 255);
            }
            else if (tagCount == 11)
            {
                byte r = (byte)(HexToInt(hexChars[7]) * 16 + HexToInt(hexChars[7]));
                byte g = (byte)(HexToInt(hexChars[8]) * 16 + HexToInt(hexChars[8]));
                byte b = (byte)(HexToInt(hexChars[9]) * 16 + HexToInt(hexChars[9]));
                byte a = (byte)(HexToInt(hexChars[10]) * 16 + HexToInt(hexChars[10]));

                return new Color32(r, g, b, a);
            }
            else if (tagCount == 13)
            {
                byte r = (byte)(HexToInt(hexChars[7]) * 16 + HexToInt(hexChars[8]));
                byte g = (byte)(HexToInt(hexChars[9]) * 16 + HexToInt(hexChars[10]));
                byte b = (byte)(HexToInt(hexChars[11]) * 16 + HexToInt(hexChars[12]));

                return new Color32(r, g, b, 255);
            }
            else if (tagCount == 15)
            {
                byte r = (byte)(HexToInt(hexChars[7]) * 16 + HexToInt(hexChars[8]));
                byte g = (byte)(HexToInt(hexChars[9]) * 16 + HexToInt(hexChars[10]));
                byte b = (byte)(HexToInt(hexChars[11]) * 16 + HexToInt(hexChars[12]));
                byte a = (byte)(HexToInt(hexChars[13]) * 16 + HexToInt(hexChars[14]));

                return new Color32(r, g, b, a);
            }

            return new Color32(255, 255, 255, 255);
        }


        /// <summary>
        /// Method to convert Hex Color values to Color32
        /// </summary>
        /// <param name="hexChars"></param>
        /// <param name="startIndex"></param>
        /// <param name="length"></param>
        /// <returns></returns>
        protected Color32 HexCharsToColor(char[] hexChars, int startIndex, int length)
        {
            if (length == 7)
            {
                byte r = (byte)(HexToInt(hexChars[startIndex + 1]) * 16 + HexToInt(hexChars[startIndex + 2]));
                byte g = (byte)(HexToInt(hexChars[startIndex + 3]) * 16 + HexToInt(hexChars[startIndex + 4]));
                byte b = (byte)(HexToInt(hexChars[startIndex + 5]) * 16 + HexToInt(hexChars[startIndex + 6]));

                return new Color32(r, g, b, 255);
            }
            else if (length == 9)
            {
                byte r = (byte)(HexToInt(hexChars[startIndex + 1]) * 16 + HexToInt(hexChars[startIndex + 2]));
                byte g = (byte)(HexToInt(hexChars[startIndex + 3]) * 16 + HexToInt(hexChars[startIndex + 4]));
                byte b = (byte)(HexToInt(hexChars[startIndex + 5]) * 16 + HexToInt(hexChars[startIndex + 6]));
                byte a = (byte)(HexToInt(hexChars[startIndex + 7]) * 16 + HexToInt(hexChars[startIndex + 8]));

                return new Color32(r, g, b, a);
            }

            return s_colorWhite;
        }


        /// <summary>
        /// Method which returns the number of parameters used in a tag attribute and populates an array with such values.
        /// </summary>
        /// <param name="chars">Char[] containing the tag attribute and data</param>
        /// <param name="startIndex">The index of the first char of the data</param>
        /// <param name="length">The length of the data</param>
        /// <param name="parameters">The number of parameters contained in the Char[]</param>
        /// <returns></returns>
        int GetAttributeParameters(char[] chars, int startIndex, int length, ref float[] parameters)
        {
            int endIndex = startIndex;
            int attributeCount = 0;

            while (endIndex < startIndex + length)
            {
                parameters[attributeCount] = ConvertToFloat(chars, startIndex, length, out endIndex);

                length -= (endIndex - startIndex) + 1;
                startIndex = endIndex + 1;

                attributeCount += 1;
            }

            return attributeCount;
        }


        /// <summary>
        /// Extracts a float value from char[] assuming we know the position of the start, end and decimal point.
        /// </summary>
        /// <param name="chars"></param>
        /// <param name="startIndex"></param>
        /// <param name="length"></param>
        /// <returns></returns>
        protected float ConvertToFloat(char[] chars, int startIndex, int length)
        {
            int lastIndex;

            return ConvertToFloat(chars, startIndex, length, out lastIndex);
        }


        /// <summary>
        /// Extracts a float value from char[] given a start index and length. 
        /// </summary>
        /// <param name="chars"></param> The Char[] containing the numerical sequence.
        /// <param name="startIndex"></param> The index of the start of the numerical sequence.
        /// <param name="length"></param> The length of the numerical sequence.
        /// <param name="lastIndex"></param> Index of the last character in the validated sequence.
        /// <returns></returns>
        protected float ConvertToFloat(char[] chars, int startIndex, int length, out int lastIndex)
        {
            if (startIndex == 0) { lastIndex = 0; return -9999; }
            int endIndex = startIndex + length;

            bool isIntegerValue = true;
            float decimalPointMultiplier = 0;

            // Set value multiplier checking the first character to determine if we are using '+' or '-'
            int valueSignMultiplier = 1;
            if (chars[startIndex] == '+')
            {
                valueSignMultiplier = 1;
                startIndex += 1;
            }
            else if (chars[startIndex] == '-')
            {
                valueSignMultiplier = -1;
                startIndex += 1;
            }

            float value = 0;

            for (int i = startIndex; i < endIndex; i++)
            {
                uint c = chars[i];

                if (c >= '0' && c <= '9' || c == '.')
                {
                    if (c == '.')
                    {
                        isIntegerValue = false;
                        decimalPointMultiplier = 0.1f;
                        continue;
                    }

                    //Calculate integer and floating point value
                    if (isIntegerValue)
                        value = value * 10 + (c - 48) * valueSignMultiplier;
                    else
                {
                        value = value + (c - 48) * decimalPointMultiplier * valueSignMultiplier;
                        decimalPointMultiplier *= 0.1f;
                }

                    continue;
                }
                else if (c == ',')
                {
                    if (i + 1 < endIndex && chars[i + 1] == ' ')
                        lastIndex = i + 1;
                else
                        lastIndex = i;

                    return value;
                }
            }

            lastIndex = endIndex;
            return value;
        }


        /// <summary>
        /// Function to identify and validate the rich tag. Returns the position of the > if the tag was valid.
        /// </summary>
        /// <param name="chars"></param>
        /// <param name="startIndex"></param>
        /// <param name="endIndex"></param>
        /// <returns></returns>
        protected bool ValidateHtmlTag(UnicodeChar[] chars, int startIndex, out int endIndex)
        {
            int tagCharCount = 0;
            byte attributeFlag = 0;

            int attributeIndex = 0;
            m_xmlAttribute[attributeIndex].nameHashCode = 0;
            m_xmlAttribute[attributeIndex].valueHashCode = 0;
            m_xmlAttribute[attributeIndex].valueStartIndex = 0;
            m_xmlAttribute[attributeIndex].valueLength = 0;
            TagValueType tagValueType = m_xmlAttribute[attributeIndex].valueType = TagValueType.None;
            TagUnitType tagUnitType = m_xmlAttribute[attributeIndex].unitType = TagUnitType.Pixels;

            // Clear attribute name hash codes
            m_xmlAttribute[1].nameHashCode = 0;
            m_xmlAttribute[2].nameHashCode = 0;
            m_xmlAttribute[3].nameHashCode = 0;
            m_xmlAttribute[4].nameHashCode = 0;

            endIndex = startIndex;
            bool isTagSet = false;
            bool isValidHtmlTag = false;

            for (int i = startIndex; i < chars.Length && chars[i].unicode != 0 && tagCharCount < m_htmlTag.Length && chars[i].unicode != '<'; i++)
            {
                int unicode = chars[i].unicode;

                if (unicode == '>') // ASCII Code of End HTML tag '>'
                {
                    isValidHtmlTag = true;
                    endIndex = i;
                    m_htmlTag[tagCharCount] = (char)0;
                    break;
                }

                m_htmlTag[tagCharCount] = (char)unicode;
                tagCharCount += 1;

                if (attributeFlag == 1)
                {
                    if (tagValueType == TagValueType.None)
                    {
                        // Check for attribute type
                        if (unicode == '+' || unicode == '-' || unicode == '.' || (unicode >= '0' && unicode <= '9'))
                        {
                            tagUnitType = TagUnitType.Pixels;
                            tagValueType = m_xmlAttribute[attributeIndex].valueType = TagValueType.NumericalValue;
                            m_xmlAttribute[attributeIndex].valueStartIndex = tagCharCount - 1;
                            m_xmlAttribute[attributeIndex].valueLength += 1;
                        }
                        else if (unicode == '#')
                        {
                            tagUnitType = TagUnitType.Pixels;
                            tagValueType = m_xmlAttribute[attributeIndex].valueType = TagValueType.ColorValue;
                            m_xmlAttribute[attributeIndex].valueStartIndex = tagCharCount - 1;
                            m_xmlAttribute[attributeIndex].valueLength += 1;
                        }
                        else if (unicode == '"')
                        {
                            tagUnitType = TagUnitType.Pixels;
                            tagValueType = m_xmlAttribute[attributeIndex].valueType = TagValueType.StringValue;
                            m_xmlAttribute[attributeIndex].valueStartIndex = tagCharCount;
                        }
                        else
                        {
                            tagUnitType = TagUnitType.Pixels;
                            tagValueType = m_xmlAttribute[attributeIndex].valueType = TagValueType.StringValue;
                            m_xmlAttribute[attributeIndex].valueStartIndex = tagCharCount - 1;
                            m_xmlAttribute[attributeIndex].valueHashCode = (m_xmlAttribute[attributeIndex].valueHashCode << 5) + m_xmlAttribute[attributeIndex].valueHashCode ^ unicode;
                            m_xmlAttribute[attributeIndex].valueLength += 1;
                        }
                    }
                    else
                    {
                        if (tagValueType == TagValueType.NumericalValue)
                        {
                            // Check for termination of numerical value.
                            if (unicode == 'p' || unicode == 'e' || unicode == '%' || unicode == ' ')
                            {
                                attributeFlag = 2;
                                tagValueType = TagValueType.None;

                                switch (unicode)
                                {
                                    case 'e':
                                        m_xmlAttribute[attributeIndex].unitType = tagUnitType = TagUnitType.FontUnits;
                                        break;
                                    case '%':
                                        m_xmlAttribute[attributeIndex].unitType = tagUnitType = TagUnitType.Percentage;
                                        break;
                                    default:
                                        m_xmlAttribute[attributeIndex].unitType = tagUnitType = TagUnitType.Pixels;
                                        break;
                                }

                                attributeIndex += 1;
                                m_xmlAttribute[attributeIndex].nameHashCode = 0;
                                m_xmlAttribute[attributeIndex].valueHashCode = 0;
                                m_xmlAttribute[attributeIndex].valueType = TagValueType.None;
                                m_xmlAttribute[attributeIndex].unitType = TagUnitType.Pixels;
                                m_xmlAttribute[attributeIndex].valueStartIndex = 0;
                                m_xmlAttribute[attributeIndex].valueLength = 0;

                            }
                            else if (attributeFlag != 2)
                            {
                                m_xmlAttribute[attributeIndex].valueLength += 1;
                            }
                        }
                        else if (tagValueType == TagValueType.ColorValue)
                        {
                            if (unicode != ' ')
                            {
                                m_xmlAttribute[attributeIndex].valueLength += 1;
                            }
                            else
                            {
                                attributeFlag = 2;
                                tagValueType = TagValueType.None;
                                tagUnitType = TagUnitType.Pixels;
                                attributeIndex += 1;
                                m_xmlAttribute[attributeIndex].nameHashCode = 0;
                                m_xmlAttribute[attributeIndex].valueType = TagValueType.None;
                                m_xmlAttribute[attributeIndex].unitType = TagUnitType.Pixels;
                                m_xmlAttribute[attributeIndex].valueHashCode = 0;
                                m_xmlAttribute[attributeIndex].valueStartIndex = 0;
                                m_xmlAttribute[attributeIndex].valueLength = 0;
                            }
                        }
                        else if (tagValueType == TagValueType.StringValue)
                        {
                            // Compute HashCode value for the named tag.
                            if (unicode != '"')
                            {
                                m_xmlAttribute[attributeIndex].valueHashCode = (m_xmlAttribute[attributeIndex].valueHashCode << 5) + m_xmlAttribute[attributeIndex].valueHashCode ^ unicode;
                                m_xmlAttribute[attributeIndex].valueLength += 1;
                            }
                            else
                            {
                                attributeFlag = 2;
                                tagValueType = TagValueType.None;
                                tagUnitType = TagUnitType.Pixels;
                                attributeIndex += 1;
                                m_xmlAttribute[attributeIndex].nameHashCode = 0;
                                m_xmlAttribute[attributeIndex].valueType = TagValueType.None;
                                m_xmlAttribute[attributeIndex].unitType = TagUnitType.Pixels;
                                m_xmlAttribute[attributeIndex].valueHashCode = 0;
                                m_xmlAttribute[attributeIndex].valueStartIndex = 0;
                                m_xmlAttribute[attributeIndex].valueLength = 0;
                            }
                        }
                    }
                }


                if (unicode == '=') // '=' 
                    attributeFlag = 1;

                // Compute HashCode for the name of the attribute
                if (attributeFlag == 0 && unicode == ' ')
                {
                    if (isTagSet) return false;

                    isTagSet = true;
                    attributeFlag = 2;

                    tagValueType = TagValueType.None;
                    tagUnitType = TagUnitType.Pixels;
                    attributeIndex += 1;
                    m_xmlAttribute[attributeIndex].nameHashCode = 0;
                    m_xmlAttribute[attributeIndex].valueType = TagValueType.None;
                    m_xmlAttribute[attributeIndex].unitType = TagUnitType.Pixels;
                    m_xmlAttribute[attributeIndex].valueHashCode = 0;
                    m_xmlAttribute[attributeIndex].valueStartIndex = 0;
                    m_xmlAttribute[attributeIndex].valueLength = 0;
                }

                if (attributeFlag == 0)
                    m_xmlAttribute[attributeIndex].nameHashCode = (m_xmlAttribute[attributeIndex].nameHashCode << 3) - m_xmlAttribute[attributeIndex].nameHashCode + unicode;

                if (attributeFlag == 2 && unicode == ' ')
                    attributeFlag = 0;

            }

            if (!isValidHtmlTag)
            {
                return false;
            }

            //Debug.Log("Tag is [" + m_htmlTag.ArrayToString() + "].  Tag HashCode: " + m_xmlAttribute[0].nameHashCode + "  Tag Value HashCode: " + m_xmlAttribute[0].valueHashCode + "  Attribute 1 HashCode: " + m_xmlAttribute[1].nameHashCode + " Value HashCode: " + m_xmlAttribute[1].valueHashCode);
            //for (int i = 0; i < attributeIndex; i++)
            //    Debug.Log("Tag [" + i + "] with HashCode: " + m_xmlAttribute[i].nameHashCode + " has value of [" + new string(m_htmlTag, m_xmlAttribute[i].valueStartIndex, m_xmlAttribute[i].valueLength) + "] Numerical Value: " + ConvertToFloat(m_htmlTag, m_xmlAttribute[i].valueStartIndex, m_xmlAttribute[i].valueLength));

            #region Rich Text Tag Processing
            #if !RICH_TEXT_ENABLED
            // Special handling of the no parsing tag </noparse> </NOPARSE> tag
            if (tag_NoParsing && (m_xmlAttribute[0].nameHashCode != 53822163 && m_xmlAttribute[0].nameHashCode != 49429939))
                return false;
            else if (m_xmlAttribute[0].nameHashCode == 53822163 || m_xmlAttribute[0].nameHashCode == 49429939)
            {
                tag_NoParsing = false;
                return true;
            }

            // Color <#FFF> 3 Hex values (short form)
            if (m_htmlTag[0] == 35 && tagCharCount == 4)
            {
                m_htmlColor = HexCharsToColor(m_htmlTag, tagCharCount);
                m_colorStack.Add(m_htmlColor);
                return true;
            }
            // Color <#FFF7> 4 Hex values with alpha (short form)
            else if (m_htmlTag[0] == 35 && tagCharCount == 5)
            {
                m_htmlColor = HexCharsToColor(m_htmlTag, tagCharCount);
                m_colorStack.Add(m_htmlColor);
                return true;
            }
            // Color <#FF00FF>
            else if (m_htmlTag[0] == 35 && tagCharCount == 7) // if Tag begins with # and contains 7 characters. 
            {
                m_htmlColor = HexCharsToColor(m_htmlTag, tagCharCount);
                m_colorStack.Add(m_htmlColor);
                return true;
            }
            // Color <#FF00FF00> with alpha
            else if (m_htmlTag[0] == 35 && tagCharCount == 9) // if Tag begins with # and contains 9 characters. 
            {
                m_htmlColor = HexCharsToColor(m_htmlTag, tagCharCount);
                m_colorStack.Add(m_htmlColor);
                return true;
            }
            else
            {
                float value = 0;

                switch (m_xmlAttribute[0].nameHashCode)
                {
                    case 98: // <b>
                    case 66: // <B>
                        m_FontStyleInternal |= FontStyles.Bold;
                        m_fontStyleStack.Add(FontStyles.Bold);

                        m_FontWeightInternal = FontWeight.Bold;
                        return true;
                    case 427: // </b>
                    case 395: // </B>
                        if ((m_fontStyle & FontStyles.Bold) != FontStyles.Bold)
                        {
                            if (m_fontStyleStack.Remove(FontStyles.Bold) == 0)
                            {
                                m_FontStyleInternal &= ~FontStyles.Bold;
                                m_FontWeightInternal = m_FontWeightStack.Peek();
                            }
                        }
                        return true;
                    case 105: // <i>
                    case 73: // <I>
                        m_FontStyleInternal |= FontStyles.Italic;
                        m_fontStyleStack.Add(FontStyles.Italic);
                        return true;
                    case 434: // </i>
                    case 402: // </I>
                        if ((m_fontStyle & FontStyles.Italic) != FontStyles.Italic)
                        {
                            if (m_fontStyleStack.Remove(FontStyles.Italic) == 0)
                                m_FontStyleInternal &= ~FontStyles.Italic;
                        }
                        return true;
                    case 115: // <s>
                    case 83: // <S>
                        m_FontStyleInternal |= FontStyles.Strikethrough;
                        m_fontStyleStack.Add(FontStyles.Strikethrough);

                        if (m_xmlAttribute[1].nameHashCode == 281955 || m_xmlAttribute[1].nameHashCode == 192323)
                        {
                            m_strikethroughColor = HexCharsToColor(m_htmlTag, m_xmlAttribute[1].valueStartIndex, m_xmlAttribute[1].valueLength);
                            m_strikethroughColor.a = m_htmlColor.a < m_strikethroughColor.a ? (byte)(m_htmlColor.a) : (byte)(m_strikethroughColor .a);
                        }
                        else
                            m_strikethroughColor = m_htmlColor;

                        m_strikethroughColorStack.Add(m_strikethroughColor);

                        return true;
                    case 444: // </s>
                    case 412: // </S>
                        if ((m_fontStyle & FontStyles.Strikethrough) != FontStyles.Strikethrough)
                        {
                            if (m_fontStyleStack.Remove(FontStyles.Strikethrough) == 0)
                                m_FontStyleInternal &= ~FontStyles.Strikethrough;
                        }
                        return true;
                    case 117: // <u>
                    case 85: // <U>
                        m_FontStyleInternal |= FontStyles.Underline;
                        m_fontStyleStack.Add(FontStyles.Underline);

                        if (m_xmlAttribute[1].nameHashCode == 281955 || m_xmlAttribute[1].nameHashCode == 192323)
                        {
                            m_underlineColor = HexCharsToColor(m_htmlTag, m_xmlAttribute[1].valueStartIndex, m_xmlAttribute[1].valueLength);
                            m_underlineColor.a = m_htmlColor.a < m_underlineColor.a ? (byte)(m_htmlColor.a) : (byte)(m_underlineColor.a);
                        }
                        else
                            m_underlineColor = m_htmlColor;

                        m_underlineColorStack.Add(m_underlineColor);

                        return true;
                    case 446: // </u>
                    case 414: // </U>
                        if ((m_fontStyle & FontStyles.Underline) != FontStyles.Underline)
                        {
                            m_underlineColor = m_underlineColorStack.Remove();

                            if (m_fontStyleStack.Remove(FontStyles.Underline) == 0)
                                m_FontStyleInternal &= ~FontStyles.Underline;
                        }
                        return true;
                    case 43045: // <mark=#FF00FF80>
                    case 30245: // <MARK>
                        m_FontStyleInternal |= FontStyles.Highlight;
                        m_fontStyleStack.Add(FontStyles.Highlight);

                        m_highlightColor = HexCharsToColor(m_htmlTag, m_xmlAttribute[0].valueStartIndex, m_xmlAttribute[0].valueLength);
                        m_highlightColor.a = m_htmlColor.a < m_highlightColor.a ? (byte)(m_htmlColor.a) : (byte)(m_highlightColor.a);
                        m_highlightColorStack.Add(m_highlightColor);

                        // Handle Mark Tag Attributes
                        for (int i = 0; i < m_xmlAttribute.Length && m_xmlAttribute[i].nameHashCode != 0; i++)
                        {
                            int nameHashCode = m_xmlAttribute[i].nameHashCode;

                            switch (nameHashCode)
                            {
                                case 281955: // color
                                    break;

                                case 15087385: // padding
                                    int paramCount = GetAttributeParameters(m_htmlTag, m_xmlAttribute[i].valueStartIndex, m_xmlAttribute[i].valueLength, ref m_attributeParameterValues);
                                    if (paramCount != 4) return false;

                                    m_highlightPadding = new Vector4(m_attributeParameterValues[0], m_attributeParameterValues[1], m_attributeParameterValues[2], m_attributeParameterValues[3]); 
                                    break;
                            }
                        }

                        return true;
                    case 155892: // </mark>
                    case 143092: // </MARK>
                        if ((m_fontStyle & FontStyles.Highlight) != FontStyles.Highlight)
                        {
                            m_highlightColor = m_highlightColorStack.Remove();

                            if (m_fontStyleStack.Remove(FontStyles.Highlight) == 0)
                                m_FontStyleInternal &= ~FontStyles.Highlight;
                        }
                        return true;
                    case 6552: // <sub>
                    case 4728: // <SUB>
                        m_fontScaleMultiplier *= m_currentFontAsset.faceInfo.subscriptSize > 0 ? m_currentFontAsset.faceInfo.subscriptSize : 1;
                        m_baselineOffsetStack.Push(m_baselineOffset);
                        m_baselineOffset += m_currentFontAsset.faceInfo.subscriptOffset * m_fontScale * m_fontScaleMultiplier;

                        m_fontStyleStack.Add(FontStyles.Subscript);
                        m_FontStyleInternal |= FontStyles.Subscript;
                        return true;
                    case 22673: // </sub>
                    case 20849: // </SUB>
                        if ((m_FontStyleInternal & FontStyles.Subscript) == FontStyles.Subscript)
                        {
                            if (m_fontScaleMultiplier < 1)
                            {
                                //m_baselineOffset -= m_currentFontAsset.fontInfo.SubscriptOffset * m_fontScale * m_fontScaleMultiplier;
                                m_baselineOffset = m_baselineOffsetStack.Pop();
                                m_fontScaleMultiplier /= m_currentFontAsset.faceInfo.subscriptSize > 0 ? m_currentFontAsset.faceInfo.subscriptSize : 1;
                            }

                            if (m_fontStyleStack.Remove(FontStyles.Subscript) == 0)
                                m_FontStyleInternal &= ~FontStyles.Subscript;
                        }
                        return true;
                    case 6566: // <sup>
                    case 4742: // <SUP>
                        m_fontScaleMultiplier *= m_currentFontAsset.faceInfo.superscriptSize > 0 ? m_currentFontAsset.faceInfo.superscriptSize : 1;
                        m_baselineOffsetStack.Push(m_baselineOffset);
                        m_baselineOffset += m_currentFontAsset.faceInfo.superscriptOffset * m_fontScale * m_fontScaleMultiplier;

                        m_fontStyleStack.Add(FontStyles.Superscript);
                        m_FontStyleInternal |= FontStyles.Superscript;
                        return true;
                    case 22687: // </sup>
                    case 20863: // </SUP>
                        if ((m_FontStyleInternal & FontStyles.Superscript) == FontStyles.Superscript)
                        {
                            if (m_fontScaleMultiplier < 1)
                            {
                                //m_baselineOffset -= m_currentFontAsset.fontInfo.SuperscriptOffset * m_fontScale * m_fontScaleMultiplier;
                                m_baselineOffset = m_baselineOffsetStack.Pop();
                                m_fontScaleMultiplier /= m_currentFontAsset.faceInfo.superscriptSize > 0 ? m_currentFontAsset.faceInfo.superscriptSize : 1;
                            }

                            if (m_fontStyleStack.Remove(FontStyles.Superscript) == 0)
                                m_FontStyleInternal &= ~FontStyles.Superscript;
                        }
                        return true;
                    case -330774850: // <font-weight>
                    case 2012149182: // <FONT-WEIGHT>
                        value = ConvertToFloat(m_htmlTag, m_xmlAttribute[0].valueStartIndex, m_xmlAttribute[0].valueLength);

                        //if (value == -9999) return false;

                        //if ((m_fontStyle & FontStyles.Bold) == FontStyles.Bold)
                        //{
                        //    // Nothing happens since Bold is forced on the text.
                        //    //m_fontWeight = 700;
                        //    return true;
                        //}


                        //// Remove bold style
                        //m_style &= ~FontStyles.Bold;

                        switch ((int)value)
                        {
                            case 100:
                                m_FontWeightInternal = FontWeight.Thin;
                                break;
                            case 200:
                                m_FontWeightInternal = FontWeight.ExtraLight;
                                break;
                            case 300:
                                m_FontWeightInternal = FontWeight.Light;
                                break;
                            case 400:
                                m_FontWeightInternal = FontWeight.Regular;
                                break;
                            case 500:
                                m_FontWeightInternal = FontWeight.Medium;
                                break;
                            case 600:
                                m_FontWeightInternal = FontWeight.SemiBold;
                                break;
                            case 700:
                                m_FontWeightInternal = FontWeight.Bold;
                                break;
                            case 800:
                                m_FontWeightInternal = FontWeight.Heavy;
                                break;
                            case 900:
                                m_FontWeightInternal = FontWeight.Black;
                                break;
                        }

                        m_FontWeightStack.Add(m_FontWeightInternal);

                        return true;
                    case -1885698441: // </font-weight>
                    case 457225591: // </FONT-WEIGHT>
                        m_FontWeightStack.Remove();

                        if (m_FontStyleInternal == FontStyles.Bold)
                            m_FontWeightInternal = FontWeight.Bold;
                        else
                            m_FontWeightInternal = m_FontWeightStack.Peek();

                        return true;
                    case 6380: // <pos=000.00px> <pos=0em> <pos=50%>
                    case 4556: // <POS>
                        value = ConvertToFloat(m_htmlTag, m_xmlAttribute[0].valueStartIndex, m_xmlAttribute[0].valueLength);
                        if (value == -9999) return false;

                        switch (tagUnitType)
                        {
                            case TagUnitType.Pixels:
                                m_xAdvance = value * (m_isOrthographic ? 1.0f : 0.1f);
                                //m_isIgnoringAlignment = true;
                                return true;
                            case TagUnitType.FontUnits:
                                m_xAdvance = value * m_currentFontSize * (m_isOrthographic ? 1.0f : 0.1f);
                                //m_isIgnoringAlignment = true;
                                return true;
                            case TagUnitType.Percentage:
                                m_xAdvance = m_marginWidth * value / 100;
                                //m_isIgnoringAlignment = true;
                                return true;
                        }
                        return false;
                    case 22501: // </pos>
                    case 20677: // </POS>
                        m_isIgnoringAlignment = false;
                        return true;
                    case 16034505: // <voffset>
                    case 11642281: // <VOFFSET>
                        value = ConvertToFloat(m_htmlTag, m_xmlAttribute[0].valueStartIndex, m_xmlAttribute[0].valueLength);
                        if (value == -9999) return false;

                        switch (tagUnitType)
                        {
                            case TagUnitType.Pixels:
                                m_baselineOffset = value * (m_isOrthographic ? 1 : 0.1f);
                                return true;
                            case TagUnitType.FontUnits:
                                m_baselineOffset = value * (m_isOrthographic ? 1 : 0.1f) * m_currentFontSize;
                                return true;
                            case TagUnitType.Percentage:
                                //m_baselineOffset = m_marginHeight * val / 100;
                                return false;
                        }
                        return false;
                    case 54741026: // </voffset>
                    case 50348802: // </VOFFSET>
                        m_baselineOffset = 0;
                        return true;
                    case 43991: // <page>
                    case 31191: // <PAGE>
                        // This tag only works when Overflow - Page mode is used.
                        if (m_overflowMode == TextOverflowModes.Page)
                        {
                            m_xAdvance = 0 + tag_LineIndent + tag_Indent;
                            m_lineOffset = 0;
                            m_pageNumber += 1;
                            m_isNewPage = true;
                        }
                        return true;
                    // <BR> tag is now handled inline where it is replaced by a linefeed or \n.
                    //case 544: // <BR>
                    //case 800: // <br>
                    //    m_forceLineBreak = true;
                    //    return true;
                    case 43969: // <nobr>
                    case 31169: // <NOBR>
                        m_isNonBreakingSpace = true;
                        return true;
                    case 156816: // </nobr>
                    case 144016: // </NOBR>
                        m_isNonBreakingSpace = false;
                        return true;
                    case 45545: // <size=>
                    case 32745: // <SIZE>
                        value = ConvertToFloat(m_htmlTag, m_xmlAttribute[0].valueStartIndex, m_xmlAttribute[0].valueLength);
                        if (value == -9999) return false;

                        switch (tagUnitType)
                        {
                            case TagUnitType.Pixels:
                                if (m_htmlTag[5] == 43) // <size=+00>
                                {
                                    m_currentFontSize = m_fontSize + value;
                                    m_sizeStack.Add(m_currentFontSize);
                                    m_fontScale = (m_currentFontSize / m_currentFontAsset.faceInfo.pointSize * m_currentFontAsset.faceInfo.scale * (m_isOrthographic ? 1 : 0.1f));
                                    return true;
                                }
                                else if (m_htmlTag[5] == 45) // <size=-00>
                                {
                                    m_currentFontSize = m_fontSize + value;
                                    m_sizeStack.Add(m_currentFontSize);
                                    m_fontScale = (m_currentFontSize / m_currentFontAsset.faceInfo.pointSize * m_currentFontAsset.faceInfo.scale * (m_isOrthographic ? 1 : 0.1f));
                                    return true;
                                }
                                else // <size=00.0>
                                {
                                    m_currentFontSize = value;
                                    m_sizeStack.Add(m_currentFontSize);
                                    m_fontScale = (m_currentFontSize / m_currentFontAsset.faceInfo.pointSize * m_currentFontAsset.faceInfo.scale * (m_isOrthographic ? 1 : 0.1f));
                                    return true;
                                }
                            case TagUnitType.FontUnits:
                                m_currentFontSize = m_fontSize * value;
                                m_sizeStack.Add(m_currentFontSize);
                                m_fontScale = (m_currentFontSize / m_currentFontAsset.faceInfo.pointSize * m_currentFontAsset.faceInfo.scale * (m_isOrthographic ? 1 : 0.1f));
                                return true;
                            case TagUnitType.Percentage:
                                m_currentFontSize = m_fontSize * value / 100;
                                m_sizeStack.Add(m_currentFontSize);
                                m_fontScale = (m_currentFontSize / m_currentFontAsset.faceInfo.pointSize * m_currentFontAsset.faceInfo.scale * (m_isOrthographic ? 1 : 0.1f));
                                return true;
                        }
                        return false;
                    case 158392: // </size>
                    case 145592: // </SIZE>
                        m_currentFontSize = m_sizeStack.Remove();
                        m_fontScale = (m_currentFontSize / m_currentFontAsset.faceInfo.pointSize * m_currentFontAsset.faceInfo.scale * (m_isOrthographic ? 1 : 0.1f));
                        return true;
                    case 41311: // <font=xx>
                    case 28511: // <FONT>
                        int fontHashCode = m_xmlAttribute[0].valueHashCode;
                        int materialAttributeHashCode = m_xmlAttribute[1].nameHashCode;
                        int materialHashCode = m_xmlAttribute[1].valueHashCode;

                        // Special handling for <font=default> or <font=Default>
                        if (fontHashCode == 764638571 || fontHashCode == 523367755)
                        {
                            m_currentFontAsset = m_materialReferences[0].fontAsset;
                            m_currentMaterial = m_materialReferences[0].material;
                            m_currentMaterialIndex = 0;
                            //Debug.Log("<font=Default> assigning Font Asset [" + m_currentFontAsset.name + "] with Material [" + m_currentMaterial.name + "].");

                            m_fontScale = (m_currentFontSize / m_currentFontAsset.faceInfo.pointSize * m_currentFontAsset.faceInfo.scale * (m_isOrthographic ? 1 : 0.1f));

                            m_materialReferenceStack.Add(m_materialReferences[0]);

                            return true;
                        }

                        TMP_FontAsset tempFont;
                        Material tempMaterial;

                        // HANDLE NEW FONT ASSET
                        if (MaterialReferenceManager.TryGetFontAsset(fontHashCode, out tempFont))
                        {
                            //if (tempFont != m_currentFontAsset)
                            //{
                            //    //Debug.Log("Assigning Font Asset: " + tempFont.name);
                            //    m_currentFontAsset = tempFont;
                            //    m_fontScale = (m_currentFontSize / m_currentFontAsset.fontInfo.PointSize * m_currentFontAsset.fontInfo.Scale * (m_isOrthographic ? 1 : 0.1f));
                            //}
                        }
                        else
                        {
                            // Load Font Asset
                            tempFont = Resources.Load<TMP_FontAsset>(TMP_Settings.defaultFontAssetPath + new string(m_htmlTag, m_xmlAttribute[0].valueStartIndex, m_xmlAttribute[0].valueLength));

                            if (tempFont == null)
                                return false;

                            // Add new reference to the font asset as well as default material to the MaterialReferenceManager
                            MaterialReferenceManager.AddFontAsset(tempFont);
                        }


                        // HANDLE NEW MATERIAL
                        if (materialAttributeHashCode == 0 && materialHashCode == 0)
                        {
                            // No material specified then use default font asset material.
                            m_currentMaterial = tempFont.material;

                            m_currentMaterialIndex = MaterialReference.AddMaterialReference(m_currentMaterial, tempFont, m_materialReferences, m_materialReferenceIndexLookup);

                            m_materialReferenceStack.Add(m_materialReferences[m_currentMaterialIndex]);
                        }
                        else if (materialAttributeHashCode == 103415287 || materialAttributeHashCode == 72669687) // using material attribute
                        {
                            if (MaterialReferenceManager.TryGetMaterial(materialHashCode, out tempMaterial))
                            {
                                m_currentMaterial = tempMaterial;

                                m_currentMaterialIndex = MaterialReference.AddMaterialReference(m_currentMaterial, tempFont, m_materialReferences, m_materialReferenceIndexLookup);

                                m_materialReferenceStack.Add(m_materialReferences[m_currentMaterialIndex]);
                            }
                            else
                            {
                                // Load new material
                                tempMaterial = Resources.Load<Material>(TMP_Settings.defaultFontAssetPath + new string(m_htmlTag, m_xmlAttribute[1].valueStartIndex, m_xmlAttribute[1].valueLength));

                                if (tempMaterial == null)
                                    return false;

                                // Add new reference to this material in the MaterialReferenceManager
                                MaterialReferenceManager.AddFontMaterial(materialHashCode, tempMaterial);

                                m_currentMaterial = tempMaterial;

                                m_currentMaterialIndex = MaterialReference.AddMaterialReference(m_currentMaterial, tempFont, m_materialReferences, m_materialReferenceIndexLookup);

                                m_materialReferenceStack.Add(m_materialReferences[m_currentMaterialIndex]);
                            }
                        }
                        else
                            return false;

                        m_currentFontAsset = tempFont;
                        m_fontScale = (m_currentFontSize / m_currentFontAsset.faceInfo.pointSize * m_currentFontAsset.faceInfo.scale * (m_isOrthographic ? 1 : 0.1f));

                        return true;
                    case 154158: // </font>
                    case 141358: // </FONT>
                        {
                            MaterialReference materialReference = m_materialReferenceStack.Remove();

                            m_currentFontAsset = materialReference.fontAsset;
                            m_currentMaterial = materialReference.material;
                            m_currentMaterialIndex = materialReference.index;

                            m_fontScale = (m_currentFontSize / m_currentFontAsset.faceInfo.pointSize * m_currentFontAsset.faceInfo.scale * (m_isOrthographic ? 1 : 0.1f));

                            return true;
                        }
                    case 103415287: // <material="material name">
                    case 72669687: // <MATERIAL>
                        materialHashCode = m_xmlAttribute[0].valueHashCode;

                        // Special handling for <material=default> or <material=Default>
                        if (materialHashCode == 764638571 || materialHashCode == 523367755)
                        {
                            // Check if material font atlas texture matches that of the current font asset.
                            //if (m_currentFontAsset.atlas.GetInstanceID() != m_currentMaterial.GetTexture(ShaderUtilities.ID_MainTex).GetInstanceID()) return false;

                            m_currentMaterial = m_materialReferences[0].material;
                            m_currentMaterialIndex = 0;

                            m_materialReferenceStack.Add(m_materialReferences[0]);

                            return true;
                        }


                        // Check if material 
                        if (MaterialReferenceManager.TryGetMaterial(materialHashCode, out tempMaterial))
                        {
                            // Check if material font atlas texture matches that of the current font asset.
                            //if (m_currentFontAsset.atlas.GetInstanceID() != tempMaterial.GetTexture(ShaderUtilities.ID_MainTex).GetInstanceID()) return false;

                            m_currentMaterial = tempMaterial;

                            m_currentMaterialIndex = MaterialReference.AddMaterialReference(m_currentMaterial, m_currentFontAsset, m_materialReferences, m_materialReferenceIndexLookup);

                            m_materialReferenceStack.Add(m_materialReferences[m_currentMaterialIndex]);
                        }
                        else
                        {
                            // Load new material
                            tempMaterial = Resources.Load<Material>(TMP_Settings.defaultFontAssetPath + new string(m_htmlTag, m_xmlAttribute[0].valueStartIndex, m_xmlAttribute[0].valueLength));

                            if (tempMaterial == null)
                                return false;

                            // Check if material font atlas texture matches that of the current font asset.
                            //if (m_currentFontAsset.atlas.GetInstanceID() != tempMaterial.GetTexture(ShaderUtilities.ID_MainTex).GetInstanceID()) return false;

                            // Add new reference to this material in the MaterialReferenceManager
                            MaterialReferenceManager.AddFontMaterial(materialHashCode, tempMaterial);

                            m_currentMaterial = tempMaterial;

                            m_currentMaterialIndex = MaterialReference.AddMaterialReference(m_currentMaterial, m_currentFontAsset , m_materialReferences, m_materialReferenceIndexLookup);

                            m_materialReferenceStack.Add(m_materialReferences[m_currentMaterialIndex]);
                        }
                        return true;
                    case 374360934: // </material>
                    case 343615334: // </MATERIAL>
                        {
                            //if (m_currentMaterial.GetTexture(ShaderUtilities.ID_MainTex).GetInstanceID() != m_materialReferenceStack.PreviousItem().material.GetTexture(ShaderUtilities.ID_MainTex).GetInstanceID())
                            //    return false;

                            MaterialReference materialReference = m_materialReferenceStack.Remove();

                            m_currentMaterial = materialReference.material;
                            m_currentMaterialIndex = materialReference.index;

                            return true;
                        }
                    case 320078: // <space=000.00>
                    case 230446: // <SPACE>
                        value = ConvertToFloat(m_htmlTag, m_xmlAttribute[0].valueStartIndex, m_xmlAttribute[0].valueLength);
                        if (value == -9999) return false;

                        switch (tagUnitType)
                        {
                            case TagUnitType.Pixels:
                                m_xAdvance += value * (m_isOrthographic ? 1 : 0.1f);
                                return true;
                            case TagUnitType.FontUnits:
                                m_xAdvance += value * (m_isOrthographic ? 1 : 0.1f) * m_currentFontSize;
                                return true;
                            case TagUnitType.Percentage:
                                // Not applicable
                                return false;
                        }
                        return false;
                    case 276254: // <alpha=#FF>
                    case 186622: // <ALPHA>
                        if (m_xmlAttribute[0].valueLength != 3) return false;

                        m_htmlColor.a = (byte)(HexToInt(m_htmlTag[7]) * 16 + HexToInt(m_htmlTag[8]));
                        return true;

                    case 1750458: // <a name=" ">
                        return false;
                    case 426: // </a>
                        return true;
                    case 43066: // <link="name">
                    case 30266: // <LINK>
                        if (m_isParsingText && !m_isCalculatingPreferredValues)
                        {
                            int index = m_textInfo.linkCount;

                            if (index + 1 > m_textInfo.linkInfo.Length)
                                TMP_TextInfo.Resize(ref m_textInfo.linkInfo, index + 1);

                            m_textInfo.linkInfo[index].textComponent = this;
                            m_textInfo.linkInfo[index].hashCode = m_xmlAttribute[0].valueHashCode;
                            m_textInfo.linkInfo[index].linkTextfirstCharacterIndex = m_characterCount;

                            m_textInfo.linkInfo[index].linkIdFirstCharacterIndex = startIndex + m_xmlAttribute[0].valueStartIndex;
                            m_textInfo.linkInfo[index].linkIdLength = m_xmlAttribute[0].valueLength;
                            m_textInfo.linkInfo[index].SetLinkID(m_htmlTag, m_xmlAttribute[0].valueStartIndex, m_xmlAttribute[0].valueLength);
                        }
                        return true;
                    case 155913: // </link>
                    case 143113: // </LINK>
                        if (m_isParsingText && !m_isCalculatingPreferredValues)
                        {
                            if (m_textInfo.linkCount < m_textInfo.linkInfo.Length)
                            {
                                m_textInfo.linkInfo[m_textInfo.linkCount].linkTextLength = m_characterCount - m_textInfo.linkInfo[m_textInfo.linkCount].linkTextfirstCharacterIndex;

                                m_textInfo.linkCount += 1;
                            }
                        }
                        return true;
                    case 275917: // <align=>
                    case 186285: // <ALIGN>
                        switch (m_xmlAttribute[0].valueHashCode)
                        {
                            case 3774683: // <align=left>
                                m_lineJustification = TextAlignmentOptions.Left;
                                m_lineJustificationStack.Add(m_lineJustification);
                                return true;
                            case 136703040: // <align=right>
                                m_lineJustification = TextAlignmentOptions.Right;
                                m_lineJustificationStack.Add(m_lineJustification);
                                return true;
                            case -458210101: // <align=center>
                                m_lineJustification = TextAlignmentOptions.Center;
                                m_lineJustificationStack.Add(m_lineJustification);
                                return true;
                            case -523808257: // <align=justified>
                                m_lineJustification = TextAlignmentOptions.Justified;
                                m_lineJustificationStack.Add(m_lineJustification);
                                return true;
                            case 122383428: // <align=flush>
                                m_lineJustification = TextAlignmentOptions.Flush;
                                m_lineJustificationStack.Add(m_lineJustification);
                                return true;
                        }
                        return false;
                    case 1065846: // </align>
                    case 976214: // </ALIGN>
                        m_lineJustification = m_lineJustificationStack.Remove();
                        return true;
                    case 327550: // <width=xx>
                    case 237918: // <WIDTH>
                        value = ConvertToFloat(m_htmlTag, m_xmlAttribute[0].valueStartIndex, m_xmlAttribute[0].valueLength);
                        if (value == -9999) return false;

                        switch (tagUnitType)
                        {
                            case TagUnitType.Pixels:
                                m_width = value * (m_isOrthographic ? 1 : 0.1f);
                                break;
                            case TagUnitType.FontUnits:
                                return false;
                            //break;
                            case TagUnitType.Percentage:
                                m_width = m_marginWidth * value / 100;
                                break;
                        }
                        return true;
                    case 1117479: // </width>
                    case 1027847: // </WIDTH>
                        m_width = -1;
                        return true;
                    // STYLE tag is now handled inline and replaced by its definition.
                    //case 322689: // <style="name">
                    //case 233057: // <STYLE>
                    //    TMP_Style style = TMP_StyleSheet.GetStyle(m_xmlAttribute[0].valueHashCode);

                    //    if (style == null) return false;

                    //    m_styleStack.Add(style.hashCode);

                    //    // Parse Style Macro
                    //    for (int i = 0; i < style.styleOpeningTagArray.Length; i++)
                    //    {
                    //        if (style.styleOpeningTagArray[i] == 60)
                    //        {
                    //            if (ValidateHtmlTag(style.styleOpeningTagArray, i + 1, out i) == false) return false;
                    //        }
                    //    }
                    //    return true;
                    //case 1112618: // </style>
                    //case 1022986: // </STYLE>
                    //    style = TMP_StyleSheet.GetStyle(m_xmlAttribute[0].valueHashCode);

                    //    if (style == null)
                    //    {
                    //        // Get style from the Style Stack
                    //        int styleHashCode = m_styleStack.CurrentItem();
                    //        style = TMP_StyleSheet.GetStyle(styleHashCode);

                    //        m_styleStack.Remove();
                    //    }

                    //    if (style == null) return false;
                    //    //// Parse Style Macro
                    //    for (int i = 0; i < style.styleClosingTagArray.Length; i++)
                    //    {
                    //        if (style.styleClosingTagArray[i] == 60)
                    //            ValidateHtmlTag(style.styleClosingTagArray, i + 1, out i);
                    //    }
                    //    return true;
                    case 281955: // <color> <color=#FF00FF> or <color=#FF00FF00>
                    case 192323: // <COLOR=#FF00FF>
                        // <color=#FFF> 3 Hex (short hand)
                        if (m_htmlTag[6] == 35 && tagCharCount == 10)
                        {
                            m_htmlColor = HexCharsToColor(m_htmlTag, tagCharCount);
                            m_colorStack.Add(m_htmlColor);
                            return true;
                        }
                        // <color=#FFF7> 4 Hex (short hand)
                        else if (m_htmlTag[6] == 35 && tagCharCount == 11)
                        {
                            m_htmlColor = HexCharsToColor(m_htmlTag, tagCharCount);
                            m_colorStack.Add(m_htmlColor);
                            return true;
                        }
                        // <color=#FF00FF> 3 Hex pairs
                        if (m_htmlTag[6] == 35 && tagCharCount == 13)
                        {
                            m_htmlColor = HexCharsToColor(m_htmlTag, tagCharCount);
                            m_colorStack.Add(m_htmlColor);
                            return true;
                        }
                        // <color=#FF00FF00> 4 Hex pairs
                        else if (m_htmlTag[6] == 35 && tagCharCount == 15)
                        {
                            m_htmlColor = HexCharsToColor(m_htmlTag, tagCharCount);
                            m_colorStack.Add(m_htmlColor);
                            return true;
                        }

                        // <color=name>
                        switch (m_xmlAttribute[0].valueHashCode)
                        {
                            case 125395: // <color=red>
                                m_htmlColor = Color.red;
                                m_colorStack.Add(m_htmlColor);
                                return true;
                            case 3573310: // <color=blue>
                                m_htmlColor = Color.blue;
                                m_colorStack.Add(m_htmlColor);
                                return true;
                            case 117905991: // <color=black>
                                m_htmlColor = Color.black;
                                m_colorStack.Add(m_htmlColor);
                                return true;
                            case 121463835: // <color=green>
                                m_htmlColor = Color.green;
                                m_colorStack.Add(m_htmlColor);
                                return true;
                            case 140357351: // <color=white>
                                m_htmlColor = Color.white;
                                m_colorStack.Add(m_htmlColor);
                                return true;
                            case 26556144: // <color=orange>
                                m_htmlColor = new Color32(255, 128, 0, 255);
                                m_colorStack.Add(m_htmlColor);
                                return true;
                            case -36881330: // <color=purple>
                                m_htmlColor = new Color32(160, 32, 240, 255);
                                m_colorStack.Add(m_htmlColor);
                                return true;
                            case 554054276: // <color=yellow>
                                m_htmlColor = Color.yellow;
                                m_colorStack.Add(m_htmlColor);
                                return true;
                        }
                        return false;

                    case 100149144: //<gradient>
                    case 69403544:  // <GRADIENT>
                        int gradientPresetHashCode = m_xmlAttribute[0].valueHashCode;
                        TMP_ColorGradient tempColorGradientPreset;

                        // Check if Color Gradient Preset has already been loaded.
                        if (MaterialReferenceManager.TryGetColorGradientPreset(gradientPresetHashCode, out tempColorGradientPreset))
                        {
                            m_colorGradientPreset = tempColorGradientPreset;
                        }
                        else
                        {
                            // Load Color Gradient Preset
                            if (tempColorGradientPreset == null)
                            {
                                tempColorGradientPreset = Resources.Load<TMP_ColorGradient>(TMP_Settings.defaultColorGradientPresetsPath + new string(m_htmlTag, m_xmlAttribute[0].valueStartIndex, m_xmlAttribute[0].valueLength));
                            }

                            if (tempColorGradientPreset == null)
                                return false;

                            MaterialReferenceManager.AddColorGradientPreset(gradientPresetHashCode, tempColorGradientPreset);
                            m_colorGradientPreset = tempColorGradientPreset;
                        }

                        m_colorGradientStack.Add(m_colorGradientPreset);

                        // TODO : Add support for defining preset in the tag itself

                        return true;

                    case 371094791: // </gradient>
                    case 340349191: // </GRADIENT>
                        m_colorGradientPreset = m_colorGradientStack.Remove();
                        return true;

                    case 1983971: // <cspace=xx.x>
                    case 1356515: // <CSPACE>
                        value = ConvertToFloat(m_htmlTag, m_xmlAttribute[0].valueStartIndex, m_xmlAttribute[0].valueLength);
                        if (value == -9999) return false;

                        switch (tagUnitType)
                        {
                            case TagUnitType.Pixels:
                                m_cSpacing = value * (m_isOrthographic ? 1 : 0.1f);
                                break;
                            case TagUnitType.FontUnits:
                                m_cSpacing = value * (m_isOrthographic ? 1 : 0.1f) * m_currentFontSize;
                                break;
                            case TagUnitType.Percentage:
                                return false;
                        }
                        return true;
                    case 7513474: // </cspace>
                    case 6886018: // </CSPACE>
                        if (!m_isParsingText) return true;

                        // Adjust xAdvance to remove extra space from last character.
                        if (m_characterCount > 0)
                        {
                            m_xAdvance -= m_cSpacing;
                            m_textInfo.characterInfo[m_characterCount - 1].xAdvance = m_xAdvance;
                        }
                        m_cSpacing = 0;
                        return true;
                    case 2152041: // <mspace=xx.x>
                    case 1524585: // <MSPACE>
                        value = ConvertToFloat(m_htmlTag, m_xmlAttribute[0].valueStartIndex, m_xmlAttribute[0].valueLength);
                        if (value == -9999) return false;

                        switch (tagUnitType)
                        {
                            case TagUnitType.Pixels:
                                m_monoSpacing = value * (m_isOrthographic ? 1 : 0.1f);
                                break;
                            case TagUnitType.FontUnits:
                                m_monoSpacing = value * (m_isOrthographic ? 1 : 0.1f) * m_currentFontSize;
                                break;
                            case TagUnitType.Percentage:
                                return false;
                        }
                        return true;
                    case 7681544: // </mspace>
                    case 7054088: // </MSPACE>
                        m_monoSpacing = 0;
                        return true;
                    case 280416: // <class="name">
                        return false;
                    case 1071884: // </color>
                    case 982252: // </COLOR>
                        m_htmlColor = m_colorStack.Remove();
                        return true;
                    case 2068980: // <indent=10px> <indent=10em> <indent=50%>
                    case 1441524: // <INDENT>
                        value = ConvertToFloat(m_htmlTag, m_xmlAttribute[0].valueStartIndex, m_xmlAttribute[0].valueLength);
                        if (value == -9999) return false;

                        switch (tagUnitType)
                        {
                            case TagUnitType.Pixels:
                                tag_Indent = value * (m_isOrthographic ? 1 : 0.1f);
                                break;
                            case TagUnitType.FontUnits:
                                tag_Indent = value * (m_isOrthographic ? 1 : 0.1f) * m_currentFontSize;
                                break;
                            case TagUnitType.Percentage:
                                tag_Indent = m_marginWidth * value / 100;
                                break;
                        }
                        m_indentStack.Add(tag_Indent);

                        m_xAdvance = tag_Indent;
                        return true;
                    case 7598483: // </indent>
                    case 6971027: // </INDENT>
                        tag_Indent = m_indentStack.Remove();
                        //m_xAdvance = tag_Indent;
                        return true;
                    case 1109386397: // <line-indent>
                    case -842656867: // <LINE-INDENT>
                        value = ConvertToFloat(m_htmlTag, m_xmlAttribute[0].valueStartIndex, m_xmlAttribute[0].valueLength);
                        if (value == -9999) return false;

                        switch (tagUnitType)
                        {
                            case TagUnitType.Pixels:
                                tag_LineIndent = value * (m_isOrthographic ? 1 : 0.1f);
                                break;
                            case TagUnitType.FontUnits:
                                tag_LineIndent = value * (m_isOrthographic ? 1 : 0.1f) * m_currentFontSize;
                                break;
                            case TagUnitType.Percentage:
                                tag_LineIndent = m_marginWidth * value / 100;
                                break;
                        }

                        m_xAdvance += tag_LineIndent;
                        return true;
                    case -445537194: // </line-indent>
                    case 1897386838: // </LINE-INDENT>
                        tag_LineIndent = 0;
                        return true;
                    case 2246877: // <sprite=x>
                    case 1619421: // <SPRITE>
                        int spriteAssetHashCode = m_xmlAttribute[0].valueHashCode;
                        TMP_SpriteAsset tempSpriteAsset;
                        m_spriteIndex = -1;

                        // CHECK TAG FORMAT
                        if (m_xmlAttribute[0].valueType == TagValueType.None || m_xmlAttribute[0].valueType == TagValueType.NumericalValue)
                        {
                            // No Sprite Asset is assigned to the text object
                            if (m_spriteAsset != null)
                            {
                                m_currentSpriteAsset = m_spriteAsset;
                            }
                            else if (m_defaultSpriteAsset != null)
                            {
                                m_currentSpriteAsset = m_defaultSpriteAsset;
                            }
                            else if (m_defaultSpriteAsset == null)
                            {
                                if (TMP_Settings.defaultSpriteAsset != null)
                                    m_defaultSpriteAsset = TMP_Settings.defaultSpriteAsset;
                                else
                                    m_defaultSpriteAsset = Resources.Load<TMP_SpriteAsset>("Sprite Assets/Default Sprite Asset");

                                m_currentSpriteAsset = m_defaultSpriteAsset;
                            }

                            // No valid sprite asset available
                            if (m_currentSpriteAsset == null)
                                return false;
                        }
                        else
                        {
                            // A Sprite Asset has been specified
                            if (MaterialReferenceManager.TryGetSpriteAsset(spriteAssetHashCode, out tempSpriteAsset))
                            {
                                m_currentSpriteAsset = tempSpriteAsset;
                            }
                            else
                            {
                                // Load Sprite Asset
                                if (tempSpriteAsset == null)
                                {
                                    tempSpriteAsset = Resources.Load<TMP_SpriteAsset>(TMP_Settings.defaultSpriteAssetPath + new string(m_htmlTag, m_xmlAttribute[0].valueStartIndex, m_xmlAttribute[0].valueLength));
                                }

                                if (tempSpriteAsset == null)
                                    return false;

                                //Debug.Log("Loading & assigning new Sprite Asset: " + tempSpriteAsset.name);
                                MaterialReferenceManager.AddSpriteAsset(spriteAssetHashCode, tempSpriteAsset);
                                m_currentSpriteAsset = tempSpriteAsset;
                            }
                        }

                        // Handling of <sprite=index> legacy tag format.
                        if (m_xmlAttribute[0].valueType == TagValueType.NumericalValue) // <sprite=index>
                        {
                            int index = (int)ConvertToFloat(m_htmlTag, m_xmlAttribute[0].valueStartIndex, m_xmlAttribute[0].valueLength);
                            if (index == -9999) return false;

                            // Check to make sure sprite index is valid
                            if (index > m_currentSpriteAsset.spriteCharacterTable.Count - 1) return false;

                            m_spriteIndex = index;
                        }

                        m_spriteColor = s_colorWhite;
                        m_tintSprite = false;

                        // Handle Sprite Tag Attributes
                        for (int i = 0; i < m_xmlAttribute.Length && m_xmlAttribute[i].nameHashCode != 0; i++)
                        {
                            //Debug.Log("Attribute[" + i + "].nameHashCode=" + m_xmlAttribute[i].nameHashCode + "   Value:" + ConvertToFloat(m_htmlTag, m_xmlAttribute[i].valueStartIndex, m_xmlAttribute[i].valueLength));
                            int nameHashCode = m_xmlAttribute[i].nameHashCode;
                            int index = 0;

                            switch (nameHashCode)
                            {
                                case 43347: // <sprite name="">
                                case 30547: // <SPRITE NAME="">
                                    m_currentSpriteAsset = TMP_SpriteAsset.SearchForSpriteByHashCode(m_currentSpriteAsset, m_xmlAttribute[i].valueHashCode, true, out index);
                                    if (index == -1) return false;

                                    m_spriteIndex = index;
                                    break;
                                case 295562: // <sprite index=>
                                case 205930: // <SPRITE INDEX=>
                                    index = (int)ConvertToFloat(m_htmlTag, m_xmlAttribute[1].valueStartIndex, m_xmlAttribute[1].valueLength);
                                    if (index == -9999) return false;

                                    // Check to make sure sprite index is valid
                                    if (index > m_currentSpriteAsset.spriteCharacterTable.Count - 1) return false;

                                    m_spriteIndex = index;
                                    break;
                                case 45819: // tint
                                case 33019: // TINT
                                    m_tintSprite = ConvertToFloat(m_htmlTag, m_xmlAttribute[i].valueStartIndex, m_xmlAttribute[i].valueLength) != 0;
                                    break;
                                case 281955: // color=#FF00FF80
                                case 192323: // COLOR
                                    m_spriteColor = HexCharsToColor(m_htmlTag, m_xmlAttribute[i].valueStartIndex, m_xmlAttribute[i].valueLength);
                                    break;
                                case 39505: // anim="0,16,12"  start, end, fps
                                case 26705: // ANIM
                                    //Debug.Log("Start: " + m_xmlAttribute[i].valueStartIndex + "  Length: " + m_xmlAttribute[i].valueLength);
                                    int paramCount = GetAttributeParameters(m_htmlTag, m_xmlAttribute[i].valueStartIndex, m_xmlAttribute[i].valueLength, ref m_attributeParameterValues);
                                    if (paramCount != 3) return false;

                                    m_spriteIndex = (int)m_attributeParameterValues[0];

                                    if (m_isParsingText)
                                    {
                                        // TODO : fix this!
                                        //if (m_attributeParameterValues[0] > m_currentSpriteAsset.spriteInfoList.Count - 1 || m_attributeParameterValues[1] > m_currentSpriteAsset.spriteInfoList.Count - 1)
                                        //    return false;

                                        spriteAnimator.DoSpriteAnimation(m_characterCount, m_currentSpriteAsset, m_spriteIndex, (int)m_attributeParameterValues[1], (int)m_attributeParameterValues[2]);
                                    }

                                    break;
                                //case 45545: // size
                                //case 32745: // SIZE

                                //    break;
                                default:
                                    if (nameHashCode != 2246877 && nameHashCode != 1619421)
                                        return false;
                                    break;
                            }
                        }

                        if (m_spriteIndex == -1) return false;

                        // Material HashCode for the Sprite Asset is the Sprite Asset Hash Code
                        m_currentMaterialIndex = MaterialReference.AddMaterialReference(m_currentSpriteAsset.material, m_currentSpriteAsset, m_materialReferences, m_materialReferenceIndexLookup);

                        m_textElementType = TMP_TextElementType.Sprite;
                        return true;
                    case 730022849: // <lowercase>
                    case 514803617: // <LOWERCASE>
                        m_FontStyleInternal |= FontStyles.LowerCase;
                        m_fontStyleStack.Add(FontStyles.LowerCase);
                        return true;
                    case -1668324918: // </lowercase>
                    case -1883544150: // </LOWERCASE>
                        if ((m_fontStyle & FontStyles.LowerCase) != FontStyles.LowerCase)
                        {
                            if (m_fontStyleStack.Remove(FontStyles.LowerCase) == 0)
                                m_FontStyleInternal &= ~FontStyles.LowerCase;
                        }
                        return true;
                    case 13526026: // <allcaps>
                    case 9133802: // <ALLCAPS>
                    case 781906058: // <uppercase>
                    case 566686826: // <UPPERCASE>
                        m_FontStyleInternal |= FontStyles.UpperCase;
                        m_fontStyleStack.Add(FontStyles.UpperCase);
                        return true;
                    case 52232547: // </allcaps>
                    case 47840323: // </ALLCAPS>
                    case -1616441709: // </uppercase>
                    case -1831660941: // </UPPERCASE>
                        if ((m_fontStyle & FontStyles.UpperCase) != FontStyles.UpperCase)
                        {
                            if (m_fontStyleStack.Remove(FontStyles.UpperCase) == 0)
                                m_FontStyleInternal &= ~FontStyles.UpperCase;
                        }
                        return true;
                    case 766244328: // <smallcaps>
                    case 551025096: // <SMALLCAPS>
                        m_FontStyleInternal |= FontStyles.SmallCaps;
                        m_fontStyleStack.Add(FontStyles.SmallCaps);
                        return true;
                    case -1632103439: // </smallcaps>
                    case -1847322671: // </SMALLCAPS>
                        if ((m_fontStyle & FontStyles.SmallCaps) != FontStyles.SmallCaps)
                        {
                            if (m_fontStyleStack.Remove(FontStyles.SmallCaps) == 0)
                                m_FontStyleInternal &= ~FontStyles.SmallCaps;
                        }
                        return true;
                    case 2109854: // <margin=00.0> <margin=00em> <margin=50%>
                    case 1482398: // <MARGIN>
                        // Check value type
                        switch (m_xmlAttribute[0].valueType)
                        {
                            case TagValueType.NumericalValue:
                                value = ConvertToFloat(m_htmlTag, m_xmlAttribute[0].valueStartIndex, m_xmlAttribute[0].valueLength); // px
                                if (value == -9999) return false;

                                // Determine tag unit type
                                switch (tagUnitType)
                                {
                                    case TagUnitType.Pixels:
                                        m_marginLeft = value * (m_isOrthographic ? 1 : 0.1f);
                                        break;
                                    case TagUnitType.FontUnits:
                                        m_marginLeft = value * (m_isOrthographic ? 1 : 0.1f) * m_currentFontSize;
                                        break;
                                    case TagUnitType.Percentage:
                                        m_marginLeft = (m_marginWidth - (m_width != -1 ? m_width : 0)) * value / 100;
                                        break;
                                }
                                m_marginLeft = m_marginLeft >= 0 ? m_marginLeft : 0;
                                m_marginRight = m_marginLeft;
                                return true;

                            case TagValueType.None:
                                for (int i = 1; i < m_xmlAttribute.Length && m_xmlAttribute[i].nameHashCode != 0; i++)
                                {
                                    // Get attribute name
                                    int nameHashCode = m_xmlAttribute[i].nameHashCode;

                                    switch (nameHashCode)
                                    {
                                        case 42823:  // <margin left=value>
                                            value = ConvertToFloat(m_htmlTag, m_xmlAttribute[i].valueStartIndex, m_xmlAttribute[i].valueLength); // px
                                            if (value == -9999) return false;

                                            switch (m_xmlAttribute[i].unitType)
                                            {
                                                case TagUnitType.Pixels:
                                                    m_marginLeft = value * (m_isOrthographic ? 1 : 0.1f);
                                                    break;
                                                case TagUnitType.FontUnits:
                                                    m_marginLeft = value * (m_isOrthographic ? 1 : 0.1f) * m_currentFontSize;
                                                    break;
                                                case TagUnitType.Percentage:
                                                    m_marginLeft = (m_marginWidth - (m_width != -1 ? m_width : 0)) * value / 100;
                                                    break;
                                            }
                                            m_marginLeft = m_marginLeft >= 0 ? m_marginLeft : 0;
                                            break;

                                        case 315620: // <margin right=value>
                                            value = ConvertToFloat(m_htmlTag, m_xmlAttribute[i].valueStartIndex, m_xmlAttribute[i].valueLength); // px
                                            if (value == -9999) return false;

                                            switch (m_xmlAttribute[i].unitType)
                                            {
                                                case TagUnitType.Pixels:
                                                    m_marginRight = value * (m_isOrthographic ? 1 : 0.1f);
                                                    break;
                                                case TagUnitType.FontUnits:
                                                    m_marginRight = value * (m_isOrthographic ? 1 : 0.1f) * m_currentFontSize;
                                                    break;
                                                case TagUnitType.Percentage:
                                                    m_marginRight = (m_marginWidth - (m_width != -1 ? m_width : 0)) * value / 100;
                                                    break;
                                            }
                                            m_marginRight = m_marginRight >= 0 ? m_marginRight : 0;
                                            break;
                                    }
                                }
                                return true;
                        }

                        return false;
                    case 7639357: // </margin>
                    case 7011901: // </MARGIN>
                        m_marginLeft = 0;
                        m_marginRight = 0;
                        return true;
                    case 1100728678: // <margin-left=xx.x>
                    case -855002522: // <MARGIN-LEFT>
                        value = ConvertToFloat(m_htmlTag, m_xmlAttribute[0].valueStartIndex, m_xmlAttribute[0].valueLength); // px
                        if (value == -9999) return false;

                        switch (tagUnitType)
                        {
                            case TagUnitType.Pixels:
                                m_marginLeft = value * (m_isOrthographic ? 1 : 0.1f);
                                break;
                            case TagUnitType.FontUnits:
                                m_marginLeft = value * (m_isOrthographic ? 1 : 0.1f) * m_currentFontSize;
                                break;
                            case TagUnitType.Percentage:
                                m_marginLeft = (m_marginWidth - (m_width != -1 ? m_width : 0)) * value / 100;
                                break;
                        }
                        m_marginLeft = m_marginLeft >= 0 ? m_marginLeft : 0;
                        return true;
                    case -884817987: // <margin-right=xx.x>
                    case -1690034531: // <MARGIN-RIGHT>
                        value = ConvertToFloat(m_htmlTag, m_xmlAttribute[0].valueStartIndex, m_xmlAttribute[0].valueLength); // px
                        if (value == -9999) return false;

                        switch (tagUnitType)
                        {
                            case TagUnitType.Pixels:
                                m_marginRight = value * (m_isOrthographic ? 1 : 0.1f);
                                break;
                            case TagUnitType.FontUnits:
                                m_marginRight = value * (m_isOrthographic ? 1 : 0.1f) * m_currentFontSize;
                                break;
                            case TagUnitType.Percentage:
                                m_marginRight = (m_marginWidth - (m_width != -1 ? m_width : 0)) * value / 100;
                                break;
                        }
                        m_marginRight = m_marginRight >= 0 ? m_marginRight : 0;
                        return true;
                    case 1109349752: // <line-height=xx.x>
                    case -842693512: // <LINE-HEIGHT>
                        value = ConvertToFloat(m_htmlTag, m_xmlAttribute[0].valueStartIndex, m_xmlAttribute[0].valueLength);
                        if (value == -9999 || value == 0) return false;

                        switch (tagUnitType)
                        {
                            case TagUnitType.Pixels:
                                m_lineHeight = value * (m_isOrthographic ? 1 : 0.1f); 
                                break;
                            case TagUnitType.FontUnits:
                                m_lineHeight = value * (m_isOrthographic ? 1 : 0.1f) * m_currentFontSize;
                                break;
                            case TagUnitType.Percentage:
                                m_lineHeight = m_fontAsset.faceInfo.lineHeight * value / 100 * m_fontScale;
                                break;
                        }
                        return true;
                    case -445573839: // </line-height>
                    case 1897350193: // </LINE-HEIGHT>
                        m_lineHeight = TMP_Math.FLOAT_UNSET;
                        return true;
                    case 15115642: // <noparse>
                    case 10723418: // <NOPARSE>
                        tag_NoParsing = true;
                        return true;
                    case 1913798: // <action>
                    case 1286342: // <ACTION>
                        int actionID = m_xmlAttribute[0].valueHashCode;

                        if (m_isParsingText)
                        {
                            m_actionStack.Add(actionID);

                            Debug.Log("Action ID: [" + actionID + "] First character index: " + m_characterCount);


                        }
                        //if (m_isParsingText)
                        //{
                        // TMP_Action action = TMP_Action.GetAction(m_xmlAttribute[0].valueHashCode);
                        //}
                        return true;
                    case 7443301: // </action>
                    case 6815845: // </ACTION>
                        if (m_isParsingText)
                        {
                            Debug.Log("Action ID: [" + m_actionStack.CurrentItem() + "] Last character index: " + (m_characterCount - 1));
                        }

                        m_actionStack.Remove();
                        return true;
                    case 315682: // <scale=xx.x>
                    case 226050: // <SCALE=xx.x>
                        value = ConvertToFloat(m_htmlTag, m_xmlAttribute[0].valueStartIndex, m_xmlAttribute[0].valueLength);
                        if (value == -9999) return false;

                        m_FXMatrix = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, new Vector3(value, 1, 1));
                        m_isFXMatrixSet = true;

                        return true;
                    case 1105611: // </scale>
                    case 1015979: // </SCALE>
                        m_isFXMatrixSet = false;
                        return true;
                    case 2227963: // <rotate=xx.x>
                    case 1600507: // <ROTATE=xx.x>
                        // TODO: Add ability to use Random Rotation

                        value = ConvertToFloat(m_htmlTag, m_xmlAttribute[0].valueStartIndex, m_xmlAttribute[0].valueLength);
                        if (value == -9999) return false;

                        m_FXMatrix = Matrix4x4.TRS(Vector3.zero, Quaternion.Euler(0, 0, value), Vector3.one);
                        m_isFXMatrixSet = true;

                        return true;
                    case 7757466: // </rotate>
                    case 7130010: // </ROTATE>
                        m_isFXMatrixSet = false;
                        return true;
                    case 317446: // <table>
                    case 227814: // <TABLE>
                        switch (m_xmlAttribute[1].nameHashCode)
                        {
                            case 327550: // width
                                float tableWidth = ConvertToFloat(m_htmlTag, m_xmlAttribute[1].valueStartIndex, m_xmlAttribute[1].valueLength);

                                switch (tagUnitType)
                                {
                                    case TagUnitType.Pixels:
                                        Debug.Log("Table width = " + tableWidth + "px.");
                                        break;
                                    case TagUnitType.FontUnits:
                                        Debug.Log("Table width = " + tableWidth + "em.");
                                        break;
                                    case TagUnitType.Percentage:
                                        Debug.Log("Table width = " + tableWidth + "%.");
                                        break;
                                }
                                break;
                        }
                        return true;
                    case 1107375: // </table>
                    case 1017743: // </TABLE>
                        return true;
                    case 926: // <tr>
                    case 670: // <TR>
                        return true;
                    case 3229: // </tr>
                    case 2973: // </TR>
                        return true;
                    case 916: // <th>
                    case 660: // <TH>
                        // Set style to bold and center alignment
                        return true;
                    case 3219: // </th>
                    case 2963: // </TH>
                        return true;
                    case 912: // <td>
                    case 656: // <TD>
                              // Style options
                        for (int i = 1; i < m_xmlAttribute.Length && m_xmlAttribute[i].nameHashCode != 0; i++)
                        {
                            switch (m_xmlAttribute[i].nameHashCode)
                            {
                                case 327550: // width
                                    float tableWidth = ConvertToFloat(m_htmlTag, m_xmlAttribute[i].valueStartIndex, m_xmlAttribute[i].valueLength);

                                    switch (tagUnitType)
                                    {
                                        case TagUnitType.Pixels:
                                            Debug.Log("Table width = " + tableWidth + "px.");
                                            break;
                                        case TagUnitType.FontUnits:
                                            Debug.Log("Table width = " + tableWidth + "em.");
                                            break;
                                        case TagUnitType.Percentage:
                                            Debug.Log("Table width = " + tableWidth + "%.");
                                            break;
                                    }
                                    break;
                                case 275917: // align
                                    switch (m_xmlAttribute[i].valueHashCode)
                                    {
                                        case 3774683: // left
                                            Debug.Log("TD align=\"left\".");
                                            break;
                                        case 136703040: // right
                                            Debug.Log("TD align=\"right\".");
                                            break;
                                        case -458210101: // center
                                            Debug.Log("TD align=\"center\".");
                                            break;
                                        case -523808257: // justified
                                            Debug.Log("TD align=\"justified\".");
                                            break;
                                    }
                                    break;
                            }
                        }

                        return true;
                    case 3215: // </td>
                    case 2959: // </TD>
                        return true;
                }
            }
            #endif
            #endregion

            return false;
        }
    }
}

using UnityEngine;
using UnityEngine.TextCore;

namespace TMPro
{
    public struct TMP_Vertex
    {
        public Vector3 position;
        public Vector2 uv;
        public Vector2 uv2;
        public Vector2 uv4;
        public Color32 color;

        //public Vector3 normal;
        //public Vector4 tangent;
    }

    /// <summary>
    /// Structure containing information about individual text elements (character or sprites).
    /// </summary>
    public struct TMP_CharacterInfo
    {
        public char character; // Should be changed to an int to handle UTF 32
        /// <summary>
        /// Index of the character in the raw string.
        /// </summary>
        public int index; // Index of the character in the input string.
        public int stringLength;
        public TMP_TextElementType elementType;

        public TMP_TextElement textElement;
        public TMP_FontAsset fontAsset;
        public TMP_SpriteAsset spriteAsset;
        public int spriteIndex;
        public Material material;
        public int materialReferenceIndex;
        public bool isUsingAlternateTypeface;

        public float pointSize;

        //public short wordNumber;
        public int lineNumber;
        //public short charNumber;
        public int pageNumber;


        public int vertexIndex;
        public TMP_Vertex vertex_BL;
        public TMP_Vertex vertex_TL;
        public TMP_Vertex vertex_TR;
        public TMP_Vertex vertex_BR;

        public Vector3 topLeft;
        public Vector3 bottomLeft;
        public Vector3 topRight;
        public Vector3 bottomRight;
        public float origin;
        public float ascender;
        public float baseLine;
        public float descender;

        public float xAdvance;
        public float aspectRatio;
        public float scale;
        public Color32 color;
        public Color32 underlineColor;
        public Color32 strikethroughColor;
        public Color32 highlightColor;
        public FontStyles style;
        public bool isVisible;
        //public bool isIgnoringAlignment;
    }
}

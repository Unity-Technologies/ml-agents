using System;
using UnityEngine;
using UnityEngine.TextCore;

namespace TMPro
{
    public enum TextElementType : byte
    {
        Character   = 0x1,
        Sprite      = 0x2,
    }

    /// <summary>
    /// Base class for all text elements like Character and SpriteCharacter.
    /// </summary>
    [Serializable]
    public class TMP_TextElement
    {
        /// <summary>
        /// The type of text element which can be a character or sprite.
        /// </summary>
        public TextElementType elementType { get { return m_ElementType; } }

        /// <summary>
        /// The unicode value (code point) of the character.
        /// </summary>
        public uint unicode { get { return m_Unicode; } set { m_Unicode = value; } }

        /// <summary>
        /// The glyph used by this text element.
        /// </summary>
        public Glyph glyph { get { return m_Glyph; } set { m_Glyph = value; } }

        /// <summary>
        /// The index of the glyph used by this text element.
        /// </summary>
        public uint glyphIndex { get { return m_GlyphIndex; } set { m_GlyphIndex = value; } }

        /// <summary>
        /// The relative scale of the character.
        /// </summary>
        public float scale { get { return m_Scale; } set { m_Scale = value; } }

        // =============================================
        // Private backing fields for public properties.
        // =============================================

        [SerializeField]
        protected TextElementType m_ElementType;

        [SerializeField]
        private uint m_Unicode;

        private Glyph m_Glyph;

        [SerializeField]
        private uint m_GlyphIndex;

        [SerializeField]
        private float m_Scale;
    }
}

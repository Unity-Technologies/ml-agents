using System;
using UnityEngine.TextCore;

namespace TMPro
{
    /// <summary>
    /// A basic element of text.
    /// </summary>
    [Serializable]
    public class TMP_Character : TMP_TextElement
    {
        /// <summary>
        /// Default constructor.
        /// </summary>
        public TMP_Character()
        {
            m_ElementType = TextElementType.Character;
            this.scale = 1.0f;
        }

        /// <summary>
        /// Constructor for new character
        /// </summary>
        /// <param name="unicode">Unicode value.</param>
        /// <param name="glyph">Glyph</param>
        public TMP_Character(uint unicode, Glyph glyph)
        {
            m_ElementType = TextElementType.Character;

            this.unicode = unicode;
            this.glyph = glyph;
            this.glyphIndex = glyph.index;
            this.scale = 1.0f;
        }

        /// <summary>
        /// Constructor for new character
        /// </summary>
        /// <param name="unicode">Unicode value.</param>
        /// <param name="glyphIndex">Glyph index.</param>
        internal TMP_Character(uint unicode, uint glyphIndex)
        {
            m_ElementType = TextElementType.Character;

            this.unicode = unicode;
            this.glyph = null;
            this.glyphIndex = glyphIndex;
            this.scale = 1.0f;
        }
    }
}

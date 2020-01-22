using System;
using UnityEngine;

namespace TMPro
{
    /// <summary>
    /// A basic element of text representing a pictograph, image, sprite or emoji.
    /// </summary>
    [Serializable]
    public class TMP_SpriteCharacter : TMP_TextElement
    {
        /// <summary>
        /// The name of the sprite element.
        /// </summary>
        public string name
        {
            get { return m_Name; }
            set
            {
                if (value == m_Name)
                    return;

                m_Name = value;
                m_HashCode = TMP_TextParsingUtilities.GetHashCodeCaseSensitive(m_Name);
            }
        }

        /// <summary>
        /// The hashcode value which is computed from the name of the sprite element.
        /// This value is read-only and updated when the name of the text sprite is changed.
        /// </summary>
        public int hashCode { get { return m_HashCode; } }


        // =============================================
        // Private backing fields for public properties.
        // =============================================

        [SerializeField]
        private string m_Name;

        [SerializeField]
        private int m_HashCode;


        // ********************
        // CONSTRUCTORS
        // ********************

        /// <summary>
        /// Default constructor.
        /// </summary>
        public TMP_SpriteCharacter()
        {
            m_ElementType = TextElementType.Sprite;
        }

        /// <summary>
        /// Constructor for new sprite character.
        /// </summary>
        /// <param name="unicode">Unicode value of the sprite character.</param>
        /// <param name="glyph">Glyph used by the sprite character.</param>
        public TMP_SpriteCharacter(uint unicode, TMP_SpriteGlyph glyph)
        {
            m_ElementType = TextElementType.Sprite;

            this.unicode = unicode;
            this.glyphIndex = glyph.index;
            this.glyph = glyph;
            this.scale = 1.0f;
        }

    }
}
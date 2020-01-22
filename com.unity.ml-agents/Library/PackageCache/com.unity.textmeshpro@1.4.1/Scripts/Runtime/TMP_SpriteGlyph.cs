using System;
using UnityEngine;
using UnityEngine.TextCore;

namespace TMPro
{
    /// <summary>
    /// The visual representation of the sprite character using this glyph.
    /// </summary>
    [Serializable]
    public class TMP_SpriteGlyph : Glyph
    {
        /// <summary>
        /// An optional reference to the underlying sprite used to create this glyph. 
        /// </summary>
        public Sprite sprite;


        // ********************
        // CONSTRUCTORS
        // ********************

        public TMP_SpriteGlyph() { }

        /// <summary>
        /// Constructor for new sprite glyph.
        /// </summary>
        /// <param name="index">Index of the sprite glyph.</param>
        /// <param name="metrics">Metrics which define the position of the glyph in the context of text layout.</param>
        /// <param name="glyphRect">GlyphRect which defines the coordinates of the glyph in the atlas texture.</param>
        /// <param name="scale">Scale of the glyph.</param>
        /// <param name="atlasIndex">Index of the atlas texture that contains the glyph.</param>      
        public TMP_SpriteGlyph(uint index, GlyphMetrics metrics, GlyphRect glyphRect, float scale, int atlasIndex)
        {
            this.index = index;
            this.metrics = metrics;
            this.glyphRect = glyphRect;
            this.scale = scale;
            this.atlasIndex = atlasIndex;
        }

        /// <summary>
        /// Constructor for new sprite glyph.
        /// </summary>
        /// <param name="index">>Index of the sprite glyph.</param>
        /// <param name="metrics">Metrics which define the position of the glyph in the context of text layout.</param>
        /// <param name="glyphRect">GlyphRect which defines the coordinates of the glyph in the atlas texture.</param>
        /// <param name="scale">Scale of the glyph.</param>
        /// <param name="atlasIndex">Index of the atlas texture that contains the glyph.</param>
        /// <param name="sprite">A reference to the Unity Sprite representing this sprite glyph.</param>
        public TMP_SpriteGlyph(uint index, GlyphMetrics metrics, GlyphRect glyphRect, float scale, int atlasIndex, Sprite sprite)
        {
            this.index = index;
            this.metrics = metrics;
            this.glyphRect = glyphRect;
            this.scale = scale;
            this.atlasIndex = atlasIndex;
            this.sprite = sprite;
        }
    }
}
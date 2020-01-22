using UnityEngine;
using System;
using System.Collections;


namespace TMPro
{

    /// <summary>
    /// Base class for all text elements like characters (glyphs) and sprites.
    /// </summary>
    [Serializable]
    public class TMP_TextElement_Legacy
    {
        public int id;
        public float x;
        public float y;
        public float width;
        public float height;
        public float xOffset;
        public float yOffset;
        public float xAdvance;
        public float scale;
    }
}

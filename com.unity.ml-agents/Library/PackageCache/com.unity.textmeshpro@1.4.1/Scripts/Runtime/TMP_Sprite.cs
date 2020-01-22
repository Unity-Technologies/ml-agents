using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;


namespace TMPro
{

    // Class which contains the Sprite Info for each sprite contained in the sprite asset.
    [Serializable]
    public class TMP_Sprite : TMP_TextElement_Legacy
    {
        //public int fileID;
        //public int id;
        public string name;
        public int hashCode;
        public int unicode;
        //public float x;
        //public float y;
        //public float width;
        //public float height;
        public Vector2 pivot;
        //public float xOffset; // Pivot X
        //public float yOffset; // Pivot Y
        //public float xAdvance;
        //public float scale;

        public Sprite sprite;
    }
}
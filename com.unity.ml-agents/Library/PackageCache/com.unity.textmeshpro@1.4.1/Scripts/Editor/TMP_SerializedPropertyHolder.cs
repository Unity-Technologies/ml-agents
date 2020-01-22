using UnityEngine;
using UnityEditor;

namespace TMPro
{
    class TMP_SerializedPropertyHolder : ScriptableObject
    {
        public TMP_FontAsset fontAsset;
        public uint firstCharacter;
        public uint secondCharacter;

        public TMP_GlyphPairAdjustmentRecord glyphPairAdjustmentRecord = new TMP_GlyphPairAdjustmentRecord(new TMP_GlyphAdjustmentRecord(), new TMP_GlyphAdjustmentRecord());
    }
}
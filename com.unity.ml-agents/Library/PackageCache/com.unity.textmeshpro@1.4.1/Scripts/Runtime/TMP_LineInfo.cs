namespace TMPro
{

    /// <summary>
    /// Structure which contains information about the individual lines of text.
    /// </summary>
    public struct TMP_LineInfo
    {
        internal int controlCharacterCount;

        public int characterCount;
        public int visibleCharacterCount;
        public int spaceCount;
        public int wordCount;
        public int firstCharacterIndex;
        public int firstVisibleCharacterIndex;
        public int lastCharacterIndex;
        public int lastVisibleCharacterIndex;

        public float length;
        public float lineHeight;
        public float ascender;
        public float baseline;
        public float descender;
        public float maxAdvance;

        public float width;
        public float marginLeft;
        public float marginRight;

        public TextAlignmentOptions alignment;
        public Extents lineExtents;


        /// <summary>
        /// Function returning the current line of text.
        /// </summary>
        /// <returns></returns>
        //public string GetLineText()
        //{
        //    string word = string.Empty;
        //    TMP_CharacterInfo[] charInfo = textComponent.textInfo.characterInfo;

        //    for (int i = firstCharacterIndex; i < lastCharacterIndex + 1; i++)
        //    {
        //        word += charInfo[i].character;
        //    }

        //    return word;
        //}
    }
}
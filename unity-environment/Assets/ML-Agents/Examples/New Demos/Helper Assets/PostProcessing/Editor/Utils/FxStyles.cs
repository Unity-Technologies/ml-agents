using UnityEngine;

namespace UnityEditor.PostProcessing
{
    public static class FxStyles
    {
        public static GUIStyle tickStyleRight;
        public static GUIStyle tickStyleLeft;
        public static GUIStyle tickStyleCenter;

        public static GUIStyle preSlider;
        public static GUIStyle preSliderThumb;
        public static GUIStyle preButton;
        public static GUIStyle preDropdown;

        public static GUIStyle preLabel;
        public static GUIStyle hueCenterCursor;
        public static GUIStyle hueRangeCursor;

        public static GUIStyle centeredBoldLabel;
        public static GUIStyle wheelThumb;
        public static Vector2 wheelThumbSize;

        public static GUIStyle header;
        public static GUIStyle headerCheckbox;
        public static GUIStyle headerFoldout;

        public static Texture2D playIcon;
        public static Texture2D checkerIcon;
        public static Texture2D paneOptionsIcon;

        public static GUIStyle centeredMiniLabel;

        static FxStyles()
        {
            tickStyleRight = new GUIStyle("Label")
            {
                alignment = TextAnchor.MiddleRight,
                fontSize = 9
            };

            tickStyleLeft = new GUIStyle("Label")
            {
                alignment = TextAnchor.MiddleLeft,
                fontSize = 9
            };

            tickStyleCenter = new GUIStyle("Label")
            {
                alignment = TextAnchor.MiddleCenter,
                fontSize = 9
            };

            preSlider = new GUIStyle("PreSlider");
            preSliderThumb = new GUIStyle("PreSliderThumb");
            preButton = new GUIStyle("PreButton");
            preDropdown = new GUIStyle("preDropdown");

            preLabel = new GUIStyle("ShurikenLabel");

            hueCenterCursor = new GUIStyle("ColorPicker2DThumb")
            {
                normal = { background = (Texture2D)EditorGUIUtility.LoadRequired("Builtin Skins/DarkSkin/Images/ShurikenPlus.png") },
                fixedWidth = 6,
                fixedHeight = 6
            };

            hueRangeCursor = new GUIStyle(hueCenterCursor)
            {
                normal = { background = (Texture2D)EditorGUIUtility.LoadRequired("Builtin Skins/DarkSkin/Images/CircularToggle_ON.png") }
            };

            wheelThumb = new GUIStyle("ColorPicker2DThumb");

            centeredBoldLabel = new GUIStyle(GUI.skin.GetStyle("Label"))
            {
                alignment = TextAnchor.UpperCenter,
                fontStyle = FontStyle.Bold
            };

            centeredMiniLabel = new GUIStyle(EditorStyles.centeredGreyMiniLabel)
            {
                alignment = TextAnchor.UpperCenter
            };

            wheelThumbSize = new Vector2(
                    !Mathf.Approximately(wheelThumb.fixedWidth, 0f) ? wheelThumb.fixedWidth : wheelThumb.padding.horizontal,
                    !Mathf.Approximately(wheelThumb.fixedHeight, 0f) ? wheelThumb.fixedHeight : wheelThumb.padding.vertical
                    );

            header = new GUIStyle("ShurikenModuleTitle")
            {
                font = (new GUIStyle("Label")).font,
                border = new RectOffset(15, 7, 4, 4),
                fixedHeight = 22,
                contentOffset = new Vector2(20f, -2f)
            };

            headerCheckbox = new GUIStyle("ShurikenCheckMark");
            headerFoldout = new GUIStyle("Foldout");

            playIcon = (Texture2D)EditorGUIUtility.LoadRequired("Builtin Skins/DarkSkin/Images/IN foldout act.png");
            checkerIcon = (Texture2D)EditorGUIUtility.LoadRequired("Icons/CheckerFloor.png");

            if (EditorGUIUtility.isProSkin)
                paneOptionsIcon = (Texture2D)EditorGUIUtility.LoadRequired("Builtin Skins/DarkSkin/Images/pane options.png");
            else
                paneOptionsIcon = (Texture2D)EditorGUIUtility.LoadRequired("Builtin Skins/LightSkin/Images/pane options.png");
        }
    }
}

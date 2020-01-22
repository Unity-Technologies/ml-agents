using UnityEngine;
using UnityEditor;
using System.Collections;


namespace TMPro.EditorUtilities
{

    public static class TMP_UIStyleManager
    {
        public static GUIStyle label;
        public static GUIStyle textAreaBoxWindow;
        public static GUIStyle boldFoldout;
        public static GUIStyle panelTitle;
        public static GUIStyle sectionHeader;
        public static GUIStyle centeredLabel;
        public static GUIStyle rightLabel;
        public static GUIStyle wrappingTextArea;

        public static GUIStyle alignmentButtonLeft;
        public static GUIStyle alignmentButtonMid;
        public static GUIStyle alignmentButtonRight;

        // Alignment Button Textures
        public static Texture2D alignLeft;
        public static Texture2D alignCenter;
        public static Texture2D alignRight;
        public static Texture2D alignJustified;
        public static Texture2D alignFlush;
        public static Texture2D alignGeoCenter;
        public static Texture2D alignTop;
        public static Texture2D alignMiddle;
        public static Texture2D alignBottom;
        public static Texture2D alignBaseline;
        public static Texture2D alignMidline;
        public static Texture2D alignCapline;
        public static Texture2D sectionHeaderTexture;
        
        public static GUIContent[] alignContentA;
        public static GUIContent[] alignContentB;

        static TMP_UIStyleManager()
        {
            // Find to location of the TextMesh Pro Asset Folder (as users may have moved it)
            var tmproAssetFolderPath = TMP_EditorUtility.packageRelativePath;

            if (EditorGUIUtility.isProSkin)
            {
                alignLeft = AssetDatabase.LoadAssetAtPath(tmproAssetFolderPath + "/Editor Resources/Textures/btn_AlignLeft.psd", typeof(Texture2D)) as Texture2D;
                alignCenter = AssetDatabase.LoadAssetAtPath(tmproAssetFolderPath + "/Editor Resources/Textures/btn_AlignCenter.psd", typeof(Texture2D)) as Texture2D;
                alignRight = AssetDatabase.LoadAssetAtPath(tmproAssetFolderPath + "/Editor Resources/Textures/btn_AlignRight.psd", typeof(Texture2D)) as Texture2D;
                alignJustified = AssetDatabase.LoadAssetAtPath(tmproAssetFolderPath + "/Editor Resources/Textures/btn_AlignJustified.psd", typeof(Texture2D)) as Texture2D;
                alignFlush = AssetDatabase.LoadAssetAtPath(tmproAssetFolderPath + "/Editor Resources/Textures/btn_AlignFlush.psd", typeof(Texture2D)) as Texture2D;
                alignGeoCenter = AssetDatabase.LoadAssetAtPath(tmproAssetFolderPath + "/Editor Resources/Textures/btn_AlignCenterGeo.psd", typeof(Texture2D)) as Texture2D;
                alignTop = AssetDatabase.LoadAssetAtPath(tmproAssetFolderPath + "/Editor Resources/Textures/btn_AlignTop.psd", typeof(Texture2D)) as Texture2D;
                alignMiddle = AssetDatabase.LoadAssetAtPath(tmproAssetFolderPath + "/Editor Resources/Textures/btn_AlignMiddle.psd", typeof(Texture2D)) as Texture2D;
                alignBottom = AssetDatabase.LoadAssetAtPath(tmproAssetFolderPath + "/Editor Resources/Textures/btn_AlignBottom.psd", typeof(Texture2D)) as Texture2D;
                alignBaseline = AssetDatabase.LoadAssetAtPath(tmproAssetFolderPath + "/Editor Resources/Textures/btn_AlignBaseLine.psd", typeof(Texture2D)) as Texture2D;
                alignMidline = AssetDatabase.LoadAssetAtPath(tmproAssetFolderPath + "/Editor Resources/Textures/btn_AlignMidLine.psd", typeof(Texture2D)) as Texture2D;
                alignCapline = AssetDatabase.LoadAssetAtPath(tmproAssetFolderPath + "/Editor Resources/Textures/btn_AlignCapLine.psd", typeof(Texture2D)) as Texture2D;
                sectionHeaderTexture = AssetDatabase.LoadAssetAtPath(tmproAssetFolderPath + "/Editor Resources/Textures/SectionHeader_Dark.psd", typeof(Texture2D)) as Texture2D;
            }
            else
            {
                alignLeft = AssetDatabase.LoadAssetAtPath(tmproAssetFolderPath + "/Editor Resources/Textures/btn_AlignLeft_Light.psd", typeof(Texture2D)) as Texture2D;
                alignCenter = AssetDatabase.LoadAssetAtPath(tmproAssetFolderPath + "/Editor Resources/Textures/btn_AlignCenter_Light.psd", typeof(Texture2D)) as Texture2D;
                alignRight = AssetDatabase.LoadAssetAtPath(tmproAssetFolderPath + "/Editor Resources/Textures/btn_AlignRight_Light.psd", typeof(Texture2D)) as Texture2D;
                alignJustified = AssetDatabase.LoadAssetAtPath(tmproAssetFolderPath + "/Editor Resources/Textures/btn_AlignJustified_Light.psd", typeof(Texture2D)) as Texture2D;
                alignFlush = AssetDatabase.LoadAssetAtPath(tmproAssetFolderPath + "/Editor Resources/Textures/btn_AlignFlush_Light.psd", typeof(Texture2D)) as Texture2D;
                alignGeoCenter = AssetDatabase.LoadAssetAtPath(tmproAssetFolderPath + "/Editor Resources/Textures/btn_AlignCenterGeo_Light.psd", typeof(Texture2D)) as Texture2D;
                alignTop = AssetDatabase.LoadAssetAtPath(tmproAssetFolderPath + "/Editor Resources/Textures/btn_AlignTop_Light.psd", typeof(Texture2D)) as Texture2D;
                alignMiddle = AssetDatabase.LoadAssetAtPath(tmproAssetFolderPath + "/Editor Resources/Textures/btn_AlignMiddle_Light.psd", typeof(Texture2D)) as Texture2D;
                alignBottom = AssetDatabase.LoadAssetAtPath(tmproAssetFolderPath + "/Editor Resources/Textures/btn_AlignBottom_Light.psd", typeof(Texture2D)) as Texture2D;
                alignBaseline = AssetDatabase.LoadAssetAtPath(tmproAssetFolderPath + "/Editor Resources/Textures/btn_AlignBaseLine_Light.psd", typeof(Texture2D)) as Texture2D;
                alignMidline = AssetDatabase.LoadAssetAtPath(tmproAssetFolderPath + "/Editor Resources/Textures/btn_AlignMidLine_Light.psd", typeof(Texture2D)) as Texture2D;
                alignCapline = AssetDatabase.LoadAssetAtPath(tmproAssetFolderPath + "/Editor Resources/Textures/btn_AlignCapLine_Light.psd", typeof(Texture2D)) as Texture2D;
                sectionHeaderTexture = AssetDatabase.LoadAssetAtPath(tmproAssetFolderPath + "/Editor Resources/Textures/SectionHeader_Light.psd", typeof(Texture2D)) as Texture2D;
            }

            label = new GUIStyle(EditorStyles.label) { richText = true, wordWrap = true, stretchWidth = true };
            textAreaBoxWindow = new GUIStyle(EditorStyles.textArea) { richText = true };
            boldFoldout = new GUIStyle(EditorStyles.foldout) { fontStyle = FontStyle.Bold };
            panelTitle = new GUIStyle(EditorStyles.label) { fontStyle = FontStyle.Bold };

            sectionHeader = new GUIStyle(EditorStyles.label) { fixedHeight = 22, richText = true, border = new RectOffset(9, 9, 0, 0), overflow = new RectOffset(9, 0, 0, 0), padding = new RectOffset(0, 0, 4, 0) };
            sectionHeader.normal.background = sectionHeaderTexture;

            centeredLabel = new GUIStyle(EditorStyles.label) { alignment = TextAnchor.MiddleCenter};
            rightLabel = new GUIStyle(EditorStyles.label) { alignment = TextAnchor.MiddleRight, richText = true };


            alignmentButtonLeft = new GUIStyle(EditorStyles.miniButtonLeft);
            alignmentButtonLeft.padding.left = 4;
            alignmentButtonLeft.padding.right = 4;
            alignmentButtonLeft.padding.top = 2;
            alignmentButtonLeft.padding.bottom = 2;

            alignmentButtonMid = new GUIStyle(EditorStyles.miniButtonMid);
            alignmentButtonMid.padding.left = 4;
            alignmentButtonMid.padding.right = 4;
            alignmentButtonLeft.padding.top = 2;
            alignmentButtonLeft.padding.bottom = 2;

            alignmentButtonRight = new GUIStyle(EditorStyles.miniButtonRight);
            alignmentButtonRight.padding.left = 4;
            alignmentButtonRight.padding.right = 4;
            alignmentButtonLeft.padding.top = 2;
            alignmentButtonLeft.padding.bottom = 2;

            wrappingTextArea = new GUIStyle(EditorStyles.textArea);
            wrappingTextArea.wordWrap = true;

            alignContentA = new []
            { 
                new GUIContent(alignLeft, "Left"), 
                new GUIContent(alignCenter, "Center"), 
                new GUIContent(alignRight, "Right"), 
                new GUIContent(alignJustified, "Justified"),
                new GUIContent(alignFlush, "Flush"),
                new GUIContent(alignGeoCenter, "Geometry Center")
            };

            alignContentB = new []
            { 
                new GUIContent(alignTop, "Top"), 
                new GUIContent(alignMiddle, "Middle"), 
                new GUIContent(alignBottom, "Bottom"),
                new GUIContent(alignBaseline, "Baseline"),
                new GUIContent(alignMidline, "Midline"),
                new GUIContent(alignCapline, "Capline")
            };
        }
    }
}

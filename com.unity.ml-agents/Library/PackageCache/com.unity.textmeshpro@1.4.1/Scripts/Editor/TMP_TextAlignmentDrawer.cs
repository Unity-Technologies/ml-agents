using UnityEngine;
using UnityEditor;

namespace TMPro.EditorUtilities
{

    [CustomPropertyDrawer(typeof(TextAlignmentOptions))]
    public class TMP_TextAlignmentDrawer : PropertyDrawer
    {
        const int k_AlignmentButtonWidth = 24;
        const int k_AlignmentButtonHeight = 20;
        const int k_WideViewWidth = 504;
        const int k_ControlsSpacing = 6;
        const int k_GroupWidth = k_AlignmentButtonWidth * 6;
        static readonly int k_TextAlignmentHash = "DoTextAligmentControl".GetHashCode();

        public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
        {
            return EditorGUIUtility.currentViewWidth > k_WideViewWidth ? k_AlignmentButtonHeight : k_AlignmentButtonHeight * 2 + 3;
        }

        public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
        {
            var id = GUIUtility.GetControlID(k_TextAlignmentHash, FocusType.Keyboard, position);
            
            EditorGUI.BeginProperty(position, label, property);
            {
                var controlArea = EditorGUI.PrefixLabel(position, id, label);
                
                var horizontalAligment = new Rect(controlArea.x, controlArea.y, k_GroupWidth, k_AlignmentButtonHeight);
                var verticalAligment = new Rect(!(EditorGUIUtility.currentViewWidth > k_WideViewWidth) ? controlArea.x : horizontalAligment.xMax + k_ControlsSpacing, !(EditorGUIUtility.currentViewWidth > k_WideViewWidth) ? controlArea.y + k_AlignmentButtonHeight + 3 : controlArea.y, k_GroupWidth, k_AlignmentButtonHeight);

                EditorGUI.BeginChangeCheck();

                var selectedHorizontal = DoHorizontalAligmentControl(horizontalAligment, property);
                var selectedVertical = DoVerticalAligmentControl(verticalAligment, property);

                if (EditorGUI.EndChangeCheck())
                {
                    var value = (0x1 << selectedHorizontal) | (0x100 << selectedVertical);
                    property.intValue = value;
                }
            }
            EditorGUI.EndProperty();
        }

        static int DoHorizontalAligmentControl(Rect position, SerializedProperty alignment)
        {
            var selected = TMP_EditorUtility.GetHorizontalAlignmentGridValue(alignment.intValue);

            var values = new bool[6];

            values[selected] = true;

            if (alignment.hasMultipleDifferentValues)
            {
                foreach (var obj in alignment.serializedObject.targetObjects)
                {
                    var text = obj as TMP_Text;
                    if (text != null)
                    {
                        values[TMP_EditorUtility.GetHorizontalAlignmentGridValue((int)text.alignment)] = true;
                    }
                }
            }

            position.width = k_AlignmentButtonWidth;

            for (var i = 0; i < values.Length; i++)
            {
                var oldValue = values[i];
                var newValue = TMP_EditorUtility.EditorToggle(position, oldValue, TMP_UIStyleManager.alignContentA[i], i == 0 ? TMP_UIStyleManager.alignmentButtonLeft : (i == 5 ? TMP_UIStyleManager.alignmentButtonRight : TMP_UIStyleManager.alignmentButtonMid));
                if (newValue != oldValue)
                {
                    selected = i;
                }
                position.x += position.width;
            }

            return selected;
        }

        static int DoVerticalAligmentControl(Rect position, SerializedProperty alignment)
        {
            var selected = TMP_EditorUtility.GetVerticalAlignmentGridValue(alignment.intValue);

            var values = new bool[6];

            values[selected] = true;

            if (alignment.hasMultipleDifferentValues)
            {
                foreach (var obj in alignment.serializedObject.targetObjects)
                {
                    var text = obj as TMP_Text;
                    if (text != null)
                    {
                        values[TMP_EditorUtility.GetVerticalAlignmentGridValue((int)text.alignment)] = true;
                    }
                }
            }

            position.width = k_AlignmentButtonWidth;

            for (var i = 0; i < values.Length; i++)
            {
                var oldValue = values[i];
                var newValue = TMP_EditorUtility.EditorToggle(position, oldValue, TMP_UIStyleManager.alignContentB[i], i == 0 ? TMP_UIStyleManager.alignmentButtonLeft : (i == 5 ? TMP_UIStyleManager.alignmentButtonRight : TMP_UIStyleManager.alignmentButtonMid));
                if (newValue != oldValue)
                {
                    selected = i;
                }
                position.x += position.width;
            }

            return selected;
        }
    }
}

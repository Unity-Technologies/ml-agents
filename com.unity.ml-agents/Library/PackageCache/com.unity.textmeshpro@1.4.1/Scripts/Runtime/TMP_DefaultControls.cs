using UnityEngine;
using System.Collections;
using UnityEngine.UI;


namespace TMPro
{

    public static class TMP_DefaultControls
    {
        public struct Resources
        {
            public Sprite standard;
            public Sprite background;
            public Sprite inputField;
            public Sprite knob;
            public Sprite checkmark;
            public Sprite dropdown;
            public Sprite mask;
        }

        private const float kWidth = 160f;
        private const float kThickHeight = 30f;
        private const float kThinHeight = 20f;
        private static Vector2 s_ThickElementSize = new Vector2(kWidth, kThickHeight);
        private static Vector2 s_ThinElementSize = new Vector2(kWidth, kThinHeight);
        //private static Vector2 s_ImageElementSize = new Vector2(100f, 100f);
        private static Color s_DefaultSelectableColor = new Color(1f, 1f, 1f, 1f);
        //private static Color s_PanelColor = new Color(1f, 1f, 1f, 0.392f);
        private static Color s_TextColor = new Color(50f / 255f, 50f / 255f, 50f / 255f, 1f);


        private static GameObject CreateUIElementRoot(string name, Vector2 size)
        {
            GameObject child = new GameObject(name);
            RectTransform rectTransform = child.AddComponent<RectTransform>();
            rectTransform.sizeDelta = size;
            return child;
        }

        static GameObject CreateUIObject(string name, GameObject parent)
        {
            GameObject go = new GameObject(name);
            go.AddComponent<RectTransform>();
            SetParentAndAlign(go, parent);
            return go;
        }

        private static void SetDefaultTextValues(TMP_Text lbl)
        {
            // Set text values we want across UI elements in default controls.
            // Don't set values which are the same as the default values for the Text component,
            // since there's no point in that, and it's good to keep them as consistent as possible.
            lbl.color = s_TextColor;
            lbl.fontSize = 14;
        }

        private static void SetDefaultColorTransitionValues(Selectable slider)
        {
            ColorBlock colors = slider.colors;
            colors.highlightedColor = new Color(0.882f, 0.882f, 0.882f);
            colors.pressedColor = new Color(0.698f, 0.698f, 0.698f);
            colors.disabledColor = new Color(0.521f, 0.521f, 0.521f);
        }

        private static void SetParentAndAlign(GameObject child, GameObject parent)
        {
            if (parent == null)
                return;

            child.transform.SetParent(parent.transform, false);
            SetLayerRecursively(child, parent.layer);
        }

        private static void SetLayerRecursively(GameObject go, int layer)
        {
            go.layer = layer;
            Transform t = go.transform;
            for (int i = 0; i < t.childCount; i++)
                SetLayerRecursively(t.GetChild(i).gameObject, layer);
        }

        // Actual controls

        public static GameObject CreateScrollbar(Resources resources)
        {
            // Create GOs Hierarchy
            GameObject scrollbarRoot = CreateUIElementRoot("Scrollbar", s_ThinElementSize);

            GameObject sliderArea = CreateUIObject("Sliding Area", scrollbarRoot);
            GameObject handle = CreateUIObject("Handle", sliderArea);

            Image bgImage = scrollbarRoot.AddComponent<Image>();
            bgImage.sprite = resources.background;
            bgImage.type = Image.Type.Sliced;
            bgImage.color = s_DefaultSelectableColor;

            Image handleImage = handle.AddComponent<Image>();
            handleImage.sprite = resources.standard;
            handleImage.type = Image.Type.Sliced;
            handleImage.color = s_DefaultSelectableColor;

            RectTransform sliderAreaRect = sliderArea.GetComponent<RectTransform>();
            sliderAreaRect.sizeDelta = new Vector2(-20, -20);
            sliderAreaRect.anchorMin = Vector2.zero;
            sliderAreaRect.anchorMax = Vector2.one;

            RectTransform handleRect = handle.GetComponent<RectTransform>();
            handleRect.sizeDelta = new Vector2(20, 20);

            Scrollbar scrollbar = scrollbarRoot.AddComponent<Scrollbar>();
            scrollbar.handleRect = handleRect;
            scrollbar.targetGraphic = handleImage;
            SetDefaultColorTransitionValues(scrollbar);

            return scrollbarRoot;
        }

        public static GameObject CreateButton(Resources resources)
        {
            GameObject buttonRoot = CreateUIElementRoot("Button", s_ThickElementSize);

            GameObject childText = new GameObject("Text (TMP)");
            childText.AddComponent<RectTransform>();
            SetParentAndAlign(childText, buttonRoot);

            Image image = buttonRoot.AddComponent<Image>();
            image.sprite = resources.standard;
            image.type = Image.Type.Sliced;
            image.color = s_DefaultSelectableColor;

            Button bt = buttonRoot.AddComponent<Button>();
            SetDefaultColorTransitionValues(bt);

            TextMeshProUGUI text = childText.AddComponent<TextMeshProUGUI>();
            text.text = "Button";
            text.alignment = TextAlignmentOptions.Center;
            SetDefaultTextValues(text);

            RectTransform textRectTransform = childText.GetComponent<RectTransform>();
            textRectTransform.anchorMin = Vector2.zero;
            textRectTransform.anchorMax = Vector2.one;
            textRectTransform.sizeDelta = Vector2.zero;

            return buttonRoot;
        }

        public static GameObject CreateText(Resources resources)
        {
            GameObject go = CreateUIElementRoot("Text (TMP)", s_ThickElementSize);

            TextMeshProUGUI lbl = go.AddComponent<TextMeshProUGUI>();
            lbl.text = "New Text";
            SetDefaultTextValues(lbl);

            return go;
        }


        public static GameObject CreateInputField(Resources resources)
        {
            GameObject root = CreateUIElementRoot("InputField (TMP)", s_ThickElementSize);

            GameObject textArea = CreateUIObject("Text Area", root);
            GameObject childPlaceholder = CreateUIObject("Placeholder", textArea);
            GameObject childText = CreateUIObject("Text", textArea);

            Image image = root.AddComponent<Image>();
            image.sprite = resources.inputField;
            image.type = Image.Type.Sliced;
            image.color = s_DefaultSelectableColor;

            TMP_InputField inputField = root.AddComponent<TMP_InputField>();
            SetDefaultColorTransitionValues(inputField);

            // Use UI.Mask for Unity 5.0 - 5.1 and 2D RectMask for Unity 5.2 and up
            textArea.AddComponent<RectMask2D>();

            RectTransform textAreaRectTransform = textArea.GetComponent<RectTransform>();
            textAreaRectTransform.anchorMin = Vector2.zero;
            textAreaRectTransform.anchorMax = Vector2.one;
            textAreaRectTransform.sizeDelta = Vector2.zero;
            textAreaRectTransform.offsetMin = new Vector2(10, 6);
            textAreaRectTransform.offsetMax = new Vector2(-10, -7);


            TextMeshProUGUI text = childText.AddComponent<TextMeshProUGUI>();
            text.text = "";
            text.enableWordWrapping = false;
            text.extraPadding = true;
            text.richText = true;
            SetDefaultTextValues(text);

            TextMeshProUGUI placeholder = childPlaceholder.AddComponent<TextMeshProUGUI>();
            placeholder.text = "Enter text...";
            placeholder.fontSize = 14;
            placeholder.fontStyle = FontStyles.Italic;
            placeholder.enableWordWrapping = false;
            placeholder.extraPadding = true;

            // Make placeholder color half as opaque as normal text color.
            Color placeholderColor = text.color;
            placeholderColor.a *= 0.5f;
            placeholder.color = placeholderColor;

            RectTransform textRectTransform = childText.GetComponent<RectTransform>();
            textRectTransform.anchorMin = Vector2.zero;
            textRectTransform.anchorMax = Vector2.one;
            textRectTransform.sizeDelta = Vector2.zero;
            textRectTransform.offsetMin = new Vector2(0, 0);
            textRectTransform.offsetMax = new Vector2(0, 0);

            RectTransform placeholderRectTransform = childPlaceholder.GetComponent<RectTransform>();
            placeholderRectTransform.anchorMin = Vector2.zero;
            placeholderRectTransform.anchorMax = Vector2.one;
            placeholderRectTransform.sizeDelta = Vector2.zero;
            placeholderRectTransform.offsetMin = new Vector2(0, 0);
            placeholderRectTransform.offsetMax = new Vector2(0, 0);

            inputField.textViewport = textAreaRectTransform;
            inputField.textComponent = text;
            inputField.placeholder = placeholder;
            inputField.fontAsset = text.font;

            return root;
        }

        public static GameObject CreateDropdown(Resources resources)
        {
            GameObject root = CreateUIElementRoot("Dropdown", s_ThickElementSize);

            GameObject label = CreateUIObject("Label", root);
            GameObject arrow = CreateUIObject("Arrow", root);
            GameObject template = CreateUIObject("Template", root);
            GameObject viewport = CreateUIObject("Viewport", template);
            GameObject content = CreateUIObject("Content", viewport);
            GameObject item = CreateUIObject("Item", content);
            GameObject itemBackground = CreateUIObject("Item Background", item);
            GameObject itemCheckmark = CreateUIObject("Item Checkmark", item);
            GameObject itemLabel = CreateUIObject("Item Label", item);

            // Sub controls.

            GameObject scrollbar = CreateScrollbar(resources);
            scrollbar.name = "Scrollbar";
            SetParentAndAlign(scrollbar, template);

            Scrollbar scrollbarScrollbar = scrollbar.GetComponent<Scrollbar>();
            scrollbarScrollbar.SetDirection(Scrollbar.Direction.BottomToTop, true);

            RectTransform vScrollbarRT = scrollbar.GetComponent<RectTransform>();
            vScrollbarRT.anchorMin = Vector2.right;
            vScrollbarRT.anchorMax = Vector2.one;
            vScrollbarRT.pivot = Vector2.one;
            vScrollbarRT.sizeDelta = new Vector2(vScrollbarRT.sizeDelta.x, 0);

            // Setup item UI components.

            TextMeshProUGUI itemLabelText = itemLabel.AddComponent<TextMeshProUGUI>();
            SetDefaultTextValues(itemLabelText);
            itemLabelText.alignment = TextAlignmentOptions.Left;

            Image itemBackgroundImage = itemBackground.AddComponent<Image>();
            itemBackgroundImage.color = new Color32(245, 245, 245, 255);

            Image itemCheckmarkImage = itemCheckmark.AddComponent<Image>();
            itemCheckmarkImage.sprite = resources.checkmark;

            Toggle itemToggle = item.AddComponent<Toggle>();
            itemToggle.targetGraphic = itemBackgroundImage;
            itemToggle.graphic = itemCheckmarkImage;
            itemToggle.isOn = true;

            // Setup template UI components.

            Image templateImage = template.AddComponent<Image>();
            templateImage.sprite = resources.standard;
            templateImage.type = Image.Type.Sliced;

            ScrollRect templateScrollRect = template.AddComponent<ScrollRect>();
            templateScrollRect.content = (RectTransform)content.transform;
            templateScrollRect.viewport = (RectTransform)viewport.transform;
            templateScrollRect.horizontal = false;
            templateScrollRect.movementType = ScrollRect.MovementType.Clamped;
            templateScrollRect.verticalScrollbar = scrollbarScrollbar;
            templateScrollRect.verticalScrollbarVisibility = ScrollRect.ScrollbarVisibility.AutoHideAndExpandViewport;
            templateScrollRect.verticalScrollbarSpacing = -3;

            Mask scrollRectMask = viewport.AddComponent<Mask>();
            scrollRectMask.showMaskGraphic = false;

            Image viewportImage = viewport.AddComponent<Image>();
            viewportImage.sprite = resources.mask;
            viewportImage.type = Image.Type.Sliced;

            // Setup dropdown UI components.

            TextMeshProUGUI labelText = label.AddComponent<TextMeshProUGUI>();
            SetDefaultTextValues(labelText);
            labelText.alignment = TextAlignmentOptions.Left;

            Image arrowImage = arrow.AddComponent<Image>();
            arrowImage.sprite = resources.dropdown;

            Image backgroundImage = root.AddComponent<Image>();
            backgroundImage.sprite = resources.standard;
            backgroundImage.color = s_DefaultSelectableColor;
            backgroundImage.type = Image.Type.Sliced;

            TMP_Dropdown dropdown = root.AddComponent<TMP_Dropdown>();
            dropdown.targetGraphic = backgroundImage;
            SetDefaultColorTransitionValues(dropdown);
            dropdown.template = template.GetComponent<RectTransform>();
            dropdown.captionText = labelText;
            dropdown.itemText = itemLabelText;

            // Setting default Item list.
            itemLabelText.text = "Option A";
            dropdown.options.Add(new TMP_Dropdown.OptionData {text = "Option A" });
            dropdown.options.Add(new TMP_Dropdown.OptionData {text = "Option B" });
            dropdown.options.Add(new TMP_Dropdown.OptionData {text = "Option C" });
            dropdown.RefreshShownValue();

            // Set up RectTransforms.

            RectTransform labelRT = label.GetComponent<RectTransform>();
            labelRT.anchorMin = Vector2.zero;
            labelRT.anchorMax = Vector2.one;
            labelRT.offsetMin = new Vector2(10, 6);
            labelRT.offsetMax = new Vector2(-25, -7);

            RectTransform arrowRT = arrow.GetComponent<RectTransform>();
            arrowRT.anchorMin = new Vector2(1, 0.5f);
            arrowRT.anchorMax = new Vector2(1, 0.5f);
            arrowRT.sizeDelta = new Vector2(20, 20);
            arrowRT.anchoredPosition = new Vector2(-15, 0);

            RectTransform templateRT = template.GetComponent<RectTransform>();
            templateRT.anchorMin = new Vector2(0, 0);
            templateRT.anchorMax = new Vector2(1, 0);
            templateRT.pivot = new Vector2(0.5f, 1);
            templateRT.anchoredPosition = new Vector2(0, 2);
            templateRT.sizeDelta = new Vector2(0, 150);

            RectTransform viewportRT = viewport.GetComponent<RectTransform>();
            viewportRT.anchorMin = new Vector2(0, 0);
            viewportRT.anchorMax = new Vector2(1, 1);
            viewportRT.sizeDelta = new Vector2(-18, 0);
            viewportRT.pivot = new Vector2(0, 1);

            RectTransform contentRT = content.GetComponent<RectTransform>();
            contentRT.anchorMin = new Vector2(0f, 1);
            contentRT.anchorMax = new Vector2(1f, 1);
            contentRT.pivot = new Vector2(0.5f, 1);
            contentRT.anchoredPosition = new Vector2(0, 0);
            contentRT.sizeDelta = new Vector2(0, 28);

            RectTransform itemRT = item.GetComponent<RectTransform>();
            itemRT.anchorMin = new Vector2(0, 0.5f);
            itemRT.anchorMax = new Vector2(1, 0.5f);
            itemRT.sizeDelta = new Vector2(0, 20);

            RectTransform itemBackgroundRT = itemBackground.GetComponent<RectTransform>();
            itemBackgroundRT.anchorMin = Vector2.zero;
            itemBackgroundRT.anchorMax = Vector2.one;
            itemBackgroundRT.sizeDelta = Vector2.zero;

            RectTransform itemCheckmarkRT = itemCheckmark.GetComponent<RectTransform>();
            itemCheckmarkRT.anchorMin = new Vector2(0, 0.5f);
            itemCheckmarkRT.anchorMax = new Vector2(0, 0.5f);
            itemCheckmarkRT.sizeDelta = new Vector2(20, 20);
            itemCheckmarkRT.anchoredPosition = new Vector2(10, 0);

            RectTransform itemLabelRT = itemLabel.GetComponent<RectTransform>();
            itemLabelRT.anchorMin = Vector2.zero;
            itemLabelRT.anchorMax = Vector2.one;
            itemLabelRT.offsetMin = new Vector2(20, 1);
            itemLabelRT.offsetMax = new Vector2(-10, -2);

            template.SetActive(false);

            return root;
        }
    }
}

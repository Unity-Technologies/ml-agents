using System;
using UnityEditor;
using UnityEngine;

#if UNITY_2019_1_OR_NEWER
using UnityEngine.UIElements;
#else
using UnityEngine.Experimental.UIElements;
using UnityEngine.Experimental.UIElements.StyleEnums;
#endif

namespace UnityEditor.Collaboration
{
    internal class StatusView : VisualElement
    {
        Image m_Image;
        Label m_Message;
        Button m_Button;
        Action m_Callback;

        public Texture icon
        {
            get { return m_Image.image; }
            set
            {
                m_Image.image = value;
                m_Image.visible = value != null;
                // Until "display: hidden" is added, this is the only way to hide an element
                m_Image.style.height = value != null ? 150 : 0;
            }
        }

        public string message
        {
            get { return m_Message.text; }
            set
            {
                m_Message.text = value;
                m_Message.visible = value != null;
            }
        }

        public string buttonText
        {
            get { return m_Button.text; }
            set
            {
                m_Button.text = value;
                UpdateButton();
            }
        }

        public Action callback
        {
            get { return m_Callback; }
            set
            {
                m_Callback = value;
                UpdateButton();
            }
        }

        public StatusView()
        {
            name = "StatusView";

            this.StretchToParentSize();

            m_Image = new Image() { name = "StatusIcon", visible = false, style = { height = 0f }};
            m_Message = new Label() { name = "StatusMessage", visible = false};
            m_Button = new Button(InternalCallaback) { name = "StatusButton", visible = false};

            Add(m_Image);
            Add(m_Message);
            Add(m_Button);
        }

        private void UpdateButton()
        {
            m_Button.visible = m_Button.text != null && m_Callback != null;
        }

        private void InternalCallaback()
        {
            m_Callback();
        }
    }
}

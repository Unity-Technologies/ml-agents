using System;
using UnityEditor;
using UnityEditor.Collaboration;
using UnityEngine;

#if UNITY_2019_1_OR_NEWER
using UnityEngine.UIElements;
#else
using UnityEngine.Experimental.UIElements;
#endif

namespace UnityEditor.Collaboration
{
    internal class BuildStatusButton : Button
    {
        private readonly string iconPrefix = "Icons/Collab.Build";
        private readonly string iconSuffix = ".png";
        Label labelElement = new Label();
        Image iconElement = new Image() {name = "BuildIcon"};

        public BuildStatusButton(Action clickEvent) : base(clickEvent)
        {
            iconElement.image = EditorGUIUtility.Load(iconPrefix + iconSuffix) as Texture;
            labelElement.text = "Build Now";
            Add(iconElement);
            Add(labelElement);
        }

        public BuildStatusButton(Action clickEvent, BuildState state, int failures) : base(clickEvent)
        {
            switch (state)
            {
                case BuildState.InProgress:
                    iconElement.image = EditorGUIUtility.Load(iconPrefix + iconSuffix) as Texture;
                    labelElement.text = "In progress";
                    break;

                case BuildState.Failed:
                    iconElement.image = EditorGUIUtility.Load(iconPrefix + "Failed" + iconSuffix) as Texture;
                    labelElement.text = failures + ((failures == 1) ? " failure" : " failures");
                    break;

                case BuildState.Success:
                    iconElement.image = EditorGUIUtility.Load(iconPrefix + "Succeeded" + iconSuffix) as Texture;
                    labelElement.text = "success";
                    break;
            }

            Add(iconElement);
            Add(labelElement);
        }
    }
}

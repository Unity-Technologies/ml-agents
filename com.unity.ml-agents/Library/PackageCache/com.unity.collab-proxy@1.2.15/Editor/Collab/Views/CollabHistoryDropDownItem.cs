using System;
using System.IO;
using System.Linq;
using UnityEngine;

#if UNITY_2019_1_OR_NEWER
using UnityEngine.UIElements;
#else
using UnityEngine.Experimental.UIElements;
#endif


namespace UnityEditor.Collaboration
{
    internal class CollabHistoryDropDownItem : VisualElement
    {
        public CollabHistoryDropDownItem(string path, string action)
        {
            var fileName = Path.GetFileName(path);
            var isFolder = Path.GetFileNameWithoutExtension(path).Equals(fileName);
            var fileIcon = GetIconElement(action, fileName, isFolder);
            var metaContainer = new VisualElement();
            var fileNameLabel = new Label
            {
                name = "FileName",
                text = fileName
            };
            var filePathLabel = new Label
            {
                name = "FilePath",
                text = path
            };
            metaContainer.Add(fileNameLabel);
            metaContainer.Add(filePathLabel);
            Add(fileIcon);
            Add(metaContainer);
        }

        private Image GetIconElement(string action, string fileName, bool isFolder)
        {
            var prefix = isFolder ? "Folder" : "File";
            var actionName = action.First().ToString().ToUpper() + action.Substring(1);
            // Use the same icon for renamed and moved files
            actionName = actionName.Equals("Renamed") ? "Moved" : actionName;
            var iconElement = new Image
            {
                name = "FileIcon",
                image = EditorGUIUtility.LoadIcon("Icons/Collab." + prefix + actionName + ".png")
            };
            return iconElement;
        }
    }
}

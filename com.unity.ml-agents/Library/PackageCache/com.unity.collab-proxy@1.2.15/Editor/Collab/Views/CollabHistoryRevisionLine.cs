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
    internal class CollabHistoryRevisionLine : VisualElement
    {
        public CollabHistoryRevisionLine(int number)
        {
            AddNumber(number);
            AddLine("topLine");
            AddLine("bottomLine");
            AddIndicator();
        }

        public CollabHistoryRevisionLine(DateTime date, bool isFullDateObtained)
        {
            AddLine(isFullDateObtained ? "obtainedDateLine" : "absentDateLine");
            AddHeader(GetFormattedHeader(date));
            AddToClassList("revisionLineHeader");
        }

        private void AddHeader(string content)
        {
            Add(new Label
            {
                text = content
            });
        }

        private void AddIndicator()
        {
            Add(new VisualElement
            {
                name = "RevisionIndicator"
            });
        }

        private void AddLine(string className = null)
        {
            var line = new VisualElement
            {
                name = "RevisionLine"
            };
            if (!String.IsNullOrEmpty(className))
            {
                line.AddToClassList(className);
            }
            Add(line);
        }

        private void AddNumber(int number)
        {
            Add(new Label
            {
                text = number.ToString(),
                name = "RevisionIndex"
            });
        }

        private string GetFormattedHeader(DateTime date)
        {
            string result = "Commits on " + date.ToString("MMM d");
            switch (date.Day)
            {
                case 1:
                case 21:
                case 31:
                    result += "st";
                    break;
                case 2:
                case 22:
                    result += "nd";
                    break;
                case 3:
                case 23:
                    result += "rd";
                    break;
                default:
                    result += "th";
                    break;
            }
            return result;
        }
    }
}

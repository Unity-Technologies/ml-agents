using System;

namespace UnityEditor.PostProcessing
{
    public class PostProcessingModelEditorAttribute : Attribute
    {
        public readonly Type type;
        public readonly bool alwaysEnabled;

        public PostProcessingModelEditorAttribute(Type type, bool alwaysEnabled = false)
        {
            this.type = type;
            this.alwaysEnabled = alwaysEnabled;
        }
    }
}

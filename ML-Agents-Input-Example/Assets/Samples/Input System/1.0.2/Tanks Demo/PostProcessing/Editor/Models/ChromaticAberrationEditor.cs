#if UNITY_EDITOR
using UnityEngine.PostProcessing;

namespace UnityEditor.PostProcessing
{
    [PostProcessingModelEditor(typeof(ChromaticAberrationModel))]
    public class ChromaticaAberrationModelEditor : DefaultPostFxModelEditor
    {
    }
}
#endif

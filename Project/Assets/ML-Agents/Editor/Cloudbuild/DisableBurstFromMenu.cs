#if UNITY_CLOUDBUILD
using UnityEditor;

namespace MLAgents
{
    public class DisableBurstFromMenu
    {
        public static void DisableBurstCompilation()
        {
            EditorApplication.ExecuteMenuItem("Jobs/Burst/Enable Compilation");
        }
    }
}
#endif

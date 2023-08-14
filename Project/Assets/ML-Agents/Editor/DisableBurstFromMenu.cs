#if UNITY_CLOUD_BUILD
using UnityEditor;

// TODO do we still need this?

public class DisableBurstFromMenu
{
    /// This method is needed to disable Burst compilation on windows for our cloudbuild tests.
    /// Barracuda 0.4.0-preview depends on a version of Burst (1.1.1) which does not allow
    /// users to disable burst compilation on a per platform basis.  The burst version 1.3.0-preview-1
    /// allows for cross compilation, but is not released yet.
    ///
    /// We will be able to remove this when
    /// 1. Barracuda updates burst 1.3.0-preview-1 or
    /// 2. We update our edior version for our tests to 2019.1+
    public static void DisableBurstCompilation()
    {
        EditorApplication.ExecuteMenuItem("Jobs/Burst/Enable Compilation");
    }
}
#endif

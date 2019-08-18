#if UNITY_CLOUD_BUILD

namespace MLAgents
{
    public static class Builder
    {
        public static void PreExport()
        {
            BuilderUtils.SwitchAllLearningBrainToControlMode();
        }
    }
}

#endif

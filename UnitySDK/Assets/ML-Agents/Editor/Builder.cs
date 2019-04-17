#if UNITY_CLOUD_BUILD

namespace MLAgents
{

	public class Builder
	{
		public static void PreExport()
		{
			BuilderUtils.SwitchAllLearningBrainToControlMode();
		}
	}
}

#endif

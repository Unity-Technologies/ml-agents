namespace Unity.MLAgents.Actuators
{
    public interface IHeuristic
    {
        void Heuristic(in ActionBuffers actionBuffersOut);
    }
}

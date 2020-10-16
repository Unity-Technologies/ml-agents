using System;

namespace Unity.MLAgents
{
    internal class RecursionChecker : IDisposable
    {
        private bool m_IsRunning;
        private string m_MethodName;

        public RecursionChecker(string methodName)
        {
            m_MethodName = methodName;
        }

        public IDisposable Start()
        {
            if (m_IsRunning)
            {
                throw new UnityAgentsException(
                    $"{m_MethodName} called recursively. " +
                    "This might happen if you call EnvironmentStep() or EndEpisode() from custom " +
                    "code such as CollectObservations() or OnActionReceived()."
                );
            }
            m_IsRunning = true;
            return this;
        }

        public void Dispose()
        {
            m_IsRunning = false;
        }
    }
}

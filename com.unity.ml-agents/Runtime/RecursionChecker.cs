using System;

namespace Unity.MLAgents
{
    internal class RecursionChecker : IDisposable
    {
        private bool m_IsRunning;

        public IDisposable Start()
        {
            if (m_IsRunning)
            {
                throw new UnityAgentsException("Don't do this.");
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

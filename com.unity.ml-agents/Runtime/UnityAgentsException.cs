using System;

namespace MLAgents
{
    /// <summary>
    /// Contains exceptions specific to ML-Agents.
    /// </summary>
    [Serializable]
    public class UnityAgentsException : Exception
    {
        /// <summary>
        /// When a UnityAgentsException is called, the timeScale is set to 0.
        /// The simulation will end since no steps will be taken.
        /// </summary>
        /// <param name="message">The exception message</param>
        public UnityAgentsException(string message) : base(message)
        {
        }

        /// <summary>
        /// A constructor is needed for serialization when an exception propagates
        /// from a remoting server to the client.
        /// </summary>
        /// <param name="info">Data for serializing/de-serializing</param>
        /// <param name="context">Describes the source and destination of the serialized stream</param>
        protected UnityAgentsException(
            System.Runtime.Serialization.SerializationInfo info,
            System.Runtime.Serialization.StreamingContext context)
        {
        }
    }
}

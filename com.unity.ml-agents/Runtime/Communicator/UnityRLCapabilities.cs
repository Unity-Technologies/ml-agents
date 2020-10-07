using UnityEngine;

namespace Unity.MLAgents
{
    internal class UnityRLCapabilities
    {
        public bool BaseRLCapabilities;
        public bool ConcatenatedPngObservations;
        public bool CompressedChannelMapping;

        /// <summary>
        /// A class holding the capabilities flags for Reinforcement Learning across C# and the Trainer codebase.  This
        /// struct will be used to inform users if and when they are using C# / Trainer features that are mismatched.
        /// </summary>
        public UnityRLCapabilities(bool baseRlCapabilities = true, bool concatenatedPngObservations = true, bool compressedChannelMapping = true)
        {
            BaseRLCapabilities = baseRlCapabilities;
            ConcatenatedPngObservations = concatenatedPngObservations;
            CompressedChannelMapping = compressedChannelMapping;
        }

        /// <summary>
        /// Will print a warning to the console if Python does not support base capabilities and will
        /// return <value>true</value> if the warning was printed.
        /// </summary>
        /// <returns></returns>
        public bool WarnOnPythonMissingBaseRLCapabilities()
        {
            if (BaseRLCapabilities)
            {
                return false;
            }
            Debug.LogWarning("Unity has connected to a Training process that does not support" +
                             "Base Reinforcement Learning Capabilities.  Please make sure you have the" +
                             " latest training codebase installed for this version of the ML-Agents package.");
            return true;
        }

    }
}

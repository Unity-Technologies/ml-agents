using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents.CommunicatorObjects;

namespace MLAgents
{
    public struct CommunicatorParameters
    {
        public int port;
    }


    /**
     * This is the interface used to generate coordinators. 
     * This does not need to be modified nor implemented to create a 
     * Unity environment.
     */
    public interface Communicator
    {
        // TODO : Comments
        UnityInput Initialize(UnityOutput academyParameters,
                                    out UnityInput unityInput);

        UnityInput Exchange(UnityOutput unityOutput);

        /// <summary>
        /// Close the communicator.
        /// </summary>
        void Close();

    }
}

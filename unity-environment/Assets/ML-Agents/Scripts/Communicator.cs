using System.Collections;
using System.Collections.Generic;
using UnityEngine;


namespace MLAgents.Communicator
{

    /**
     * This is the interface used to generate coordinators. 
     * This does not need to be modified nor implemented to create a 
     * Unity environment.
     */
    public interface Communicator
    {
        /// <summary>
        ///  Implement this method to Send the academy communicators
        /// </summary>
        /// <returns>The python Parameters and the first input</returns>
        PythonParameters Initialize(AcademyParameters academyParameters,
                                    out UnityRLInput unityImput);

        /// <summary>
        ///  Sends the UnityOutput via communication
        /// </summary>
        /// <returns>The new inputs.</returns>
        /// <param name="unityOutput">The Unity output.</param>
        UnityRLInput SendOuput(UnityRLOutput unityOutput);

        void Close();

        ///// <summary>
        ///// Gets the last command received by the communicator.
        ///// </summary>
        ///// <returns>The command.</returns>
        //Command GetCommand();

        ///// <summary>
        ///// Gets the latest environment parameters.
        ///// </summary>
        ///// <returns>The environment parameters.</returns>
        //EnvironmentParameters GetEnvironmentParameters();

    }
}

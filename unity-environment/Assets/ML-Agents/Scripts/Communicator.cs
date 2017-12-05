using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/** \brief AcademyParameters is a structure containing basic information about the 
 * training environment. */
/** The AcademyParameters will be sent via socket at the start of the Environment.
 * This structure does not need to be modified.
 */
public struct AcademyParameters
{
    public string AcademyName;
    /**< \brief The name of the Academy. If the communicator is External, 
     * it will be the name of the Academy GameObject */
    public string apiNumber;
    /**< \brief The API number for the communicator. */
    public string logPath;
	/**< \brief The location of the logfile*/
	public Dictionary<string, float> resetParameters;
    /**< \brief The default reset parameters are sent via socket*/
    public List<string> brainNames;
    /**< \brief A list of the all the brains names sent via socket*/
    public List<BrainParameters> brainParameters;
    /**< \brief  A list of the External brains parameters sent via socket*/
    public List<string> externalBrainNames;
    /**< \brief  A list of the External brains names sent via socket*/
}

public enum ExternalCommand
{
    STEP,
    RESET,
    QUIT
}

/**
 * This is the interface used to generate coordinators. 
 * This does not need to be modified nor implemented to create a 
 * Unity environment.
 */
public interface Communicator
{

    /// Implement this method to allow brains to subscribe to the 
    /// decisions made outside of Unity
    void SubscribeBrain(Brain brain);

    /// First contact between Communicator and external process
    bool CommunicatorHandShake();

    /// Implement this method to initialize the communicator
    void InitializeCommunicator();

    /// Implement this method to receive actions from outside of Unity and 
    /// update the actions of the brains that subscribe
    void UpdateActions();

    /// Implement this method to return the ExternalCommand that 
    /// was given outside of Unity
    ExternalCommand GetCommand();

    /// Implement this method to return the new dictionary of resetParameters 
    /// that was given outside of Unity
    Dictionary<string, float> GetResetParameters();


}

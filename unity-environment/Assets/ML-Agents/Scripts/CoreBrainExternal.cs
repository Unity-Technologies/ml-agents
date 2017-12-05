using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// CoreBrain which decides actions via communication with an external system such as Python.
public class CoreBrainExternal : ScriptableObject, CoreBrain
{

    public Brain brain;
    /**< Reference to the brain that uses this CoreBrainExternal */

    ExternalCommunicator coord;

    /// Creates the reference to the brain
    public void SetBrain(Brain b)
    {
        brain = b;
    }

    /// Generates the communicator for the Academy if none was present and
    ///  subscribe to ExternalCommunicator if it was present.
    public void InitializeCoreBrain()
    {
        if (brain.gameObject.transform.parent.gameObject.GetComponent<Academy>().communicator == null)
        {
            coord = null;
            throw new UnityAgentsException(string.Format("The brain {0} was set to" +
                " External mode" +
                " but Unity was unable to read the" +
                " arguments passed at launch.", brain.gameObject.name));
        }
        else if (brain.gameObject.transform.parent.gameObject.GetComponent<Academy>().communicator is ExternalCommunicator)
        {
            coord = (ExternalCommunicator)brain.gameObject.transform.parent.gameObject.GetComponent<Academy>().communicator;
            coord.SubscribeBrain(brain);
        }

    }

    /// Uses the communicator to retrieve the actions, memories and values and
    ///  sends them to the agents
    public void DecideAction()
    {
        if (coord != null)
        {
            brain.SendActions(coord.GetDecidedAction(brain.gameObject.name));
            brain.SendMemories(coord.GetMemories(brain.gameObject.name));
            brain.SendValues(coord.GetValues(brain.gameObject.name));
        }
    }

    /// Uses the communicator to send the states, observations, rewards and
    ///  dones outside of Unity
    public void SendState()
    {
        if (coord != null)
        {
            coord.giveBrainInfo(brain);
        }
    }

    /// Nothing needs to appear in the inspector 
    public void OnInspector()
    {

    }
}

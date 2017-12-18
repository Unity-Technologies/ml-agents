using System.Collections;
using System.Collections.Generic;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

/// CoreBrain which decides actions using developer-provided Decision.cs script.
public class CoreBrainHeuristic : ScriptableObject, CoreBrain
{
    [SerializeField]
    private bool broadcast = true;

    /**< Reference to the brain that uses this CoreBrainHeuristic */
    public Brain brain;

    ExternalCommunicator coord;

    /**< Reference to the Decision component used to decide the actions */
    public Decision decision;

    /// Create the reference to the brain
    public void SetBrain(Brain b)
    {
        brain = b;
    }

    /// Create the reference to decision
    public void InitializeCoreBrain()
    {
        decision = brain.gameObject.GetComponent<Decision>();

        if ((brain.gameObject.transform.parent.gameObject.GetComponent<Academy>().communicator == null)
            || (!broadcast))
        {
            coord = null;
        }
        else if (brain.gameObject.transform.parent.gameObject.GetComponent<Academy>().communicator is ExternalCommunicator)
        {
            coord = (ExternalCommunicator)brain.gameObject.transform.parent.gameObject.GetComponent<Academy>().communicator;
            coord.SubscribeBrain(brain);
        }
    }

    /// Uses the Decision Component to decide that action to take
    public void DecideAction()
    {
        if (decision == null)
        {
            throw new UnityAgentsException("The Brain is set to Heuristic, but no decision script attached to it");
        }

        var actions = new Dictionary<int, float[]>();
        var new_memories = new Dictionary<int, float[]>();
        Dictionary<int, List<float>> states = brain.CollectStates();
        Dictionary<int, List<Camera>> observations = brain.CollectObservations();
        Dictionary<int, float> rewards = brain.CollectRewards();
        Dictionary<int, bool> dones = brain.CollectDones();
        Dictionary<int, float[]> old_memories = brain.CollectMemories();

        foreach (KeyValuePair<int, Agent> idAgent in brain.agents)
        {
            actions.Add(idAgent.Key, decision.Decide(
                states[idAgent.Key],
                observations[idAgent.Key],
                rewards[idAgent.Key],
                dones[idAgent.Key],
                old_memories[idAgent.Key]));
        }
        foreach (KeyValuePair<int, Agent> idAgent in brain.agents)
        {
            new_memories.Add(idAgent.Key, decision.MakeMemory(
                states[idAgent.Key],
                observations[idAgent.Key],
                rewards[idAgent.Key],
                dones[idAgent.Key],
                old_memories[idAgent.Key]));
        }
        brain.SendActions(actions);
        brain.SendMemories(new_memories);
    }

    /// Nothing needs to be implemented, the states are collected in DecideAction
    public void SendState()
    {
        if (coord!=null)
        {
            coord.giveBrainInfo(brain);
        }
    }

    /// Displays an error if no decision component is attached to the brain
    public void OnInspector()
    {
#if UNITY_EDITOR
        EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);
        broadcast = EditorGUILayout.Toggle("Broadcast", broadcast);
        if (brain.gameObject.GetComponent<Decision>() == null)
        {
            EditorGUILayout.HelpBox("You need to add a 'Decision' component to this gameObject", MessageType.Error);
        }
#endif
    }

}

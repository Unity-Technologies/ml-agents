using System.Collections;
using System.Collections.Generic;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif


/// CoreBrain which decides actions using Player input.
public class CoreBrainPlayer : ScriptableObject, CoreBrain
{
    [SerializeField]
    private bool broadcast = true;

    [System.Serializable]
    private struct DiscretePlayerAction
    {
        public KeyCode key;
        public int value;
    }

    [System.Serializable]
    private struct ContinuousPlayerAction
    {
        public KeyCode key;
        public int index;
        public float value;
    }
        
    ExternalCommunicator coord;

    [SerializeField]
    /// Contains the mapping from input to continuous actions
    private ContinuousPlayerAction[] continuousPlayerActions;
    [SerializeField]
    /// Contains the mapping from input to discrete actions
    private DiscretePlayerAction[] discretePlayerActions;
    [SerializeField]
    private int defaultAction = 0;

    /// Reference to the brain that uses this CoreBrainPlayer
    public Brain brain;

    /// Create the reference to the brain
    public void SetBrain(Brain b)
    {
        brain = b;
    }

    /// Nothing to implement
    public void InitializeCoreBrain()
    {
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

    /// Uses the continuous inputs or dicrete inputs of the player to 
    /// decide action
    public void DecideAction()
    {
        if (brain.brainParameters.actionSpaceType == StateType.continuous)
        {
            float[] action = new float[brain.brainParameters.actionSize];
            foreach (ContinuousPlayerAction cha in continuousPlayerActions)
            {
                if (Input.GetKey(cha.key))
                {
                    action[cha.index] = cha.value;
                }
            }
            Dictionary<int, float[]> actions = new Dictionary<int, float[]>();
            foreach (KeyValuePair<int, Agent> idAgent in brain.agents)
            {
                actions.Add(idAgent.Key, action);
            }
            brain.SendActions(actions);
        }
        else
        {
            float[] action = new float[1] { defaultAction };
            foreach (DiscretePlayerAction dha in discretePlayerActions)
            {
                if (Input.GetKey(dha.key))
                {
                    action[0] = (float)dha.value;
                    break;
                }
            }
            Dictionary<int, float[]> actions = new Dictionary<int, float[]>();
            foreach (KeyValuePair<int, Agent> idAgent in brain.agents)
            {
                actions.Add(idAgent.Key, action);
            }
            brain.SendActions(actions);
        }
    }

    /// Nothing to implement, the Player does not use the state to make 
    /// decisions
    public void SendState()
    {
        if (coord != null)
        {
            coord.giveBrainInfo(brain);
        }
        else
        {
            //The states are collected in order to debug the CollectStates method.
            brain.CollectStates();
        }
    }

    /// Displays continuous or discrete input mapping in the inspector
    public void OnInspector()
    {
#if UNITY_EDITOR
        EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);
        broadcast = EditorGUILayout.Toggle("Broadcast", broadcast);
        SerializedObject serializedBrain = new SerializedObject(this);
        if (brain.brainParameters.actionSpaceType == StateType.continuous)
        {
            GUILayout.Label("Edit the continuous inputs for you actions", EditorStyles.boldLabel);
            SerializedProperty chas = serializedBrain.FindProperty("continuousPlayerActions");
            serializedBrain.Update();
            EditorGUILayout.PropertyField(chas, true);
            serializedBrain.ApplyModifiedProperties();
            if (continuousPlayerActions == null)
            {
                continuousPlayerActions = new ContinuousPlayerAction[0];
            }
            foreach (ContinuousPlayerAction cha in continuousPlayerActions)
            {
                if (cha.index >= brain.brainParameters.actionSize)
                {
                    EditorGUILayout.HelpBox(string.Format("Key {0} is assigned to index {1} but the action size is only of size {2}"
                        , cha.key.ToString(), cha.index.ToString(), brain.brainParameters.actionSize.ToString()), MessageType.Error);
                }
            }

        }
        else
        {
            GUILayout.Label("Edit the discrete inputs for you actions", EditorStyles.boldLabel);
            defaultAction = EditorGUILayout.IntField("Default Action", defaultAction);
            SerializedProperty dhas = serializedBrain.FindProperty("discretePlayerActions");
            serializedBrain.Update();
            EditorGUILayout.PropertyField(dhas, true);
            serializedBrain.ApplyModifiedProperties();
        }
#endif
    }
}

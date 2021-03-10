using System;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace Unity.MLAgents
{
#if UNITY_EDITOR
    [InitializeOnLoad]
#endif
    internal class MLAgentsManager
    {
        static MLAgentsManager()
        {
#if UNITY_EDITOR
            InitializeInEditor();
#else
            InitializeInPlayer();
#endif
        }

        private static void InitializeInEditor()
        {

        }

        private static void InitializeInPlayer()
        {

        }
    }

#if UNITY_EDITOR
    [Serializable]
    internal struct SerializedState
    {
        public MLAgentsSettings settings;
    }
#endif
}

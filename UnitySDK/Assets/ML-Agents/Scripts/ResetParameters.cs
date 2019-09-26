using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Serialization;

namespace MLAgents
{
    [Serializable]
    public class ResetParameters : Dictionary<string, float>, ISerializationCallbackReceiver
    {
        [Serializable]
        public struct ResetParameter
        {
            public string key;
            public float value;
        }

        [FormerlySerializedAs("resetParameters")]
        [SerializeField] private List<ResetParameter> m_ResetParameters = new List<ResetParameter>();

        public void OnBeforeSerialize()
        {
            m_ResetParameters.Clear();

            foreach (var pair in this)
            {
                m_ResetParameters.Add(new ResetParameter { key = pair.Key, value = pair.Value });
            }
        }

        public void OnAfterDeserialize()
        {
            Clear();


            for (var i = 0; i < m_ResetParameters.Count; i++)
            {
                if (ContainsKey(m_ResetParameters[i].key))
                {
                    Debug.LogError("The ResetParameters contains the same key twice");
                }
                else
                {
                    Add(m_ResetParameters[i].key, m_ResetParameters[i].value);
                }
            }
        }
    }
}

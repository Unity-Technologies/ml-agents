using System;
using System.Collections.Generic;
using UnityEngine;


[System.Serializable]
public class ResetParameters : Dictionary<string, float>, ISerializationCallbackReceiver
{

    [System.Serializable]
    public struct ResetParameter
    {
        public string key;
        public float value;
    }
    [SerializeField]
    private List<ResetParameter> resetParameters = new List<ResetParameter>();

    public void OnBeforeSerialize()
    {
        resetParameters.Clear();

        foreach (KeyValuePair<string, float> pair in this)
        {
            ResetParameter rp = new ResetParameter();
            rp.key = pair.Key;

            rp.value = pair.Value;
            resetParameters.Add(rp);
        }

    }

    public void OnAfterDeserialize()
    {
        this.Clear();



        for (int i = 0; i < resetParameters.Count; i++)
        {
            if (this.ContainsKey(resetParameters[i].key))
            {
                Debug.LogError("The ResetParameters contains the same key twice");
            }
            else
            {
                this.Add(resetParameters[i].key, resetParameters[i].value);
            }
        }
    }
}
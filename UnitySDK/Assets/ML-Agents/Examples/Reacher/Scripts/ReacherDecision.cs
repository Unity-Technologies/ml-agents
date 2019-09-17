using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class ReacherDecision : Decision {

    public override float[] Decide (List<float> state, List<Texture2D> observation, float reward, bool done, List<float> memory)
    {
        float[] action = new float[4];
        for (int i = 0; i < 4; i++) {
            action[i] = Random.Range(-1f, 1f);
        }
        return action;

    }

    public override List<float> MakeMemory (List<float> state, List<Texture2D> observation, float reward, bool done, List<float> memory)
    {
        return new List<float>();
        
    }
}

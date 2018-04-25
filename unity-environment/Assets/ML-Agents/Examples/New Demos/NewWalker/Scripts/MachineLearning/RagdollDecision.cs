using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RagdollDecision : MonoBehaviour, Decision
{
    public int decisionNumber;
    public float[] staticDecisions;

    public bool useStaticDecision;
   

    public float[] Decide(List<float> state, List<Texture2D> observation, float reward, bool done,List<float> memory)
    {
        if (useStaticDecision)
            return staticDecisions;
        float[] dec = new float[decisionNumber];
        for (int i = 0; i < decisionNumber; i++)
            dec[i] = Random.Range(-1f, 1f);

        return dec;
    }

   

    public List<float> MakeMemory(List<float> state, List<Texture2D> observation, float reward, bool done, List<float> memory)
    {
        return new List<float>();
    }


    // Use this for initialization
    void Start()
    {
       
    }

    // Update is called once per frame
    void Update()
    {

    }
}

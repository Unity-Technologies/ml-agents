using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Ball3DDecision : MonoBehaviour, Decision
{
    public float rotationSpeed = 2f;
    public float[] Decide(List<float> state, List<Texture2D> observation, float reward, bool done, List<float> memory)
    {
        if (gameObject.GetComponent<Brain>().brainParameters.vectorActionSpaceType == StateType.continuous)
        {
            List<float> act = new List<float>();
            //state[5] is the velocity of the ball in the x orientation. We use this number to control the Platform's z axis rotation speed, 
            //so that the Platform is tilted in the x orientation correspondingly. 
            act.Add(state[5] * rotationSpeed);
            //state[7] is the velocity of the ball in the z orientation. We use this number to control the Platform's x axis rotation speed,  
            //so that the Platform is tilted in the z orientation correspondingly. 
            act.Add(-state[7] * rotationSpeed);
            return act.ToArray();
        }
        //If the vector action space type is discrete, then we don't do anything. 	
        else
        {
            return new float[1]{ 1f };
        }
    }

    public List<float> MakeMemory(List<float> state, List<Texture2D> observation, float reward, bool done, List<float> memory)
    {
        return new List<float>();
    }
}

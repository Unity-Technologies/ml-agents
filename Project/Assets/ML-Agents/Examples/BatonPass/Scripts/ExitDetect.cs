using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;

public class ExitDetect : MonoBehaviour
{
    public BatonPassArea Area;
    public float RewardAtExit;

    void OnCollisionEnter(Collision other)
    {
        if (other.gameObject.CompareTag("agent"))
        {
            Area.AddReward(Academy.Instance.EnvironmentParameters.GetWithDefault("exit_reward", RewardAtExit));
            other.gameObject.GetComponent<BatonPassAgent>().Kill();
        }
    }
}

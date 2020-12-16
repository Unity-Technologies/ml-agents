using UnityEngine;

[CreateAssetMenu(fileName = "GoalConfig", menuName = "ML-Agents/GridWorld/Goal Config", order = 0)]
public class GoalConfig : ScriptableObject
{
    public GameObject prefab;
    public float rewardScore;
}

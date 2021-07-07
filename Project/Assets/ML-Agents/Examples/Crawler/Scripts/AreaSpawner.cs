using UnityEngine;
using Unity.MLAgents;


public class AreaSpawner : MonoBehaviour
{
    public GameObject PrefabObject;
    public int NumAreas = 1;
    public Vector3 Spacing = new Vector3(10, 0, 0);


    public void Start()
    {
        NumAreas = (int)Academy.Instance.EnvironmentParameters.GetWithDefault("num_area", NumAreas);
        Vector3 pos = new Vector3(0, 0, 0);
        for (int i = 0; i < NumAreas; i++)
        {
            Instantiate(PrefabObject, pos, Quaternion.identity);
            pos += Spacing;
        }
    }
}

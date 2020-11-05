using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Match3TileSelector : MonoBehaviour
{
    private Dictionary<int, MeshRenderer> tileDict = new Dictionary<int, MeshRenderer>();
    // Start is called before the first frame update
    void Awake()
    {
        for (int i = 0; i < tileTypes.Length; i++)
        {
            tileDict.Add(i, tileTypes[i].GetComponent<MeshRenderer>());
        }
        SetActiveTile(0, 0);
    }

    // Update is called once per frame
    void Update()
    {

    }

    public GameObject[] tileTypes = new GameObject[0];
    public Material[] materialTypes = new Material[0];
    public void SetActiveTile(int typeIndex, int matIndex)
    {
        for (int i = 0; i < tileTypes.Length; i++)
        {
            if (i == typeIndex)
            {
                tileTypes[i].SetActive(true);
                //                print($"Activated {typeIndex} with Mat {matIndex}");
                tileDict[i].sharedMaterial = materialTypes[matIndex];
            }
            else
            {
                tileTypes[i].SetActive(false);
            }
        }
    }


    //    public void SetActiveTile(int t)
    //    {
    //        for (int i = 0; i < tileTypes.Length; i++)
    //        {
    //            if (i == t)
    //            {
    //                tileTypes[i].SetActive(true);
    //                //                print($"Activated {t}");
    //                //                print(tileTypes[i].gameObject.name);
    //            }
    //            else
    //            {
    //                tileTypes[i].SetActive(false);
    //            }
    //        }
    //    }
}

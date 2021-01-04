using System.Collections.Generic;
using UnityEngine;

public class Match3TileSelector : MonoBehaviour
{
    public GameObject emptyTile;
    public GameObject[] tileTypes = new GameObject[0];
    public Material[] materialTypes = new Material[0];

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

    public void AllTilesOff()
    {
        foreach (var item in tileTypes)
        {
            item.SetActive(false);
        }
    }

    public void SetActiveTile(int typeIndex, int matIndex)
    {
        if (matIndex == -1)
        {
            AllTilesOff();
            emptyTile.SetActive(true);
        }
        else
        {
            emptyTile.SetActive(false);
            for (int i = 0; i < tileTypes.Length; i++)
            {
                if (i == typeIndex)
                {
                    tileTypes[i].SetActive(true);
                    tileDict[i].sharedMaterial = materialTypes[matIndex];
                }
                else
                {
                    tileTypes[i].SetActive(false);
                }
            }
        }
    }
}

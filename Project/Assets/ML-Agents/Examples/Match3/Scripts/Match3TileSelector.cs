using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Match3TileSelector : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        SetActiveTile(0);
    }

    // Update is called once per frame
    void Update()
    {

    }

    public GameObject[] tileTypes = new GameObject[0];
    public GameObject[] specialTileTypes = new GameObject[0];
    public void SetActiveTile(int t)
    {
        for (int i = 0; i < tileTypes.Length; i++)
        {
            if (i == t)
            {
                tileTypes[i].SetActive(true);
                //                print(tileTypes[i].gameObject.name);
            }
            else
            {
                tileTypes[i].SetActive(false);
            }
        }
    }
}

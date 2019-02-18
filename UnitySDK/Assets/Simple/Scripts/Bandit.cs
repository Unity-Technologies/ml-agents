using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Bandit : MonoBehaviour {

    public Material Gold;
    public Material Silver;
    public Material Bronze;

    private MeshRenderer _mesh;
    private Material _reset;
    
	// Use this for initialization
	void Start () {
        _mesh = GetComponent<MeshRenderer>();
        _reset = _mesh.material;
	}
	
    public int PullArm(int arm)
    {
        var reward = 0;
        switch (arm)
        {
            case 1:
                _mesh.material = Gold;
                reward = 3;
                break;
            case 2:
                _mesh.material = Bronze;
                reward = 1;
                break;
            case 3:
                _mesh.material = Bronze;
                reward = 1;
                break;
            case 4:
                _mesh.material = Silver;
                reward = 2;
                break;
        }

        return reward;
    }

    public void Reset()
    {
        _mesh.material = _reset;
    }
}

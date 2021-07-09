//using System.Collections;
//using System.Collections.Generic;
//using UnityEngine;
//
//public class ImpactFlash : MonoBehaviour
//{
//    // Start is called before the first frame update
//    void Start()
//    {
//
//    }
//
//    // Update is called once per frame
//    void Update()
//    {
//
//    }
//
//    IEnumerator Flash()
//    {
//        WaitForFixedUpdate wait = new WaitForFixedUpdate();
//        if (bodyMesh)
//        {
//            bodyMesh.material.color = damageColor;
//        }
//        float timer = 0;
//        while (timer < damageFlashDuration)
//        {
//            timer += Time.fixedDeltaTime;
//            yield return wait;
//        }
//        if (bodyMesh)
//        {
//            bodyMesh.material.color = startingColor;
//        }
//    }
//}

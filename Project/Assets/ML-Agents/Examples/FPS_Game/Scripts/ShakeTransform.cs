//using System.Collections;
//using System.Collections.Generic;
//using UnityEngine;
//
//public class ShakeTransform : MonoBehaviour
//{
//    [Header("TIMING")] public float duration;
//    public float amount = .1f;
//
////    //if not
////    public Transform rootTransform;
//    private Vector3 startPos;
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
//
//    IEnumerator HandleShooting()
//    {
//        WaitForFixedUpdate wait = new WaitForFixedUpdate();
//        var timer = 0;
//        startPos = transform.localPosition;
//        while (timer < duration)
//        {
//            var pos = startPos + (Random.insideUnitSphere * amount);
//            transform.localPosition = pos;
//            yield return wait;
//        }
//    }
//}

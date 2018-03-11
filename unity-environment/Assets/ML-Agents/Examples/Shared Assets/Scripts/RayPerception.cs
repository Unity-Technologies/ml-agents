using System.Collections.Generic;
using UnityEngine;

public class RayPerception : MonoBehaviour
{

    List<float> perceptionVector = new List<float>();

    public List<float> Percieve(float rayDistance,
                         float[] rayAngles, string[] detectableObjects,
                          float startOffset, float endOffset)
    {
        perceptionVector.Clear();
        foreach (float angle in rayAngles)
        {
            float noise = 0f;
            float noisyAngle = angle + Random.Range(-noise, noise);
            Vector3 position = transform.TransformDirection(
                GiveCatersian(rayDistance, noisyAngle));
            position.y = endOffset;
            Debug.DrawRay(transform.position + new Vector3(0f, startOffset, 0f),
                          position, Color.black, 0.01f, true);
            RaycastHit hit;
            float[] subList = new float[detectableObjects.Length + 2];
            if (Physics.SphereCast(transform.position +
                                   new Vector3(0f, startOffset, 0f), 0.5f,
                                   position, out hit, rayDistance))
            {
                for (int i = 0; i < detectableObjects.Length; i++)
                {
                    if (hit.collider.gameObject.CompareTag(detectableObjects[i]))
                    {
                        subList[i] = 1;
                        subList[detectableObjects.Length + 1] = hit.distance / rayDistance;
                        break;
                    }
                }
            }
            else
            {
                subList[detectableObjects.Length] = 1f;
            }
            perceptionVector.AddRange(subList);
        }
        return perceptionVector;
    }

    public Vector3 GiveCatersian(float radius, float angle)
    {
        float x = radius * Mathf.Cos(DegreeToRadian(angle));
        float z = radius * Mathf.Sin(DegreeToRadian(angle));
        return new Vector3(x, 1f, z);
    }

    public float DegreeToRadian(float degree)
    {
        return degree * Mathf.PI / 180f;
    }
}

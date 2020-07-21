using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WalkGroup : MonoBehaviour
{
    [Range(0, 15)]
    public float walkingSpeed = 15; //The walking speed to try and achieve
    public float m_maxWalkingSpeed = 15; //The max walking speed
}

using UnityEngine;

namespace MujocoUnity
{
    [System.Serializable]
    public class MujocoJoint
    {
        public Joint Joint;
        public string Name;
        public string JointName;
        public Vector2 CtrlRange;
        public bool? CtrlLimited;
        public float? Gear;
    }
}
using UnityEngine;

namespace Cinemachine.Examples
{

    [AddComponentMenu("")] // Don't display in add component menu
    public class MixingCameraBlend : MonoBehaviour
    {
        public enum AxisEnum { X, Z, XZ };

        public Transform followTarget;
        public float initialBottomWeight = 20f;
        public AxisEnum axisToTrack;

        private CinemachineMixingCamera vcam;

        void Start()
        {
            if (followTarget)
            {
                vcam = GetComponent<CinemachineMixingCamera>();
                vcam.m_Weight0 = initialBottomWeight;
            }
        }

        void Update()
        {
            if (followTarget)
            {
                switch (axisToTrack)
                {
                    case (AxisEnum.X):
                        vcam.m_Weight1 = Mathf.Abs(followTarget.transform.position.x);
                        break;
                    case (AxisEnum.Z):
                        vcam.m_Weight1 = Mathf.Abs(followTarget.transform.position.z);
                        break;
                    case (AxisEnum.XZ):
                        vcam.m_Weight1 =
                            Mathf.Abs(Mathf.Abs(followTarget.transform.position.x) +
                                      Mathf.Abs(followTarget.transform.position.z));
                        break;
                }
            }
        }
    }

}

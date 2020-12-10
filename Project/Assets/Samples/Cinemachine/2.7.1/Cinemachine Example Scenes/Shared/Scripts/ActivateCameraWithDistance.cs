using UnityEngine;

namespace Cinemachine.Examples
{

    [AddComponentMenu("")] // Don't display in add component menu
    public class ActivateCameraWithDistance : MonoBehaviour
    {
        public GameObject objectToCheck;
        public float distanceToObject = 15f;
        public CinemachineVirtualCameraBase initialActiveCam;
        public CinemachineVirtualCameraBase switchCameraTo;

        CinemachineBrain brain;

        void Start()
        {
            brain = Camera.main.GetComponent<CinemachineBrain>();
            SwitchCam(initialActiveCam);
        }

        // Update is called once per frame
        void Update()
        {
            if (objectToCheck && switchCameraTo)
            {
                if (Vector3.Distance(transform.position, objectToCheck.transform.position) < distanceToObject)
                {
                    SwitchCam(switchCameraTo);
                }
                else
                {
                    SwitchCam(initialActiveCam);
                }
            }
        }

        public void SwitchCam(CinemachineVirtualCameraBase vcam)
        {
            if (brain == null || vcam == null)
                return;
            if (brain.ActiveVirtualCamera != (ICinemachineCamera)vcam)
                vcam.MoveToTopOfPrioritySubqueue();
        }
    }

}

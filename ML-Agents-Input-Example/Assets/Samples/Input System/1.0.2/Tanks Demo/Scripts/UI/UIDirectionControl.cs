using UnityEngine;
using UnityEngine.Serialization;

namespace Complete
{
    public class UIDirectionControl : MonoBehaviour
    {
        // This class is used to make sure world space UI
        // elements such as the health bar face the correct direction.

        [FormerlySerializedAs("m_UseRelativeRotation")]
        public bool useRelativeRotation = true;       // Use relative rotation should be used for this gameobject?

        Quaternion m_RelativeRotation;          // The local rotatation at the start of the scene.

        void Start()
        {
            m_RelativeRotation = transform.parent.localRotation;
        }

        void Update()
        {
            if (useRelativeRotation)
                transform.rotation = m_RelativeRotation;
        }
    }
}

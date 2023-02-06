using System;
using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using UnityEngine;


[Serializable]
public struct CJControlSettings
{
    public string name;
    public Vector3 target;
    public Range3D range;
    public Vector3 originalPosition;
    public Quaternion originalRotation;

}

[Serializable]
public class Range3D
{
    public Range xRange;
    public Range yRange;
    public Range zRange;

    public Range3D(float xLow, float xHigh, float yLow, float yHigh, float zLow, float zHigh)
    {
        xRange = new Range(xLow, xHigh);
        yRange = new Range(xLow, xHigh);
        zRange = new Range(xLow, xHigh);
    }
}

[Serializable]
public class Range
{
    public float low;
    public float high;

    public Range(float low, float high)
    {
        this.low = low;
        this.high = high;
    }

    public float Scale(float input)
    {
        var t = (1.0f + input) / 2.0f;
        return Mathf.Round(Mathf.Lerp(low, high, t));

        // if (input >= 0f)
        // {
        //     return Mathf.RoundToInt(Mathf.Lerp(0f, m_High, input));
        // }
        // else
        // {
        //     return Mathf.RoundToInt(Mathf.Lerp(m_Low, 0f, 1.0f + input));
        // }
    }

    public float InverseScale(float input)
    {
        var t = Mathf.InverseLerp(low, high, input);
        return 2.0f * t - 1.0f;

        // if (input >= 0f)
        // {
        //     return Mathf.InverseLerp(0f, m_High, input);
        // }
        // else
        // {
        //     return Mathf.InverseLerp(m_Low, 0f, input) - 1.0f;
        // }
    }
}

public class ConfigurableJointController : MonoBehaviour
{
    public float spring = 1000;
    public float damping = 250;
    public float forceLimit = 500;
    public bool kinematicRoot;
    public float totalMass;

    [SerializeField]
    public CJControlSettings[] cjControlSettings;

    [Header("Body Parts")]
    public Transform hips;
    public Transform spine;
    public Transform head;
    public Transform thighL;
    public Transform shinL;
    public Transform footL;
    public Transform thighR;
    public Transform shinR;
    public Transform footR;
    public Transform armL;
    public Transform forearmL;
    public Transform handL;
    public Transform armR;
    public Transform forearmR;
    public Transform handR;

    ConfigurableJoint[] m_ConfigurableJointChain;
    Rigidbody[] m_RigidbodyChain;
    Agent m_Agent;
    bool m_IsAgentNull;
    Vector3 m_RootOriginalPosition;
    Quaternion m_RootOriginalRotation;

    // Start is called before the first frame update
    void Start()
    {
        m_Agent = GetComponent<Agent>();
        m_IsAgentNull = m_Agent == null;
        PopulateParameters();
        m_ConfigurableJointChain[0].GetComponent<Rigidbody>().isKinematic = kinematicRoot;
        m_RootOriginalPosition = hips.position;
        m_RootOriginalRotation = hips.rotation;
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        if (m_IsAgentNull)
        {
            SetCJointTargets();
        }
    }

#if UNITY_EDITOR
    void OnValidate()
    {
        PopulateParameters();
    }
#endif
    void PopulateParameters(bool force = false)
    {
        if (m_ConfigurableJointChain == null || m_ConfigurableJointChain.Length == 0 || force)
        {
            m_ConfigurableJointChain = new[]
            {
                hips.GetComponent<ConfigurableJoint>(),
                spine.GetComponent<ConfigurableJoint>(),
                head.GetComponent<ConfigurableJoint>(),
                thighL.GetComponent<ConfigurableJoint>(),
                shinL.GetComponent<ConfigurableJoint>(),
                footL.GetComponent<ConfigurableJoint>(),
                thighR.GetComponent<ConfigurableJoint>(),
                shinR.GetComponent<ConfigurableJoint>(),
                footR.GetComponent<ConfigurableJoint>(),
                armL.GetComponent<ConfigurableJoint>(),
                forearmL.GetComponent<ConfigurableJoint>(),
                handL.GetComponent<ConfigurableJoint>(),
                armR.GetComponent<ConfigurableJoint>(),
                forearmR.GetComponent<ConfigurableJoint>(),
                handR.GetComponent<ConfigurableJoint>(),
            };
        }

        if (m_RigidbodyChain == null || m_RigidbodyChain.Length == 0 || force)
        {
            m_RigidbodyChain = new[]
            {
                hips.GetComponent<Rigidbody>(),
                spine.GetComponent<Rigidbody>(),
                head.GetComponent<Rigidbody>(),
                thighL.GetComponent<Rigidbody>(),
                shinL.GetComponent<Rigidbody>(),
                footL.GetComponent<Rigidbody>(),
                thighR.GetComponent<Rigidbody>(),
                shinR.GetComponent<Rigidbody>(),
                footR.GetComponent<Rigidbody>(),
                armL.GetComponent<Rigidbody>(),
                forearmL.GetComponent<Rigidbody>(),
                handL.GetComponent<Rigidbody>(),
                armR.GetComponent<Rigidbody>(),
                forearmR.GetComponent<Rigidbody>(),
                handR.GetComponent<Rigidbody>(),
            };
        }
        if (cjControlSettings == null || cjControlSettings.Length == 0 || force)
        {
            cjControlSettings = new CJControlSettings[m_ConfigurableJointChain.Length - 1];
            for (int i = 0; i < cjControlSettings.Length; i++)
            {
                cjControlSettings[i].name = m_ConfigurableJointChain[i + 1].name;
                cjControlSettings[i].target = m_ConfigurableJointChain[i + 1].targetRotation.eulerAngles;
                cjControlSettings[i].range = new Range3D(
                    m_ConfigurableJointChain[i + 1].lowAngularXLimit.limit,
                    m_ConfigurableJointChain[i + 1].highAngularXLimit.limit,
                    -m_ConfigurableJointChain[i + 1].angularYLimit.limit,
                    m_ConfigurableJointChain[i + 1].angularYLimit.limit,
                    -m_ConfigurableJointChain[i + 1].angularZLimit.limit,
                    m_ConfigurableJointChain[i + 1].angularZLimit.limit
                );
                cjControlSettings[i].originalPosition = m_ConfigurableJointChain[i].transform.localPosition;
                cjControlSettings[i].originalRotation = m_ConfigurableJointChain[i].transform.localRotation;
            }
        }
        if (totalMass == 0)
        {
            foreach (var rb in m_RigidbodyChain)
            {
                totalMass += rb.mass;
            }
        }
    }
    public void SetCJointTargets(ActionSegment<float> angles)
    {
        for (int i = 0; i < cjControlSettings.Length; i++)
        {
            var subindex = 3 * i;
            var xAngle = cjControlSettings[i].range.xRange.Scale(angles[subindex]);
            var yAngle = cjControlSettings[i].range.yRange.Scale(angles[++subindex]);
            var zAngle = cjControlSettings[i].range.zRange.Scale(angles[++subindex]);
            var targetRotation = new Vector3(xAngle, yAngle, zAngle);
            SetCJointTarget(m_ConfigurableJointChain[i + 1], targetRotation);
        }
    }

    public void SetCJointTargets()
    {
        for (int i = 0; i < cjControlSettings.Length; i++)
        {
            SetCJointTarget(m_ConfigurableJointChain[i + 1], cjControlSettings[i].target);
        }
    }

    public void SetCJointTarget(ConfigurableJoint joint, Vector3 target)
    {
        joint.targetRotation = Quaternion.Euler(target);
    }

    public void SetCJointPositionsAndRotations()
    {
        for (int i = 0; i < cjControlSettings.Length; i++)
        {
            m_RigidbodyChain[i + 1].transform.position = cjControlSettings[i].originalPosition;
            m_RigidbodyChain[i + 1].transform.rotation = cjControlSettings[i].originalRotation;
        }
    }

    public IEnumerator ResetCJointTargetsAndPositions()
    {
        m_ConfigurableJointChain[0].GetComponent<Rigidbody>().isKinematic = true;
        ZeroCJointPhysics();
        ZeroCJointPhysicsSettings();
        // yield return new WaitForSeconds(1.0f / 50);
        SetCJointPositionsAndRotations();
        SetCJointTargets();
        SetCJointPhysicsSettings();
        yield return new WaitForSeconds(1.0f / 50);
        m_ConfigurableJointChain[0].GetComponent<Rigidbody>().isKinematic = kinematicRoot;
    }
    void SetCJointPhysicsSettings()
    {
        for (int i = 1; i < m_ConfigurableJointChain.Length; i++)
        {
            m_ConfigurableJointChain[i].rotationDriveMode = RotationDriveMode.Slerp;
            var drive = m_ConfigurableJointChain[i].slerpDrive;
            drive.positionSpring = spring;
            drive.positionDamper = damping;
            drive.maximumForce = forceLimit;
            m_ConfigurableJointChain[i].slerpDrive = drive;
        }

    }

    public void ZeroCJointPhysics()
    {
        foreach (var rb in m_RigidbodyChain)
        {
            rb.velocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
            rb.Sleep();
        }
    }

    public void ZeroCJointPhysicsSettings()
    {
        for (int i = 1; i < m_ConfigurableJointChain.Length; i++)
        {
            m_ConfigurableJointChain[i].rotationDriveMode = RotationDriveMode.Slerp;
            var drive = m_ConfigurableJointChain[i].slerpDrive;
            drive.positionSpring = 0f;
            drive.positionDamper = 0f;
            drive.maximumForce = forceLimit;
            m_ConfigurableJointChain[i].slerpDrive = drive;
        }
    }

}

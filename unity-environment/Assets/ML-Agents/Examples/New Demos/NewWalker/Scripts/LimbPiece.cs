using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LimbPiece : MonoBehaviour
{
    public ConfigurableJoint joint;
    Ragdoll owner;
    public new Rigidbody rigidbody;
    public bool touchingGround = false;
    // Use this for initialization
    void Awake()
    {
        joint = GetComponent<ConfigurableJoint>();
        owner = GetComponentInParent<Ragdoll>();
        rigidbody = GetComponent<Rigidbody>();
    }

    public void SetNormalizedTargetRotation(float x, float y, float z)
    {
        //rigidbody.AddRelativeTorque(x * 15f, y * 15f, z * 15f, ForceMode.Acceleration);
        //return;
        //x = Mathf.InverseLerp(-1f,1f,x);
        x = Mathf.Clamp(x, -1f, 1f);
        y = Mathf.Clamp(y, -1f, 1f);
        z = Mathf.Clamp(z, -1f, 1f);
        y = (y + 1f) * 0.5f;
        z = (z + 1f) * 0.5f;

        float xRot;

        if (x <= 0f)
        {
            x = x + 1f;
            xRot = Mathf.Lerp(joint.lowAngularXLimit.limit, 0f, x);
        }
        else
        {
            
            xRot = Mathf.Lerp(0f, joint.highAngularXLimit.limit, x);
        }
        float yRot = Mathf.Lerp( -joint.angularYLimit.limit, joint.angularYLimit.limit,y);
        float zRot = Mathf.Lerp(-joint.angularZLimit.limit, joint.angularZLimit.limit, z);

        joint.targetRotation = Quaternion.Euler(xRot, yRot, zRot);
    }

    public void AdjustNormalizedTargetRotation(float xI,float yI, float zI)
    {
        Vector3 ang =  joint.targetRotation.eulerAngles;
        float xCur = Mathf.InverseLerp(joint.lowAngularXLimit.limit, joint.highAngularXLimit.limit, ang.x);
        float yCur = Mathf.InverseLerp(-joint.angularYLimit.limit, joint.angularYLimit.limit, ang.y);
        float zCur = Mathf.InverseLerp(-joint.angularZLimit.limit, joint.angularZLimit.limit, ang.z);

        xCur += xI * 0.1f;
        yCur += yI * 0.1f;
        zCur += zI * 0.1f;
    
    }

    public Vector3 GetCurrentNormalizedTargetRotation()
    {
        Vector3 ang = joint.targetRotation.eulerAngles;
        float xCur = Mathf.InverseLerp(joint.lowAngularXLimit.limit, joint.highAngularXLimit.limit, ang.x);
        float yCur = Mathf.InverseLerp(-joint.angularYLimit.limit, joint.angularYLimit.limit, ang.y);
        float zCur = Mathf.InverseLerp(-joint.angularZLimit.limit, joint.angularZLimit.limit, ang.z);

        return new Vector3(xCur, yCur, zCur);
    }
    public Vector3 RelativePosFromPelvis
    {
        get
        {
            return transform.position - owner.pelvis.transform.position;
        }
    }

    public Vector3 LocalPosInPelvis
    {
        get
        {
            return owner.pelvis.transform.InverseTransformPoint(transform.position);
        }
    }
    public Vector3 Velocity
    {
        get
        {
            return rigidbody.velocity;
        }
    }

    public float Height
    {
        get
        {
            return transform.position.y;
        }
    }
    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.layer == LayerMask.NameToLayer("ground"))
        {
            touchingGround = true;
        }
    }

    private void OnCollisionExit(Collision collision)
    {
        if (collision.gameObject.layer == LayerMask.NameToLayer("ground"))
        {
            touchingGround = false;
        }
    }
}

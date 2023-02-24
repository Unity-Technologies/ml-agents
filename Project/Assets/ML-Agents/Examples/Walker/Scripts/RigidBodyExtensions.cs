using UnityEngine;

public static class RigidbodyExtensions
{
    public static Vector3 GetLocalVelocity(this Rigidbody body)
    {
        var parentTransform = body.transform.parent;
        return parentTransform.InverseTransformVector(body.velocity);
    }
}

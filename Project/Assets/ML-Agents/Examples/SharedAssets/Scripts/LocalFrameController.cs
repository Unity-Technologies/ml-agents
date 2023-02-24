using UnityEngine;

public class LocalFrameController : MonoBehaviour
{
    public Quaternion UpdateLocalFrame(Transform root, bool display)
    {
        var headingRotation = CalculateHeadingQuaternionInverse(root);
        transform.SetPositionAndRotation(root.position, headingRotation);
        UpdateDisplay(display);
        return headingRotation;
    }
    public void UpdateLocalFrame(Transform root, Transform target, bool display)
    {
        var heading = CalculateHeading(root, target);
        var lookRot = Quaternion.AngleAxis(heading, Vector3.up);
        transform.SetPositionAndRotation(root.position, lookRot);
        UpdateDisplay(display);
    }

    float CalculateHeading(Transform root, Transform target)
    {
        var direction = (target.position - root.position).normalized;
        var rotationDirection = Quaternion.FromToRotation(direction, Vector3.forward);
        var heading = -rotationDirection.eulerAngles.y;
        return heading;
    }

    void UpdateDisplay(bool display)
    {
        var meshRenderers = GetComponentsInChildren<MeshRenderer>();
        foreach (var mesh in meshRenderers)
        {
            mesh.enabled = display;
        }
    }

    Quaternion CalculateHeadingQuaternionInverse(Transform root)
    {
        var heading = CalculateHeading(root.rotation) * Mathf.Rad2Deg;
        var headingQuaternion = Quaternion.AngleAxis(heading, Vector3.up);
        return headingQuaternion;
    }

    float CalculateHeading(Quaternion rootRotation)
    {
        var rotatonDirection = rootRotation * Vector3.forward;
        return Mathf.Atan2(rotatonDirection.x, rotatonDirection.z);
    }
}

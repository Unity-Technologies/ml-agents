using UnityEngine;

public class PointAtAimTarget : MonoBehaviour
{
    [Tooltip("This object represents the aim target.  We always point toeards this")]
    public Transform AimTarget;

    void Update()
    {
        // Aim at the aim target
        if (AimTarget == null)
            return;
        var dir = AimTarget.position - transform.position;
        if (dir.sqrMagnitude > 0.01f)
            transform.rotation = Quaternion.LookRotation(dir);
    }
}

using UnityEngine;
 
/// <summary>
/// A Transform Extension that ignores scale for TransformPoint operations.
/// </summary>
public static class TransformExtensions
{
    
    /// <summary>
    /// Transform position from world space to local space unscaled.
    /// </summary>
    public static Vector3 TransformPointUnscaled(this Transform transform, Vector3 position)
    {
        var localToWorldMatrix = Matrix4x4.TRS(transform.position, transform.rotation, Vector3.one);
        return localToWorldMatrix.MultiplyPoint3x4(position);
    }
 
    /// <summary>
    /// Transform position from local space to world space unscaled.
    /// </summary>
    public static Vector3 InverseTransformPointUnscaled(this Transform transform, Vector3 position)
    {
        var worldToLocalMatrix = Matrix4x4.TRS(transform.position, transform.rotation, Vector3.one).inverse;
        return worldToLocalMatrix.MultiplyPoint3x4(position);
    }
}
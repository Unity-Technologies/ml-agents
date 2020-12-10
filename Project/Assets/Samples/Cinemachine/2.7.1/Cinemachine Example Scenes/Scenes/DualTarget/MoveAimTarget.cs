using UnityEngine;
using Cinemachine;

public class MoveAimTarget : MonoBehaviour
{
    public CinemachineBrain Brain;
    public RectTransform ReticleImage;

    [Tooltip("How far to raycast to place the aim target")]
    public float AimDistance;

    [Tooltip("Objects on these layers will be detected")]
    public LayerMask CollideAgainst;

    [TagField]
    [Tooltip("Obstacles with this tag will be ignored.  "
        + "It's a good idea to set this field to the player's tag")]
    public string IgnoreTag = string.Empty;

    /// <summary>The Vertical axis.  Value is -90..90. Controls the vertical orientation</summary>
    [Header("Axis Control")]
    [Tooltip("The Vertical axis.  Value is -90..90. Controls the vertical orientation")]
    [AxisStateProperty]
    public AxisState VerticalAxis;

    /// <summary>The Horizontal axis.  Value is -180..180.  Controls the horizontal orientation</summary>
    [Tooltip("The Horizontal axis.  Value is -180..180.  Controls the horizontal orientation")]
    [AxisStateProperty]
    public AxisState HorizontalAxis;

    private void OnValidate()
    {
        VerticalAxis.Validate();
        HorizontalAxis.Validate();
        AimDistance = Mathf.Max(1, AimDistance);
    }

    private void Reset()
    {
        AimDistance = 200;
        ReticleImage = null;
        CollideAgainst = 1;
        IgnoreTag = string.Empty;

        VerticalAxis = new AxisState(-70, 70, false, false, 10f, 0.1f, 0.1f, "Mouse Y", true);
        VerticalAxis.m_SpeedMode = AxisState.SpeedMode.InputValueGain;
        HorizontalAxis = new AxisState(-180, 180, true, false, 10f, 0.1f, 0.1f, "Mouse X", false);
        HorizontalAxis.m_SpeedMode = AxisState.SpeedMode.InputValueGain;
    }

    private void OnEnable()
    {
        CinemachineCore.CameraUpdatedEvent.RemoveListener(PlaceReticle);
        CinemachineCore.CameraUpdatedEvent.AddListener(PlaceReticle);
    }

    private void OnDisable()
    {
        CinemachineCore.CameraUpdatedEvent.RemoveListener(PlaceReticle);
    }

    private void Update()
    {
        if (Brain == null)
            return;

        HorizontalAxis.Update(Time.deltaTime);
        VerticalAxis.Update(Time.deltaTime);

        PlaceTarget();
    }

    private void PlaceTarget()
    {
        var rot = Quaternion.Euler(VerticalAxis.Value, HorizontalAxis.Value, 0);
        var camPos = Brain.CurrentCameraState.RawPosition;
        transform.position = GetProjectedAimTarget(camPos + rot * Vector3.forward, camPos);
    }

    private Vector3 GetProjectedAimTarget(Vector3 pos, Vector3 camPos)
    {
        var origin = pos;
        var fwd = (pos - camPos).normalized;
        pos += AimDistance * fwd;
        if (CollideAgainst != 0 && RaycastIgnoreTag(
            new Ray(origin, fwd),
            out RaycastHit hitInfo, AimDistance, CollideAgainst))
        {
            pos = hitInfo.point;
        }
        return pos;
    }

    private bool RaycastIgnoreTag(
        Ray ray, out RaycastHit hitInfo, float rayLength, int layerMask)
    {
        const float PrecisionSlush = 0.001f;
        float extraDistance = 0;
        while (Physics.Raycast(
            ray, out hitInfo, rayLength, layerMask,
            QueryTriggerInteraction.Ignore))
        {
            if (IgnoreTag.Length == 0 || !hitInfo.collider.CompareTag(IgnoreTag))
            {
                hitInfo.distance += extraDistance;
                return true;
            }

            // Ignore the hit.  Pull ray origin forward in front of obstacle
            Ray inverseRay = new Ray(ray.GetPoint(rayLength), -ray.direction);
            if (!hitInfo.collider.Raycast(inverseRay, out hitInfo, rayLength))
                break;
            float deltaExtraDistance = rayLength - (hitInfo.distance - PrecisionSlush);
            if (deltaExtraDistance < PrecisionSlush)
                break;
            extraDistance += deltaExtraDistance;
            rayLength = hitInfo.distance - PrecisionSlush;
            if (rayLength < PrecisionSlush)
                break;
            ray.origin = inverseRay.GetPoint(rayLength);
        }
        return false;
    }

    void PlaceReticle(CinemachineBrain brain)
    {
        if (brain == null || brain != Brain || ReticleImage == null || brain.OutputCamera == null)
            return;
        PlaceTarget(); // To eliminate judder
        CameraState state = brain.CurrentCameraState;
        var cam = brain.OutputCamera;
        var r = cam.WorldToScreenPoint(transform.position);
        var r2 = new Vector2(r.x - cam.pixelWidth * 0.5f, r.y - cam.pixelHeight * 0.5f);
        ReticleImage.anchoredPosition = r2;
    }
}

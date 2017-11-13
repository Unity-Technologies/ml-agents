using UnityEngine;
using UnityEngine.Events;

[System.Serializable]
public class UnityEventFloat : UnityEvent<float> { }

public class RobotArmController4Dof : MonoBehaviour {

    public float rotateSpeed = 180f;
    public float bendSpeed = 180f;

    public GameObject Hand;
    public GameObject HitCenter;

    public RotateScript[] Rotators;
    public BendScript[] Benders;

    public Vector2 RotationMinMax = new Vector2(0, 360);
    public Vector2 BendMinMax = new Vector2(-90, 360);

    public UnityEventFloat OnSegmentMove;

    public void Reset()
    {
        foreach (var rotator in Rotators)
        {
            //rotator.Reset();
            SetRotation(rotator, Random.Range(-1f, 1f));
            Rotate(rotator, false);
        }
        foreach (var bender in Benders)
        {
            //bender.Reset();
            SetBend(bender, Random.Range(-1f, 1f));
            Bend(bender, false);
        }
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        /*
#if UNITY_EDITOR
        SetRotation(0, Input.GetAxis("Horizontal"));
        SetRotation(1, Input.GetAxis("HorizontalRight"));
        SetBend(0, Input.GetAxis("Vertical"));
        SetBend(1, Input.GetAxis("VerticalRight"));
        SetBend(2, Input.GetAxis("TriggerRight"));
#endif
*/

        foreach (var rotator in Rotators)
        {
            Rotate(rotator);
        }

        foreach (var bender in Benders)
        {
            Bend(bender);
        }
    }


    public void Rotate(RotateScript rotator, bool UseRotationSpeeds = true)
    {
        var delta = rotator.DesiredRotation - rotator.CurrentRotation;

        if (UseRotationSpeeds)
        {
            var maxRot = rotateSpeed * Time.deltaTime;
            var minRot = -rotateSpeed * Time.deltaTime;
            if (delta < -180) delta += 360; // Go around the other way
            if (delta > 180) delta -= 360; // Go around the other way
            delta = Mathf.Clamp(delta, minRot, maxRot);
            if (OnSegmentMove != null) OnSegmentMove.Invoke(delta);
        }

        rotator.transform.Rotate(0, delta, 0);
        rotator.CurrentRotation += delta;
    }

    public void Bend(BendScript bender, bool UseRotationSpeeds = true)
    {
        var delta = bender.DesiredBend - bender.CurrentBend;

        if (UseRotationSpeeds)
        {
            var maxBen = bendSpeed * Time.deltaTime;
            var minBen = -bendSpeed * Time.deltaTime;
            delta = Mathf.Clamp(delta, minBen, maxBen);
            if (OnSegmentMove != null) OnSegmentMove.Invoke(delta);
        }

        bender.transform.Rotate(delta * bender.BendMask);
        bender.CurrentBend += delta;
    }

    public void SetRotation(RotateScript rotator, float position)
    {
        rotator.DesiredRotation = (position + 1.0f) * 180; // range: 0 to 2 * 180 = 0 to 360
    }

    public void SetRotation(int index, float position) // range: -1 to 1
    {
        SetRotation(Rotators[index], position);
    }

    public void SetBend(BendScript bender, float position)
    {
        var p = (position + 1.0f) * 0.5f; // range 0 to 1
        var bendRange = BendMinMax.y - BendMinMax.x;
        bender.DesiredBend = (p * bendRange) + BendMinMax.x;
    }

    public void SetBend(int index, float position) // range: -1 to 1
    {
        SetBend(Benders[index], position);
    }


}

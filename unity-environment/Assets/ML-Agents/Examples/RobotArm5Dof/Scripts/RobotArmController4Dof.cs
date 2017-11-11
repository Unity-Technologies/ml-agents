using UnityEngine;

public class RobotArmController4Dof : MonoBehaviour {

    public float rotateSpeed = 180f;
    public float bendSpeed = 180f;

    public bool UseRotationSpeeds = false;

    //public GameObject FirstRotator;
    public GameObject Hand;
    public GameObject HitCenter;

    public RotateScript[] Rotators;
    public BendScript[] Benders;

    public Vector2 RotationMinMax = new Vector2(0, 360);
    public Vector2 BendMinMax = new Vector2(-90, 360);

    public void Reset()
    {
        foreach (var rotator in Rotators) rotator.Reset();
        foreach (var bender in Benders) bender.Reset();
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        /*
        #if UNITY_EDITOR
        SetRotation(0, (Input.GetAxis("Horizontal") * 0.5f) + 0.5f);
        SetRotation(1, (Input.GetAxis("HorizontalRight") * 0.5f) + 0.5f);
        SetBend(0, (Input.GetAxis("Vertical") / 2) + 0.5f);
        SetBend(1, (Input.GetAxis("VerticalRight") / 2) + 0.5f);
        SetBend(2, (Input.GetAxis("TriggerRight") / 2) + 0.5f);
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


    public void Rotate(RotateScript rotator)
    {
        var delta = rotator.DesiredRotation - rotator.CurrentRotation;

        if (UseRotationSpeeds)
        {
            var maxRot = rotateSpeed * Time.deltaTime;
            var minRot = -rotateSpeed * Time.deltaTime;
            if (delta < -180) delta += 360; // Go around the other way
            if (delta > 180) delta -= 360; // Go around the other way
            delta = Mathf.Clamp(delta, minRot, maxRot);
        }

        rotator.transform.Rotate(0, delta, 0);
        rotator.CurrentRotation += delta;
    }

    public void Bend(BendScript bender)
    {
        var delta = bender.DesiredBend - bender.CurrentBend;

        if (UseRotationSpeeds)
        {
            var maxBen = bendSpeed * Time.deltaTime;
            var minBen = -bendSpeed * Time.deltaTime;
            delta = Mathf.Clamp(delta, minBen, maxBen);
        }

        bender.transform.Rotate(delta * bender.BendMask);
        bender.CurrentBend += delta;
    }

    public void SetRotation(int index, float position)
    {
        Rotators[index].DesiredRotation = position * 360f;
    }

    public void SetBend(int index, float position)
    {
        var bendRange = BendMinMax.y - BendMinMax.x;
        Benders[index].DesiredBend = (position * bendRange) + BendMinMax.x;
    }


}

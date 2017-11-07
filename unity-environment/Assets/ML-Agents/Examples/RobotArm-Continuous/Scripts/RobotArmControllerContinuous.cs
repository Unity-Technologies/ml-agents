using UnityEngine;

public class RobotArmControllerContinuous : MonoBehaviour {

    public float rotateSpeed = 180f;
    public float bendSpeed = 180f;

    public bool UseRotationSpeeds = false;

    public KeyCode one;
    public KeyCode two;
    public KeyCode three;
    public KeyCode four;
    public KeyCode five;
    public KeyCode six;
    public KeyCode seven;
    public KeyCode eight;

    public KeyCode oneBend;
    public KeyCode twoBend;
    public KeyCode threeBend;
    public KeyCode fourBend;
    public KeyCode fiveBend;
    public KeyCode sixBend;
    public KeyCode sevenBend;
    public KeyCode eightBend;
    public KeyCode nineBend;

    public GameObject FirstRotator;
    public GameObject FirstBender;

    public Vector2 RotationMinMax = new Vector2(-360, 360);
    public Vector2 BendMinMax = new Vector2(-90, 360);

    public float DesiredRotation;
    public float DesiredBend;

    // Update is called once per frame
    void FixedUpdate()
    {
        if (Input.GetKey(one)) SetRotation(0f);
        if (Input.GetKey(two)) SetRotation(0.125f);
        if (Input.GetKey(three)) SetRotation(0.25f);
        if (Input.GetKey(four)) SetRotation(0.375f);
        if (Input.GetKey(five)) SetRotation(0.5f);
        if (Input.GetKey(six)) SetRotation(0.625f);
        if (Input.GetKey(seven)) SetRotation(0.75f);
        if (Input.GetKey(eight)) SetRotation(0.875f);

        if (Input.GetKey(oneBend)) SetBend(0f);
        if (Input.GetKey(twoBend)) SetBend(0.125f);
        if (Input.GetKey(threeBend)) SetBend(0.25f);
        if (Input.GetKey(fourBend)) SetBend(0.375f);
        if (Input.GetKey(fiveBend)) SetBend(0.5f);
        if (Input.GetKey(sixBend)) SetBend(0.625f);
        if (Input.GetKey(sevenBend)) SetBend(0.75f);
        if (Input.GetKey(eightBend)) SetBend(0.875f);
        if (Input.GetKey(nineBend)) SetBend(1f);

        Rotate();
        Bend();
    }

    public void Rotate()
    {
        var rot = FirstRotator.transform.rotation.eulerAngles.y;
        var delta = DesiredRotation - rot;

        if (UseRotationSpeeds)
        {
            var maxRot = rotateSpeed * Time.deltaTime;
            var minRot = -rotateSpeed * Time.deltaTime;
            if (delta < -180) delta += 360; // Go around the other way
            if (delta > 180) delta -= 360; // Go around the other way
            delta = Mathf.Clamp(delta, minRot, maxRot);
        }

        FirstRotator.transform.Rotate(0, delta, 0);
    }

    public void Bend()
    {
        var ben = FirstBender.transform.rotation.eulerAngles.x;
        if (ben > 180) ben -= 360f; // Map to the -90 to 90 degree space

        var delta = DesiredBend - ben;

        if (UseRotationSpeeds)
        {
            var maxBen = bendSpeed * Time.deltaTime;
            var minBen = -bendSpeed * Time.deltaTime;
            delta = Mathf.Clamp(delta, minBen, maxBen);
        }

        FirstBender.transform.Rotate(delta, 0, 0);
    }

    public void SetRotation(float position)
    {
        DesiredRotation = position * 360f;
    }

    public void SetBend(float position)

    {   // map input to the -90 to 90 degree space
        var bendRange = BendMinMax.y - BendMinMax.x;
        DesiredBend = (position * bendRange) + BendMinMax.x;
    }
}

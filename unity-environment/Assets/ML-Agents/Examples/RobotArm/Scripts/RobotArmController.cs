using UnityEngine;

public class RobotArmController : MonoBehaviour {

    public float rotateSpeed = 180f;
    public float bendSpeed = 180f;

    public KeyCode leftKey;
    public KeyCode rightKey;
    public KeyCode upKey;
    public KeyCode downKey;

    public GameObject FirstRotator;
    public GameObject FirstBender;

    public Vector2 RotationMinMax = new Vector2(-360, 360);
    public Vector2 BendMinMax = new Vector2(-90, 360);

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKey(leftKey)) Left();
        if (Input.GetKey(rightKey)) Right();
        if (Input.GetKey(upKey)) Up();
        if (Input.GetKey(downKey)) Down();
    }

    public void Left()
    {
        FirstRotator.transform.Rotate(0, -rotateSpeed * Time.deltaTime, 0);
    }

    public void Right()
    {
        FirstRotator.transform.Rotate(0, rotateSpeed * Time.deltaTime, 0);
    }

    public void Up()
    {
        FirstBender.transform.Rotate(bendSpeed * Time.deltaTime, 0, 0);
        var parentAngle = FirstBender.transform.parent.rotation.eulerAngles;
        var angle = FirstBender.transform.rotation.eulerAngles.x;
        angle = Mathf.Clamp(angle, BendMinMax.x, BendMinMax.y);
        FirstBender.transform.rotation = Quaternion.Euler(angle, parentAngle.y, parentAngle.z);
    }

    public void Down()
    {
        FirstBender.transform.Rotate(-bendSpeed * Time.deltaTime, 0, 0);
        var parentAngle = FirstBender.transform.parent.rotation.eulerAngles;
        var angle = FirstBender.transform.rotation.eulerAngles.x;
        angle = Mathf.Clamp(angle, BendMinMax.x, BendMinMax.y);
        FirstBender.transform.rotation = Quaternion.Euler(angle, parentAngle.y, parentAngle.z);
    }
}

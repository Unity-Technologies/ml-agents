using UnityEngine;

public class ExampleWheelController : MonoBehaviour
{
    public float acceleration;
    public Renderer motionVectorRenderer; // Reference to the custom motion vector renderer

    Rigidbody m_Rigidbody;

    static class Uniforms
    {
        internal static readonly int _MotionAmount = Shader.PropertyToID("_MotionAmount");
    }

    void Start()
    {
        m_Rigidbody = GetComponent<Rigidbody>(); // Get reference to rigidbody
        m_Rigidbody.maxAngularVelocity = 100; // Set max velocity for rigidbody
    }

    void Update()
    {
        if (Input.GetKey (KeyCode.UpArrow)) // Rotate forward
            m_Rigidbody.AddRelativeTorque(new Vector3(-1 * acceleration, 0, 0), ForceMode.Acceleration); // Add forward torque to mesh
        else if (Input.GetKey (KeyCode.DownArrow)) // Rotate backward
            m_Rigidbody.AddRelativeTorque(new Vector3(1 * acceleration, 0, 0), ForceMode.Acceleration); // Add backward torque to mesh

        float m = -m_Rigidbody.angularVelocity.x / 100; // Calculate multiplier for motion vector texture

        if (motionVectorRenderer) // If the custom motion vector texture renderer exists
            motionVectorRenderer.material.SetFloat(Uniforms._MotionAmount, Mathf.Clamp(m, -0.25f, 0.25f)); // Set the multiplier on the renderer's material
    }
}

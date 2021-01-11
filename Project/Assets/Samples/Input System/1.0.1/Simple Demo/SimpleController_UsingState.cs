using UnityEngine.InputSystem;
using UnityEngine;

// Using state of gamepad device directly.
public class SimpleController_UsingState : MonoBehaviour
{
    public float moveSpeed;
    public float rotateSpeed;
    public GameObject projectile;

    private Vector2 m_Rotation;
    private bool m_Firing;
    private float m_FireCooldown;

    public void Update()
    {
        var gamepad = Gamepad.current;
        if (gamepad == null)
            return;

        var leftStick = gamepad.leftStick.ReadValue();
        var rightStick = gamepad.rightStick.ReadValue();

        Look(rightStick);
        Move(leftStick);

        if (gamepad.buttonSouth.wasPressedThisFrame)
        {
            m_Firing = true;
            m_FireCooldown = 0;
        }
        else if (gamepad.buttonSouth.wasReleasedThisFrame)
        {
            m_Firing = false;
        }

        if (m_Firing && m_FireCooldown < Time.time)
        {
            Fire();
            m_FireCooldown = Time.time + 0.1f;
        }
    }

    private void Move(Vector2 direction)
    {
        if (direction.sqrMagnitude < 0.01)
            return;
        var scaledMoveSpeed = moveSpeed * Time.deltaTime;
        // For simplicity's sake, we just keep movement in a single plane here. Rotate
        // direction according to world Y rotation of player.
        var move = Quaternion.Euler(0, transform.eulerAngles.y, 0) * new Vector3(direction.x, 0, direction.y);
        transform.position += move * scaledMoveSpeed;
    }

    private void Look(Vector2 rotate)
    {
        if (rotate.sqrMagnitude < 0.01)
            return;
        var scaledRotateSpeed = rotateSpeed * Time.deltaTime;
        m_Rotation.y += rotate.x * scaledRotateSpeed;
        m_Rotation.x = Mathf.Clamp(m_Rotation.x - rotate.y * scaledRotateSpeed, -89, 89);
        transform.localEulerAngles = m_Rotation;
    }

    private void Fire()
    {
        var transform = this.transform;
        var newProjectile = Instantiate(projectile);
        newProjectile.transform.position = transform.position + transform.forward * 0.6f;
        newProjectile.transform.rotation = transform.rotation;
        const int size = 1;
        newProjectile.transform.localScale *= size;
        newProjectile.GetComponent<Rigidbody>().mass = Mathf.Pow(size, 3);
        newProjectile.GetComponent<Rigidbody>().AddForce(transform.forward * 20f, ForceMode.Impulse);
        newProjectile.GetComponent<MeshRenderer>().material.color =
            new Color(Random.value, Random.value, Random.value, 1.0f);
    }
}

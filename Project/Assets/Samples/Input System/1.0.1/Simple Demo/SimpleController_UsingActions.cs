using System.Collections;
using UnityEngine.InputSystem;
using UnityEngine;
using UnityEngine.InputSystem.Interactions;

// Using simple actions with callbacks.
public class SimpleController_UsingActions : MonoBehaviour
{
    public float moveSpeed;
    public float rotateSpeed;
    public float burstSpeed;
    public GameObject projectile;

    public InputAction moveAction;
    public InputAction lookAction;
    public InputAction fireAction;

    private bool m_Charging;

    private Vector2 m_Rotation;

    public void Awake()
    {
        // We could use `fireAction.triggered` in Update() but that makes it more difficult to
        // implement the charging mechanism. So instead we use the `started`, `performed`, and
        // `canceled` callbacks to run the firing logic right from within the action.

        fireAction.performed +=
            ctx =>
        {
            if (ctx.interaction is SlowTapInteraction)
            {
                StartCoroutine(BurstFire((int)(ctx.duration * burstSpeed)));
            }
            else
            {
                Fire();
            }
            m_Charging = false;
        };
        fireAction.started +=
            ctx =>
        {
            if (ctx.interaction is SlowTapInteraction)
                m_Charging = true;
        };
        fireAction.canceled +=
            ctx =>
        {
            m_Charging = false;
        };
    }

    public void OnEnable()
    {
        moveAction.Enable();
        lookAction.Enable();
        fireAction.Enable();
    }

    public void OnDisable()
    {
        moveAction.Disable();
        lookAction.Disable();
        fireAction.Disable();
    }

    public void OnGUI()
    {
        if (m_Charging)
            GUI.Label(new Rect(100, 100, 200, 100), "Charging...");
    }

    public void Update()
    {
        var look = lookAction.ReadValue<Vector2>();
        var move = moveAction.ReadValue<Vector2>();

        // Update orientation first, then move. Otherwise move orientation will lag
        // behind by one frame.
        Look(look);
        Move(move);
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

    public Vector2 rotateVector;
    private void Look(Vector2 rotate)
    {
        rotateVector = rotate;
        if (rotate.sqrMagnitude < 0.01)
            return;
        var scaledRotateSpeed = rotateSpeed * Time.deltaTime;
        m_Rotation.y += rotate.x * scaledRotateSpeed;
        m_Rotation.x = Mathf.Clamp(m_Rotation.x - rotate.y * scaledRotateSpeed, -89, 89);
        transform.localEulerAngles = m_Rotation;
    }

    private IEnumerator BurstFire(int burstAmount)
    {
        for (var i = 0; i < burstAmount; ++i)
        {
            Fire();
            yield return new WaitForSeconds(0.1f);
        }
    }

    private void Fire()
    {
        var transform = this.transform;
        var newProjectile = Instantiate(projectile);
        newProjectile.transform.position = transform.position + transform.forward * 0.6f;
        newProjectile.transform.rotation = transform.rotation;
        var size = 1;
        newProjectile.transform.localScale *= size;
        newProjectile.GetComponent<Rigidbody>().mass = Mathf.Pow(size, 3);
        newProjectile.GetComponent<Rigidbody>().AddForce(transform.forward * 20f, ForceMode.Impulse);
        newProjectile.GetComponent<MeshRenderer>().material.color =
            new Color(Random.value, Random.value, Random.value, 1.0f);
    }
}

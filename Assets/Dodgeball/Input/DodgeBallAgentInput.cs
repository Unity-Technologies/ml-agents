using UnityEngine;
using UnityEngine.SceneManagement;

//[ExecuteAlways]
public class DodgeBallAgentInput : MonoBehaviour
{
    public bool DisableInput = false;
    private DodgeBallInputActions inputActions;
    private DodgeBallInputActions.PlayerActions actionMap;
    public Vector2 moveInput;
    public bool m_throwPressed;
    public bool m_dashPressed;
    public float rotateInput;
    void Awake()
    {
        inputActions = new DodgeBallInputActions();
        actionMap = inputActions.Player;
    }
    void OnEnable()
    {
        inputActions.Enable();
    }

    private void OnDisable()
    {
        inputActions.Disable();
    }

    public bool CheckIfInputSinceLastFrame(ref bool input)
    {
        if (input)
        {
            input = false;
            return true;
        }
        return false;
    }

    void Update()
    {
        if (inputActions.UI.Restart.triggered)
        {
            SceneManager.LoadScene(SceneManager.GetActiveScene().name);
        }
    }

    void FixedUpdate()
    {
        if (DisableInput)
        {
            return;
        }
        moveInput = actionMap.Walk.ReadValue<Vector2>();
        rotateInput = actionMap.Rotate.ReadValue<float>();
        if (actionMap.Throw.triggered)
        {
            m_throwPressed = true;
        }
        if (actionMap.Dash.triggered)
        {
            m_dashPressed = true;
        }
    }
}

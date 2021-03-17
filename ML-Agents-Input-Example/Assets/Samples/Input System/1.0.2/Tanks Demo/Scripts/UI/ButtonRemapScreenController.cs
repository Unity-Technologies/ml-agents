using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public class ButtonRemapScreenController : MonoBehaviour
{
    public Button okButton;
    public InputActionAsset tanksInputActions;
    InputActionMap m_PlayerActionMap;

    void Start()
    {
        m_PlayerActionMap = tanksInputActions.FindActionMap("Player");
        m_PlayerActionMap.Disable();
        okButton.onClick.AddListener(OkButtonClicked);
    }

    void OkButtonClicked()
    {
        m_PlayerActionMap.Enable();
        SceneManager.LoadScene("NewInput");
    }
}

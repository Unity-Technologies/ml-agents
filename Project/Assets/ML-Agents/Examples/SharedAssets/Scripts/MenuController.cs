// Connected to the Cube and includes a DontDestroyOnLoad()
// LoadScene() is called by the first  script and switches to the second.

using UnityEngine;
using UnityEngine.SceneManagement;

public class MenuController : MonoBehaviour
{
    private static bool created = false;

    void Awake()
    {
        if (!created)
        {
            DontDestroyOnLoad(this.gameObject);
            created = true;
            Debug.Log("Awake: " + this.gameObject);
        }
    }

    // public void LoadScene()
    // {
    //     if (SceneManager.GetActiveScene().name == "scene1")
    //     {
    //         SceneManager.LoadScene("scene2", LoadSceneMode.Single);
    //     }
    // }
    public void LoadScene(string sceneName)
    {
        // if (SceneManager.GetActiveScene().name == "scene1")
        // {
            SceneManager.LoadScene(sceneName, LoadSceneMode.Single);
        // }
    }
}
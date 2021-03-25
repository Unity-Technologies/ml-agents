using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MyTimeScaleSetting : MonoBehaviour
{
    // s_Instance is used to cache the instance found in the scene so we don't have to look it up every time.
    private static MyTimeScaleSetting s_Instance = null;


    // A static property that finds or creates an instance of the manager object and returns it.
    public static MyTimeScaleSetting instance
    {
        get
        {
            if (s_Instance == null)
            {
                // FindObjectOfType() returns the first AManager object in the scene.
                s_Instance = FindObjectOfType(typeof(MyTimeScaleSetting)) as MyTimeScaleSetting;
            }

            // If it is still null, create a new instance
            if (s_Instance == null)
            {
                var obj = new GameObject("MyTimeScaleSetting");
                s_Instance = obj.AddComponent<MyTimeScaleSetting>();
            }

            return s_Instance;
        }
    }


    // Ensure that the instance is destroyed when the game is stopped in the editor.
    void OnApplicationQuit()
    {
        s_Instance = null;
    }


    [SerializeField]
    float m_TimeScale = 1f;

    public float MyTimeScale
    {
        get { return m_TimeScale; }
        set
        {
            m_TimeScale = value;
            Time.timeScale = value;
        }
    }

    [SerializeField]
    float m_Greedy = 0f;
    public float GreedyEpislon
    {
        get { return m_Greedy; }
        set { m_Greedy = value; }
    }

    [SerializeField]
    bool m_Train = true;
    public bool IsTraining
    {
        get { return m_Train; }
        set { m_Train = value; }
    }

    // Start is called before the first frame update
    void Start()
    {
        DontDestroyOnLoad(this.gameObject);
        if (FindObjectsOfType<MyTimeScaleSetting>().Length > 1)
        {
            Destroy(this.gameObject);
        }
    }

    // Update is called once per frame
    void Update()
    {
        Time.timeScale = m_TimeScale;
    }
}

using UnityEngine;

public class SingletonTest
{
    public static float something;

    protected SingletonTest() { }

    private static SingletonTest _instance = null;

    public static SingletonTest Instance
    {
        get
        {
            return SingletonTest._instance == null ? new SingletonTest() : SingletonTest._instance;
        }
    }
}
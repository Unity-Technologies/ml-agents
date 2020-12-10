using UnityEngine;

public class TagChanger : MonoBehaviour
{
    public void PlayerTagChanger()
    {
        this.tag = "Player";
    }
    public void UntaggedTagChanger()
    {
        this.tag = "Untagged";
    }
}

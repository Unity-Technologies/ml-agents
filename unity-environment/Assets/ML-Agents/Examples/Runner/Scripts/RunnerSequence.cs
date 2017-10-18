using UnityEngine;

/**
 * This component represents a level sequence in the game
*/
[ExecuteInEditMode]
public class RunnerSequence : MonoBehaviour
{
    [Header("Parameters")]
    [SerializeField]
	// Size of this sequence, used to generate the level and know where the next sequence will be instanciated
	private int blockSize = 16;

    [SerializeField]
	// Snap position in editor mode for child object to homogenised levels
	private bool snapPosition = true;

    [SerializeField]
    // List of positions (x) where bonus needs to be instanciated
    public Vector3[] bonusPositions;

    protected void Update()
    {
        if (snapPosition && !Application.isPlaying)
        {
            foreach (Transform child in transform)
            {
                child.localPosition = new Vector3(Mathf.RoundToInt(child.localPosition.x * 2) / 2f, child.localPosition.y, child.localPosition.z);
            }
        }
    }

	#region Properties
	public int Size
	{
		get { return blockSize; }
	}
	#endregion

	#region Debug functions
	protected void OnDrawGizmosSelected()
    {
        Gizmos.color = new Color(1, 1, 1, 0.5f);
        Gizmos.DrawWireCube(transform.position + Vector3.right * Size / 2f, new Vector3(Size, 3, 3));

        foreach (var pos in bonusPositions)
        {
            Gizmos.color = Color.yellow;
            Gizmos.DrawWireCube(transform.position + new Vector3(pos.x, pos.y, pos.z), Vector3.one * 0.5f);
        }
    }
    #endregion
}

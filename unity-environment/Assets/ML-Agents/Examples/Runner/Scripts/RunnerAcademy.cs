using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

[System.Serializable]
public class AgentSetup
{
    public RunnerAgent prefab;
    public Brain brain;
    public Color lineColor = Color.blue;
}

[System.Serializable]
public class SequenceSetup
{
    public RunnerSequence prefab;

    // This value indicate the minimum episode of the simulation needed before this sequence can be learn
    public int curriculumLevel;
}

public class RunnerAcademy : Academy {

	[Header("Game specific")]
	[SerializeField]
	// Prefab to instantiate for each bonus
	private GameObject bonusPrefab;

    [SerializeField]
    // Prefab to instantiate for ground killzone
    private GameObject killZonePrefab;

	[SerializeField]
	private Transform scrollRoot;

	[SerializeField]
	private float scrollSpeed = 0.15f;

	[SerializeField]
	private List<AgentSetup> agentSetup;

	[SerializeField]
	private SequenceSetup[] sequenceSetup;

    [SerializeField]
    private RunnerSequence startSequence;

    [SerializeField]
    private RunnerSequence endSequence;

    [SerializeField]
    private bool useCurriculum;

    private List<RunnerAgent> Agents = new List<RunnerAgent>();

    public override void InitializeAcademy()
	{
		Physics.gravity = new Vector3(0, -18, 0);

        // Instantiate killZone, scale depending on the number of agents
        var zone = Instantiate(killZonePrefab, new Vector3(40, -1.75f, agentSetup.Count / 2f - 0.5f), Quaternion.identity);
        zone.transform.localScale = new Vector3(100, 0.5f, agentSetup.Count);

		for (int i = 0; i < agentSetup.Count; i++)
        {
            var agent = Instantiate(agentSetup[i].prefab, new Vector3(0, 1, i * 1), Quaternion.identity) as RunnerAgent;
            agent.GiveBrain(agentSetup[i].brain);
            Agents.Add(agent);
        }
    }

    /// Scroll the root transform of the level
    /// Check if all agents are done
    public override void AcademyStep()
	{
        scrollRoot.localPosition = new Vector3(scrollRoot.localPosition.x - scrollSpeed, 0, 0);

        if (Agents.All((a) => a.done))
            done = true;
	}

	public override void AcademyReset()
	{
		scrollRoot.localPosition = Vector3.zero;

		foreach (Transform child in scrollRoot)
		{
		    Destroy(child.gameObject);
		}

		for (int i = 0; i < agentSetup.Count; i++)
		{
            GenerateAgentLevel(i, agentSetup[i].lineColor);
		}
	}

    /// Generate the level by chosing sequence randomly depending of the max steps
    private void GenerateAgentLevel(int i, Color sequenceColor)
	{
        float posZ = i * 1;

        int posX = AddSequence(startSequence, new Vector3(0, 0, posZ), sequenceColor);

        var available = sequenceSetup.Where((b) => b.curriculumLevel <= episodeCount || !useCurriculum).ToList();
        while (posX + endSequence.Size < scrollSpeed * maxSteps)
        {
            posX += AddSequence(available[Random.Range(0, available.Count())].prefab, new Vector3(posX, 0, posZ), sequenceColor);
        }

        AddSequence(endSequence, new Vector3(posX, 0, posZ), sequenceColor);
	}

    private int AddSequence(RunnerSequence sequence, Vector3 pos, Color color)
    {
        var newSequence = Instantiate(sequence, pos, Quaternion.identity, scrollRoot) as RunnerSequence;

        var childMaterials = newSequence.GetComponentsInChildren<MeshRenderer>();
        foreach (var m in childMaterials)
        {
            m.material.color = color;
        }

        // Add collectable bonus to this sequence
		foreach (var b in newSequence.bonusPositions)
		{
            Instantiate(bonusPrefab, newSequence.transform.position + b, Quaternion.identity, newSequence.transform);
		}

        return sequence.Size;
    }
}

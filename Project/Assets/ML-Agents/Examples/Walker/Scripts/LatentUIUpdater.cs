using System.Collections;
using System.Collections.Generic;
using TMPro;
using Unity.MLAgents;
using UnityEngine;

public class LatentUIUpdater : MonoBehaviour
{
    public WalkerASEAgent agent;
    public TextMeshProUGUI text;

    private LatentRequestor m_LatentRequestor;

    // Start is called before the first frame update
    void Start()
    {
        m_LatentRequestor = agent.GetComponent<LatentRequestor>();
    }

    // Update is called once per frame
    void Update()
    {
        text.SetText("Latent: [" + string.Join(", ", m_LatentRequestor.Latents) + "]");
    }
}

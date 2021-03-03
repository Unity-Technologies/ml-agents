using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;

public class TestDecisionRequester : MonoBehaviour
{
    public enum SteppingMode
    {
        StepAllAgentsEvery5Frames = 0,
        Step2AgentsEachFrame = 1
    }

    public SteppingMode Mode = SteppingMode.StepAllAgentsEvery5Frames;
    private int StepCount = 0;
    private Agent[] m_Agents;
    private double[] m_RunningAverageMs = new double[2];
    private int[] m_RunningCount = new int[2];

    static double s_TicksToMilliseconds = 1e-4; // 100 ns per tick

    private void Awake()
    {
        m_Agents = FindObjectsOfType<Agent>();
        Academy.Instance.AutomaticSteppingEnabled = false;
    }

    private void FixedUpdate()
    {
        if (Mode == SteppingMode.StepAllAgentsEvery5Frames)
        {
            if ((StepCount % 5) == 0)
            {
                for (var i = 0; i < m_Agents.Length; i++)
                {
                    m_Agents[i].RequestDecision();
                }
            }
        }
        else
        {
            for (var i = 0; i < m_Agents.Length; i++)
            {
                if (i % 5 == StepCount % 5)
                {
                    m_Agents[i].RequestDecision();
                }
            }

        }

        var startTicks = DateTime.Now.Ticks;
        Academy.Instance.EnvironmentStep();
        var afterTicks = DateTime.Now.Ticks;
        // Update
        {
            var elapsedMs = s_TicksToMilliseconds * (afterTicks - startTicks);
            m_RunningCount[(int)Mode]++;
            if (m_RunningCount[(int)Mode] == 1)
            {
                m_RunningAverageMs[(int)Mode] = elapsedMs;
            }
            else
            {
                m_RunningAverageMs[(int)Mode] = m_RunningAverageMs[(int)Mode] + (elapsedMs - m_RunningAverageMs[(int)Mode]) / m_RunningCount[(int)Mode];
            }
        }


        StepCount++;
        if (StepCount % 1000 == 0)
        {
            // Switch mode
            Mode = (Mode == SteppingMode.StepAllAgentsEvery5Frames)
                ? SteppingMode.Step2AgentsEachFrame
                : SteppingMode.StepAllAgentsEvery5Frames;

            Debug.Log($"Average Academy Step:" +
                      $"\n\tStep2AgentsEachFrame: {m_RunningAverageMs[(int)SteppingMode.Step2AgentsEachFrame]} ms" +
                      $"\n\tStepAllAgentsEvery5Frames: {m_RunningAverageMs[(int)SteppingMode.StepAllAgentsEvery5Frames]} ms"
                      );
        }

    }

}

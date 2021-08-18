using Math = System.Math;
using System.Linq;
using UnityEngine;
using Unity.MLAgents.Sensors;

/// <summary>
/// A sensor that samples diversity settings and writes them as goal observations
/// </summary>
public class DiversitySampler : ISensor
{
    // These members need to be changed by the editor at runtime
    public bool DisableSampling
    {
        get { return m_DisableSampling; }
        set { m_DisableSampling = value; }
    }
    public int DiscreteSetting
    {
        get
        {
            if (!m_ContinuousSampling)
            {
                return m_DiversitySetting.ToList().IndexOf(1);
            }
            else
            {
                return -1;
            }
        }
        set
        {
            if (!m_ContinuousSampling)
            {
                for (int i = 0; i < m_DiversitySize; i++)
                {
                    m_DiversitySetting[i] = i == value ? 1 : 0;
                }
            }
        }
    }
    public float[] ContinuousSetting
    {
        get
        {
            if (m_ContinuousSampling)
            {
                return m_DiversitySetting;
            }
            else
            {
                return null;
            }
        }
        set
        {
            if (m_ContinuousSampling)
            {
                m_DiversitySetting = value;
            }
        }
    }

    string m_SensorName;
    int m_DiversitySize;
    bool m_ContinuousSampling;
    bool m_BalancedSampling;
    bool m_DisableSampling = false;
    ObservationSpec m_ObservationSpec;
    float[] m_SamplingWeights;
    float[] m_DiversitySetting;

    public DiversitySampler(string name, int size, bool continuous, bool balanced)
    {
        m_SensorName = name;
        m_DiversitySize = size;
        m_ContinuousSampling = continuous;
        m_BalancedSampling = balanced;

        m_ObservationSpec = ObservationSpec.Vector(m_DiversitySize, ObservationType.GoalSignal);
        m_DiversitySetting = new float[m_DiversitySize];

        if (m_BalancedSampling)
        {
            m_SamplingWeights = new float[m_DiversitySize];
            for (int i = 0; i < m_DiversitySize; i++)
            {
                m_SamplingWeights[i] = 1f / m_DiversitySize;
            }
        }

        Reset();
    }

    /// <summary>
    /// Samples a new diversity vector
    /// </summary>
    public void Reset() 
    {
        if (m_DisableSampling)
        {
            return;
        }

        if (m_ContinuousSampling)
        {
            for (int i = 0; i < m_DiversitySize; i++)
            {
                ContinuousSetting[i] = Random.Range(-1f, 1f);
            }
        }
        else
        {
            float[] probs;
            if (m_BalancedSampling)
            {
                probs = WeightsToProbs(m_SamplingWeights);
                float rand = Random.Range(0f, 1f);
                for (int i = 0; i < m_DiversitySize; i++)
                {
                    if (rand < probs[i])
                    {
                        DiscreteSetting = i;
                        break;
                    }
                    rand -= probs[i];
                }
            }
            else
            {
                DiscreteSetting = Random.Range(0, m_DiversitySize);
            }
        }
    }

    private float[] WeightsToProbs(float[] weights)
    {
        float den = 0;
        foreach (float w in weights)
        {
            den += (float)Math.Exp(w);
        }

        float[] probs = new float[m_DiversitySize];
        for (int i = 0; i < m_DiversitySize; i++)
        {
            probs[i] = (float)Math.Exp(weights[i]) / den;
        }

        return probs;
    }

    /// <summary>
    /// Updates sampling weights
    /// </summary>
    public void Update() 
    {
        if (m_BalancedSampling && !m_ContinuousSampling)
        {
            for (int i = 0; i < m_DiversitySize; i++)
            {
                float modifier = i == DiscreteSetting ? -0.01f : 0.01f / (m_DiversitySize - 1);
                m_SamplingWeights[i] = m_SamplingWeights[i] + modifier;
                // m_SamplingWeights[i] = 0.999f * m_SamplingWeights[i] + 0.001f * (i == DiscreteSetting ? 0f : 1f);
            }
        }
    }

    /// <inheritdoc/>
    public virtual int Write(ObservationWriter writer)
    {
        writer.AddList(m_DiversitySetting);
        return m_DiversitySize;
    }

    /// <inheritdoc/>
    public string GetName()
    {
        return m_SensorName;
    }

    /// <inheritdoc/>
    public ObservationSpec GetObservationSpec()
    {
        return m_ObservationSpec;
    }

    /// <inheritdoc/>
    public virtual byte[] GetCompressedObservation()
    {
        return null;
    }

    /// <inheritdoc/>
    public virtual CompressionSpec GetCompressionSpec()
    {
        return CompressionSpec.Default();
    }
}

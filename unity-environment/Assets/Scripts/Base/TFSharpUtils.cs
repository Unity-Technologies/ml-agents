using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

using TensorFlow;


[Serializable]
public class TFOutputHolder
{
    [SerializeField]
    private string name;

    public string Name { get { return name; } private set { name = value; } }
    public TFOutput Output { get; private set; }
    public TFGraph Graph { get; private set; } = null;
    public bool Valid { get; private set; } = false;

    public TFOutputHolder()
    {
    }
    public TFOutputHolder(string nodeName)
    {
        Name = nodeName;
    }

    public bool Initialize(TFGraph graph)
    {
        Graph = graph;
        TFOperation op = graph[Name];
        if (op != null)
        {
            Output = op[0];
            Valid = true;
        }
        else
        {
            Debug.LogWarning("TFOutput " + Name + " does not exist.");
        }

        return Valid;
    }
}



public class TFOperationHolder
{
    public string Name { get; private set; } = null;
    public TFOperation Operation { get; private set; }
    public TFGraph Graph { get; private set; } = null;
    public bool Valid { get; private set; } = false;

    public TFOperationHolder(string name)
    {
        Name = name;
    }

    public bool Initialize(TFGraph graph)
    {
        Graph = graph;
        Operation = graph[Name];
        if (Operation != null)
        {
            Valid = true;
        }
        else
        {
            Debug.LogWarning("TFOperation " + Name + " does not exist.");
        }

        return Valid;
    }
}


[Serializable]
public struct LearningRateAndMomentum
{
    public float startLearningRate;
    public float endLearningRate;
    public int changeLRSteps;
    public float startInvMomentum;
    public float endInvMomentum;
    public MathUtils.InterpolateMethod momInterpMethod;
    public MathUtils.InterpolateMethod lrInterpMethod;
    public bool modifyLR;

    public void CalculateMomentumAndLR(int step, out float mom, out float lr)
    {

        float t = (Mathf.Clamp01((float)(step) / (float)changeLRSteps));

        lr = MathUtils.Interpolate(startLearningRate, endLearningRate, t, lrInterpMethod);
        mom = MathUtils.Interpolate(startInvMomentum, endInvMomentum, t, momInterpMethod);
        mom = 1 - 1 / mom;
        if (mom < 0)
            mom = -1;
        if (modifyLR && mom >= 0 && mom < 1)
        {
            lr = lr * (1 - mom);
        }
    }
}

/// <summary>
/// helps to get the average of data
/// </summary>
public class AutoAverage
{
    private int interval;
    public int Interval
    {
        get { return interval; }
        set { interval = Mathf.Max(value, 1); }
    }

    public float Average
    {
        get
        {
            return lastAverage;
        }
    }

    public bool JustUpdated
    {
        get; private set;
    }

    private float lastAverage = 0;
    private int currentCount = 0;
    private float sum = 0;

    public AutoAverage(int interval = 1)
    {
        Interval = interval;
        JustUpdated = false;
    }

    public void AddValue(float value)
    {
        sum += value;
        currentCount += 1;
        JustUpdated = false;
        if (currentCount >= Interval)
        {
            lastAverage = sum / currentCount;
            currentCount = 0;
            sum = 0;
            JustUpdated = true;
        }
    }


}

public static class MathUtils
{
    public enum InterpolateMethod
    {
        Linear,
        Log
    }

    /// <summary>
    /// interpolate between x1 and x2 to ty suing the interpolate method
    /// </summary>
    /// <param name="method"></param>
    /// <param name="x1"></param>
    /// <param name="x2"></param>
    /// <param name="t"></param>
    /// <returns></returns>
    public static float Interpolate(float x1, float x2, float t, InterpolateMethod method = InterpolateMethod.Linear)
    {
        if (method == InterpolateMethod.Linear)
        {
            return Mathf.Lerp(x1, x2, t);
        }
        else
        {
            return Mathf.Pow(x1, 1 - t) * Mathf.Pow(x2, t);
        }
    }

    /// <summary>
    /// Return a index randomly. The probability if a index depends on the value in that list
    /// </summary>
    /// <param name="list"></param>
    /// <returns></returns>
    public static int IndexByChance(float[] list)
    {
        float total = 0;

        foreach (var v in list)
        {
            total += v;
        }
        Debug.Assert(total > 0);

        float current = 0;
        System.Random r = new System.Random();
        float point = (float)r.NextDouble() * total;

        for (int i = 0; i < list.Length; ++i)
        {
            current += list[i];
            if (current / total >= point)
            {
                return i;
            }
        }
        return 0;
    }
    /// <summary>
    /// return the index of the max value in the list
    /// </summary>
    /// <param name="list"></param>
    /// <returns></returns>
    public static int IndexMax(float[] list)
    {
        int result = 0;
        for (int i = 1; i < list.Length; ++i)
        {
            if (list[i - 1] < list[i])
            {
                result = i;
            }
        }
        return result;
    }

    /// <summary>
    /// Shuffle a list
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="list"></param>
    /// <param name="rnd"></param>
    public static void Shuffle<T>(IList<T> list, System.Random rnd)
    {
        int n = list.Count;
        while (n > 1)
        {

            n--;
            int k = rnd.Next(0, n + 1);
            T value = list[k];
            list[k] = list[n];
            list[n] = value;
        }
    }
}